from .model import Model

import os
import torch
from typing import Optional, Union, List, Dict, Tuple
import numpy as np
import coremltools as ct

from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import (
    MISTRAL_ATTENTION_CLASSES,
    MistralAttention,
    MistralConfig,
    MistralForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers import AutoTokenizer


class SliceUpdateKeyValueCache(Cache):
    def __init__(
        self,
        shape: Tuple[int, ...],
        device="cpu",
        dtype=torch.float32,
    ) -> None:
        """KV cache of shape (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        slice_indices: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update key/value cache tensors for slice [slice_indices[0], slice_indices[1]).
        Return slice of key/value cache tensors from [0, slice_indices[1]).
        """
        if len(slice_indices) != 2:
            raise ValueError(
                f"Expect tuple of integers [start, end), got {slice_indices=}."
            )
        begin, end = slice_indices
        self.k_cache[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state
        self.v_cache[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state
        k_cache: torch.Tensor = self.k_cache[layer_idx, :, :, :end, :]
        v_cache: torch.Tensor = self.v_cache[layer_idx, :, :, :end, :]
        return k_cache, v_cache

    def get_seq_length(self, _: int | None = 0) -> int:
        """Get the sequence length of the cache."""
        return self.past_seen_tokens


class SliceUpdateMistralAttention(MistralAttention):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor | None, ...]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Slice update key/value cache
        end_step = attention_mask.shape[-1]
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            slice_indices=(end_step - q_len, end_step),
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None


class StatefulMistralForCausalLM(torch.nn.Module):
    def __init__(
        self, model_path: str, max_context_size: int = 2048, batch_size: int = 1
    ) -> None:
        super().__init__()

        # Custom attention implementation for stateful slice update key/value cache, override
        # "sdpa" to compliance with transformers.modeling_utils._autoset_attn_implementation
        MISTRAL_ATTENTION_CLASSES["sdpa"] = SliceUpdateMistralAttention
        self.model = MistralForCausalLM.from_pretrained(model_path)

        # Register KV cache buffers to be recognized as Core ML states
        config: MistralConfig = self.model.config
        self.kv_cache_shape: Tuple[int, ...] = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_context_size,
            config.hidden_size // config.num_attention_heads,
        )
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)
        self.register_buffer("keyCache", self.kv_cache.k_cache)
        self.register_buffer("valueCache", self.kv_cache.v_cache)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute past seen tokens used for updating key/value cache slices
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits


class Mistral7B(Model):
    @staticmethod
    def name():
        return "mistralai/Mistral-7B-Instruct-v0.3"

    def recommended_iterations(self) -> int:
        return 1000

    def __init__(self):
        super().__init__()
        self.max_context_size = 2048

        self.model = StatefulMistralForCausalLM(
            Mistral7B.name(), max_context_size=self.max_context_size
        ).eval()

        original_tokenizer_parallel = (
            os.environ["TOKENIZERS_PARALLELISM"]
            if "TOKENIZERS_PARALLELISM" in os.environ
            else "true"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = AutoTokenizer.from_pretrained(Mistral7B.name())
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenized = self.tokenizer(
            ["Sample input text to trace the model"],
            return_tensors="pt",
            max_length=128,  # token sequence length
            padding="max_length",
        )
        os.environ["TOKENIZERS_PARALLELISM"] = original_tokenizer_parallel

    def torch_example_input(
        self,
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        input_ids: torch.Tensor = torch.zeros((1, 2), dtype=torch.int32)
        causal_mask: torch.Tensor = torch.zeros((1, 1, 2, 5), dtype=torch.float32)

        return (input_ids, causal_mask)

    def coreml_inputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        query_length = ct.RangeDim(
            lower_bound=1, upper_bound=self.max_context_size, default=1
        )
        end_step_dim = ct.RangeDim(
            lower_bound=1, upper_bound=self.max_context_size, default=1
        )
        return [
            ct.TensorType(shape=(1, query_length), dtype=np.int32, name="input_ids"),
            ct.TensorType(
                shape=(1, 1, query_length, end_step_dim),
                dtype=np.float16,
                name="attention_mask",
            ),
        ]

    def coreml_states(self) -> Optional[List[ct.StateType]]:
        kv_cache_shape = (32, 1, 8, self.max_context_size, 128)
        return [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=kv_cache_shape, dtype=np.float16),
                name="keyCache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=kv_cache_shape, dtype=np.float16),
                name="valueCache",
            ),
        ]

    def coreml_outputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        return [ct.TensorType(dtype=np.float16, name="logits")]
