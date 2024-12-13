from .distilbert import DistilBert

from typing import Dict, List, Union
import coremltools as ct
import torch
import torch.nn as nn
import numpy as np
from transformers.models.distilbert import modeling_distilbert


class LayerNormANE(nn.Module):
    """LayerNorm optimized for Apple Neural Engine (ANE) execution

    Note: This layer only supports normalization over the final dim. It expects `num_channels`
    as an argument and not `normalized_shape` which is used by `torch.nn.LayerNorm`.
    """

    def __init__(self, num_channels, clip_mag=None, eps=1e-5, elementwise_affine=True):
        """
        Args:
            num_channels:       Number of channels (C) where the expected input data format is BC1S. S stands for sequence length.
            clip_mag:           Optional float value to use for clamping the input range before layer norm is applied.
                                If specified, helps reduce risk of overflow.
            eps:                Small value to avoid dividing by zero
            elementwise_affine: If true, adds learnable channel-wise shift (bias) and scale (weight) parameters
        """
        super().__init__()
        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        self.expected_rank = len("BC1S")

        self.num_channels = num_channels
        self.eps = eps
        self.clip_mag = clip_mag
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, inputs):
        input_rank = len(inputs.size())

        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        # Migrate the data format from BSC to BC1S (most conducive to ANE)
        if input_rank == 3 and inputs.size(2) == self.num_channels:
            inputs = inputs.transpose(1, 2).unsqueeze(2)
            input_rank = len(inputs.size())

        assert input_rank == self.expected_rank
        assert inputs.size(1) == self.num_channels

        if self.clip_mag is not None:
            inputs.clamp_(-self.clip_mag, self.clip_mag)

        channels_mean = inputs.mean(dim=1, keepdims=True)

        zero_mean = inputs - channels_mean

        zero_mean_sq = zero_mean * zero_mean

        denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()

        out = zero_mean * denom

        if self.elementwise_affine:
            out = (out + self.bias.view(1, self.num_channels, 1, 1)) * self.weight.view(
                1, self.num_channels, 1, 1
            )

        return out


# Note: Original implementation of distilbert uses an epsilon value of 1e-12
# which is not friendly with the float16 precision that ANE uses by default
EPS = 1e-7

WARN_MSG_FOR_TRAINING_ATTEMPT = (
    "This model is optimized for on-device execution only. "
    "Please use the original implementation from Hugging Face for training"
)

WARN_MSG_FOR_DICT_RETURN = (
    "coremltools does not support dict outputs. Please set return_dict=False"
)


# Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
# apply scale and bias terms in opposite orders. In order to accurately restore a
# state_dict trained using the former into the the latter, we adjust the bias term
def correct_for_bias_scale_order_inversion(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    state_dict[prefix + "bias"] = (
        state_dict[prefix + "bias"] / state_dict[prefix + "weight"]
    )
    return state_dict


class LayerNormANE(LayerNormANE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(correct_for_bias_scale_order_inversion)


class Embeddings(modeling_distilbert.Embeddings):
    """Embeddings module optimized for Apple Neural Engine"""

    def __init__(self, config):
        super().__init__(config)
        setattr(self, "LayerNorm", LayerNormANE(config.dim, eps=EPS))


class MultiHeadSelfAttention(modeling_distilbert.MultiHeadSelfAttention):
    """MultiHeadSelfAttention module optimized for Apple Neural Engine"""

    def __init__(self, config):
        super().__init__(config)

        setattr(
            self,
            "q_lin",
            nn.Conv2d(
                in_channels=config.dim,
                out_channels=config.dim,
                kernel_size=1,
            ),
        )

        setattr(
            self,
            "k_lin",
            nn.Conv2d(
                in_channels=config.dim,
                out_channels=config.dim,
                kernel_size=1,
            ),
        )

        setattr(
            self,
            "v_lin",
            nn.Conv2d(
                in_channels=config.dim,
                out_channels=config.dim,
                kernel_size=1,
            ),
        )

        setattr(
            self,
            "out_lin",
            nn.Conv2d(
                in_channels=config.dim,
                out_channels=config.dim,
                kernel_size=1,
            ),
        )

    def prune_heads(self, heads):
        raise NotImplementedError

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, dim, 1, seq_length)
            key: torch.tensor(bs, dim, 1, seq_length)
            value: torch.tensor(bs, dim, 1, seq_length)
            mask: torch.tensor(bs, seq_length) or torch.tensor(bs, seq_length, 1, 1)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            dim, 1, seq_length) Contextualized layer. Optional: only if `output_attentions=True`
        """
        # Parse tensor shapes for source and target sequences
        assert (
            len(query.size()) == 4 and len(key.size()) == 4 and len(value.size()) == 4
        )

        bs, dim, dummy, seqlen = query.size()
        # assert seqlen == key.size(3) and seqlen == value.size(3)
        # assert dim == self.dim
        # assert dummy == 1

        # Project q, k and v
        q = self.q_lin(query)
        k = self.k_lin(key)
        v = self.v_lin(value)

        # Validate mask
        if mask is not None:
            expected_mask_shape = [bs, seqlen, 1, 1]
            if mask.dtype == torch.bool:
                mask = mask.logical_not().float() * -1e4
            elif mask.dtype == torch.int64:
                mask = (1 - mask).float() * -1e4
            elif mask.dtype != torch.float32:
                raise TypeError(f"Unexpected dtype for mask: {mask.dtype}")

            if len(mask.size()) == 2:
                mask = mask.unsqueeze(2).unsqueeze(2)

            if list(mask.size()) != expected_mask_shape:
                raise RuntimeError(
                    f"Invalid shape for `mask` (Expected {expected_mask_shape}, got {list(mask.size())}"
                )

        if head_mask is not None:
            raise NotImplementedError

        # Compute scaled dot-product attention
        dim_per_head = self.dim // self.n_heads
        mh_q = q.split(
            dim_per_head, dim=1
        )  # (bs, dim_per_head, 1, max_seq_length) * n_heads
        mh_k = k.transpose(1, 3).split(
            dim_per_head, dim=3
        )  # (bs, max_seq_length, 1, dim_per_head) * n_heads
        mh_v = v.split(
            dim_per_head, dim=1
        )  # (bs, dim_per_head, 1, max_seq_length) * n_heads

        normalize_factor = float(dim_per_head) ** -0.5
        attn_weights = [
            torch.einsum("bchq,bkhc->bkhq", [qi, ki]) * normalize_factor
            for qi, ki in zip(mh_q, mh_k)
        ]  # (bs, max_seq_length, 1, max_seq_length) * n_heads

        if mask is not None:
            for head_idx in range(self.n_heads):
                attn_weights[head_idx] = attn_weights[head_idx] + mask

        attn_weights = [
            aw.softmax(dim=1) for aw in attn_weights
        ]  # (bs, max_seq_length, 1, max_seq_length) * n_heads
        attn = [
            torch.einsum("bkhq,bchk->bchq", wi, vi)
            for wi, vi in zip(attn_weights, mh_v)
        ]  # (bs, dim_per_head, 1, max_seq_length) * n_heads

        attn = torch.cat(attn, dim=1)  # (bs, dim, 1, max_seq_length)

        attn = self.out_lin(attn)

        if output_attentions:
            return attn, attn_weights.cat(dim=2)
        else:
            return (attn,)


class FFN(modeling_distilbert.FFN):
    """FFN module optimized for Apple Neural Engine"""

    def __init__(self, config):
        super().__init__(config)
        self.seq_len_dim = 3

        setattr(
            self,
            "lin1",
            nn.Conv2d(
                in_channels=config.dim,
                out_channels=config.hidden_dim,
                kernel_size=1,
            ),
        )

        setattr(
            self,
            "lin2",
            nn.Conv2d(
                in_channels=config.hidden_dim,
                out_channels=config.dim,
                kernel_size=1,
            ),
        )


class TransformerBlock(modeling_distilbert.TransformerBlock):
    def __init__(self, config):
        super().__init__(config)
        setattr(self, "attention", MultiHeadSelfAttention(config))
        setattr(self, "sa_layer_norm", LayerNormANE(config.dim, eps=EPS))
        setattr(self, "ffn", FFN(config))
        setattr(self, "output_layer_norm", LayerNormANE(config.dim, eps=EPS))


class Transformer(modeling_distilbert.Transformer):
    def __init__(self, config):
        super().__init__(config)
        setattr(
            self,
            "layer",
            nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
        )


class DistilBertModel(modeling_distilbert.DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        setattr(self, "embeddings", Embeddings(config))
        setattr(self, "transformer", Transformer(config))

        # Register hook for unsqueezing nn.Linear parameters to match nn.Conv2d parameter spec
        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

        self._use_sdpa = False

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError


class DistilBertForSequenceClassification(
    modeling_distilbert.DistilBertForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)
        setattr(self, "distilbert", DistilBertModel(config))
        setattr(self, "pre_classifier", nn.Conv2d(config.dim, config.dim, 1))
        setattr(self, "classifier", nn.Conv2d(config.dim, config.num_labels, 1))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if labels is not None or self.training:
            raise NotImplementedError(WARN_MSG_FOR_TRAINING_ATTEMPT)

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if return_dict:
            raise ValueError(WARN_MSG_FOR_DICT_RETURN)

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        hidden_state = distilbert_output[0]  # (bs, dim, 1, seq_len)
        pooled_output = hidden_state[:, :, :, 0:1]  # (bs, dim, 1, 1)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim, 1, 1)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim, 1, 1)
        logits = self.classifier(pooled_output)  # (bs, num_labels, 1, 1)
        logits = logits.squeeze(-1).squeeze(-1)  # (bs, num_labels)

        output = (logits,) + distilbert_output[1:]
        loss = None

        return ((loss,) + output) if loss is not None else output


def linear_to_conv2d_map(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights"""
    for k in state_dict:
        is_internal_proj = all(substr in k for substr in ["lin", ".weight"])
        is_output_proj = all(substr in k for substr in ["classifier", ".weight"])
        if is_internal_proj or is_output_proj:
            if len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]


class DistilBertANE(DistilBert):
    @staticmethod
    def name():
        return super(DistilBertANE, DistilBertANE).name() + "(ANE)"

    def recommended_iterations(self) -> int:
        return 1000

    def __init__(self):
        super().__init__()

        self.baseline_model = self.model

        self.model = DistilBertForSequenceClassification(
            self.baseline_model.config
        ).eval()

        state_dict = self.baseline_model.state_dict()

        if "pre_classifier.weight" in state_dict:
            state_dict["pre_classifier.weight"] = (
                state_dict["pre_classifier.weight"].unsqueeze(-1).unsqueeze(-1)
            )

        # Adjust weights for `classifier`
        if "classifier.weight" in state_dict:
            state_dict["classifier.weight"] = (
                state_dict["classifier.weight"].unsqueeze(-1).unsqueeze(-1)
            )

        self.model.load_state_dict(state_dict)

    def torch_example_input(
        self,
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        return (
            self.tokenized["input_ids"],
            self.tokenized["attention_mask"],
        )

    def coreml_inputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        return [
            ct.TensorType(
                f"input_{name}",
                shape=tensor.shape,
                dtype=np.int32,
            )
            for name, tensor in self.tokenized.items()
        ]

    def coreml_outputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        return None
