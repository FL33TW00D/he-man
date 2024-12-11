from .model import Model

from typing import Dict, List, Union
import coremltools as ct
import torch
import numpy as np
import os
import transformers

class DistilBert(Model):
    def name():
        return "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    def recommended_iterations(self) -> int:
        return 1000
    
    def __init__(self):
        super().__init__()

        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            DistilBert.name(),
            return_dict=False,
            torchscript=True,
        ).eval()

        original_tokenizer_parallel = os.environ["TOKENIZERS_PARALLELISM"] if "TOKENIZERS_PARALLELISM" in os.environ else "true"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(DistilBert.name())
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
        return (self.tokenized["input_ids"], self.tokenized["attention_mask"], )

    def coreml_inputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        return [
            ct.TensorType(
                f"input_{name}",
                    shape=tensor.shape,
                    dtype=np.int32,
                ) for name, tensor in self.tokenized.items()
            ]

    def coreml_outputs(self) -> List[Union[ct.TensorType, ct.ImageType]]:
        return None