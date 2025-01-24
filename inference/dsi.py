import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

mdn = "meta-llama/Llama-3.2-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(mdn)

ds_engine = deepspeed.init_inference(model,
                                 tensor_parallel={"tp_size": 2},
                                 dtype=torch.half,
                                 checkpoint=None,
                                 replace_with_kernel_inject=True)
model = ds_engine.module
output = model('Input String')