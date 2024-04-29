from transformers import AutoTokenizer
from llamavid.model import LlavaLlamaAttForCausalLM
import torch

kwargs = {}
model_path = "YanweiLi/llama-vid-7b-full-336"
kwargs['torch_dtype'] = torch.float16
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = LlavaLlamaAttForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

model.save_pretrained("work_dirs/llama-vid/llama-vid-7b-full-336")
tokenizer.save_pretrained("work_dirs/llama-vid/llama-vid-7b-full-336")