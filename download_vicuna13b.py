from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.5")

model.save_pretrained("model_zoo/LLM/vicuna/13B-V1.5")
tokenizer.save_pretrained("model_zoo/LLM/vicuna/13B-V1.5")