# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")

model.save_pretrained("/scratch-shared/scur0405/model_zoo/LLM/vicuna/7B-V1.5")
tokenizer.save_pretrained("/scratch-shared/scur0405/model_zoo/LLM/vicuna/7B-V1.5")