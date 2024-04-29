from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("YanweiLi/llama-vid-7b-full-336")
model = AutoModelForCausalLM.from_pretrained("YanweiLi/llama-vid-7b-full-336")

model.save_pretrained("work_dirs/tmp")
tokenizer.save_pretrained("work_dirs/tmp")