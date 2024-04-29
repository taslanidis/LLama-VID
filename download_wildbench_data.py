from datasets import load_dataset

dataset = load_dataset("liuhaotian/llava-bench-in-the-wild")
dataset.save_to_disk("llava-bench-in-the-wild")