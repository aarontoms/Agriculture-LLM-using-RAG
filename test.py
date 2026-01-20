import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "MeelUnv/llama-3-8b-chatbot-agriculture_"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

prompt = "What are the common pests affecting rice crops and how can they be controlled?"

out = gen(
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

print(out[0]["generated_text"])
