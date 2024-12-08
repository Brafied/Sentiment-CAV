import json
from huggingface_hub import login
import os
import transformers
import torch
from tqdm import tqdm
import pickle


with open('data/movies.json', 'r') as file:
    movies = json.load(file)


login(os.getenv('HF_TOKEN'))

quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct" # MODEL
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config # QUANTIZATION
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)


tokens_examples = []
tokens_examples_labels = []
for example in tqdm(movies, desc="Processing movies"):
    for sentiment in ['negative', 'positive']:
        prompt = "<|start_header_id|>system<|end_header_id|>\r\n"\
            "\r\n"\
            "Cutting Knowledge Date: December 2023\r\n"\
            "Today Date: 1 January 2024\r\n"\
            "\r\n"\
            "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\r\n"\
            "\r\n"\
            f"In one sentence, write a {sentiment} review of the movie {example}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        print(prompt, flush=True)

        generated_tokens_example = []

        input = tokenizer(prompt, return_tensors="pt").to(model.device)

        for _ in range(1024):
            with torch.no_grad():
                logits = model(**input).logits.squeeze(0)[-1]
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.argmax(probabilities)    
            input['input_ids'] = torch.cat([input['input_ids'], next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
            generated_tokens_example.append(tokenizer.decode(next_token.item()))
            if next_token == 128009:
                tokens_examples.append(input)
                tokens_examples_labels.append(0 if sentiment == 'negative' else 1)
                break
        print(generated_tokens_example, flush=True)
        with open('results/synthetic_short_reviews_70B_quant.pkl', 'wb') as f: # SAVE
            pickle.dump(tokens_examples, f)
        with open('results/synthetic_short_reviews_labels_70B_quant.pkl', 'wb') as f: # SAVE
            pickle.dump(tokens_examples_labels, f) 
        