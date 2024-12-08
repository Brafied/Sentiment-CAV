import json
import numpy as np
from huggingface_hub import login
import os
import transformers
import torch
from tqdm import tqdm


with open('../../data/movies_test.json', 'r') as file:
    movies = json.load(file)
# CAV = torch.from_numpy(np.load('../../results/short/last_token/short_model_parameters_70B_quant.npz')['coefficients']) # CAV
CAV = torch.from_numpy(np.load('../../results/short/all_tokens/short_model_parameters_70B_quant.npz')['coefficients']) # CAV


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


layer = model.model.layers[-1].mlp.act_fn

def create_hook_fn(dir_derivs_example):
    def hook_fn(module, grad_input, grad_output):
        gradient = grad_output[0].squeeze(0)[-1].detach().to(torch.float32)
        dir_deriv = torch.dot(gradient, CAV.to(gradient.device))
        dir_derivs_example.append(dir_deriv.cpu())
    return hook_fn


dir_derivs_examples = {}
generated_tokens_examples = {}
for i, example in tqdm(enumerate(movies), desc="Processing movies"):
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
        dir_derivs_example = []
        generated_tokens_example = []
        hook = layer.register_full_backward_hook(create_hook_fn(dir_derivs_example))

        input = tokenizer(prompt, return_tensors="pt").to(model.device)

        for _ in range(128): # PREDICTED TOKENS
            model.zero_grad()
            logits = model(**input).logits.squeeze(0)[-1]
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.argmax(probabilities)
            log_prob = torch.log(probabilities[next_token])        
            input['input_ids'] = torch.cat([input['input_ids'], next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
            log_prob.backward()
            generated_tokens_example.append(tokenizer.decode(next_token.item()))
            if next_token == 128009:
                dir_derivs_examples[f'example_{sentiment}_{i}'] = dir_derivs_example
                generated_tokens_examples[f'example_{sentiment}_{i}'] = generated_tokens_example
                break
        print(generated_tokens_example, flush=True)
        hook.remove()
        # np.savez('../../results/short/last_token/short_gradients_70B_quant.npz', **dir_derivs_examples) # SAVE
        # np.savez('../../results/short/last_token/short_generated_tokens_70B_quant.npz', **generated_tokens_examples) # SAVE
        np.savez('../../results/short/all_tokens/short_gradients_70B_quant.npz', **dir_derivs_examples) # SAVE
        np.savez('../../results/short/all_tokens/short_generated_tokens_70B_quant.npz', **generated_tokens_examples) # SAVE
