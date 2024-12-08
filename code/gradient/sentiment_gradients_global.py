import numpy as np
from huggingface_hub import login
import os
import transformers
import torch
from tqdm import tqdm


# CAV = torch.from_numpy(np.load('../../results/short/last_token/short_model_parameters_70B_quant.npz')['coefficients']) # CAV
CAV = torch.from_numpy(np.load('../../results/short/all_tokens/short_model_parameters_70B_quant.npz')['coefficients']) # CAV
# tokens_of_interest = np.load('../../data/1000_tokens.npy')
tokens_of_interest = np.load('../../data/sentiment_tokens.npy')


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

def create_hook_fn(dir_derivs_token):
    def hook_fn(module, grad_input, grad_output):
        gradient = grad_output[0].squeeze(0)[-1].detach().to(torch.float32)
        dir_deriv = torch.dot(gradient, CAV.to(gradient.device))
        dir_derivs_token.append(dir_deriv.cpu())
    return hook_fn


prompt = "<|start_header_id|>system<|end_header_id|>\r\n"\
    "\r\n"\
    "Cutting Knowledge Date: December 2023\r\n"\
    "Today Date: 1 January 2024\r\n"\
    "\r\n"\
    "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\r\n"\
    "\r\n"\
    f"In one sentence, write a negative review of the movie Birdman.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
print(prompt, flush=True)
dir_derivs_example = []
generated_tokens_example = []

input = tokenizer(prompt, return_tensors="pt").to(model.device)

for i in range(128): # PREDICTED TOKENS
    dir_derivs_token = []
    hook = layer.register_full_backward_hook(create_hook_fn(dir_derivs_token))
    
    logits = model(**input).logits.squeeze(0)[-1]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    for token in tqdm(tokens_of_interest, desc=f"Processing token {i}"):
        model.zero_grad()
        log_prob = torch.log(probabilities[token])
        print(log_prob, flush=True)
        log_prob.backward(retain_graph=True)
    hook.remove()

    next_token = torch.argmax(probabilities)
    input['input_ids'] = torch.cat([input['input_ids'], next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

    dir_derivs_example.append(dir_derivs_token)
    generated_tokens_example.append(tokenizer.decode(next_token.item()))

    # np.savez('../../results/short/last_token/short_gradients_70B_quant', **dir_derivs_examples) # SAVE
    # np.savez('../../results/short/last_token/short_generated_tokens_70B_quant', **generated_tokens_examples) # SAVE
    # np.save('../../results/short/all_tokens/short_gradients_global_top_70B_quant', np.array(dir_derivs_example)) # SAVE
    np.save('../../results/short/all_tokens/short_gradients_global_sentiment_negative_70B_quant', np.array(dir_derivs_example)) # SAVE
    np.save('../../results/short/all_tokens/short_generated_tokens_global_negative_70B_quant', np.array(generated_tokens_example)) # SAVE
    
    print(generated_tokens_example, flush=True)

    if next_token == 128009:
        break
