import pickle
from huggingface_hub import login
import os
import transformers
import torch
import numpy as np
from tqdm import tqdm


with open('../../results/short/synthetic_short_reviews_70B_quant.pkl', 'rb') as f:
    reviews = pickle.load(f)


login(os.getenv('HF_TOKEN'))

quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct" # MODEL
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


review_activations = []
def hook_fn(module, input, output):
    tokens = output[0, 42:, :].detach() # System prompts diverge at the 42nd token.
    review_activations.append(tokens.cpu().to(torch.float32).numpy())
    return output


layer = model.model.layers[-1].mlp.act_fn
hook = layer.register_forward_hook(hook_fn)

for input in tqdm(reviews, desc="Processing review batches"):
    with torch.no_grad():
        model(**input)

review_activations_np = np.array(review_activations,  dtype=object)
np.save('../../results/short/all_tokens/short_review_activations_70B_quant.npy', review_activations_np) # SAVE

hook.remove()
