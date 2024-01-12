import bitsandbytes as bnb
import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)

from transformers import(
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import os
peft_model = "model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


config = PeftConfig.from_pretrained(os.path.abspath("models"))
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="cpu",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token=tokenizer.eos_token

model = PeftModel.from_pretrained(model,os.path.abspath("models"))

generation_config = model.generation_config
generation_config.max_new_token = 1000
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

def generate_responses(question:str)->str:
    prompt=f"""
    <human>:{question}
    <assistant>:
    """.strip()

    encoding = tokenizer(prompt,return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config
        )

        response = tokenizer.decode(out[0],skip_special_tokens=True)

        assistant_start="<assistant>:"
        response_start = response.find(assistant_start)
        return response[response_start + len(assistant_start) :].strip()

generate_responses("What payment methods do you accept?")