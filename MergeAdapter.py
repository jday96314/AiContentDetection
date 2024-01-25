from transformers import AutoModelForCausalLM
from peft import PeftModel
import shutil
import os

# BASE_MODEL = "mistralai/Mistral-7B-v0.1"
# ADAPTER_PATH = "Models/StudentImitators/single_epoch_fold_1/final_model"

# BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# ADAPTER_PATH = "Models/StudentImitators/1e-4_2_Fold1/final_model"

# BASE_MODEL = "TheBloke/Llama-2-13B-fp16"
# ADAPTER_PATH = "Models/StudentImitators/Llama-2-13B-fp16_8e-4_1_Fold1/final_model"

BASE_MODEL = "OpenPipe/mistral-ft-optimized-1218"
ADAPTER_PATH = "Models/StudentImitators/mistral-ft-optimized-1e-4_2_Fold1/final_model"

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

merged_model = peft_model.merge_and_unload()

MERGED_MODEL_PATH = ADAPTER_PATH + '_merged'
merged_model.save_pretrained(MERGED_MODEL_PATH)

# Copy tokenizer from base model to merged model.
for filename in os.listdir(ADAPTER_PATH):
    if 'tokenizer' in filename:
        shutil.copyfile(f"{ADAPTER_PATH}/{filename}", f"{MERGED_MODEL_PATH}/{filename}")
