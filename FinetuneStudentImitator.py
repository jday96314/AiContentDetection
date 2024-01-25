from unsloth import FastMistralModel, FastLlamaModel
import torch

from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.utils import logging
from sklearn.model_selection import KFold
from datasets import Dataset

import pandas as pd

# TODO: Try the following foundation models:
#   - https://huggingface.co/OpenPipe/mistral-ft-optimized-1218
#   - https://huggingface.co/NousResearch/Nous-Hermes-13b


def PrepareDataset():
    human_essays = pd.read_csv('data/PERSUADE/persuade_2.0_human_scores_demo_id_github.csv')
    # human_essays['assignments_and_essays'] = human_essays.apply(lambda row: f'### Assignment: {row["assignment"]}\n### Essay: {row["full_text"]}', axis=1)
    # human_essays['assignments_and_essays'] = human_essays.apply(lambda row: f'<s>[INST] Write an essay with wording similar to a middle school or high school student for the assignment:\n{row["assignment"]}\n[/INST] {row["full_text"]}', axis=1)
    human_essays['assignments_and_essays'] = human_essays.apply(lambda row: f'<s>[INST] Write an essay with wording similar to a middle school or high school student for the assignment:\n{row["assignment"]}\n[/INST] {row["full_text"]}</s>', axis=1)
    return human_essays

def TrainModel(learning_rate, epoch_count, output_directory_name_prefix):
    MAX_SEQ_LENGTH = 2048
    HAS_BFLOAT16 = torch.cuda.is_bf16_supported()

    model, tokenizer = FastMistralModel.from_pretrained(
        model_name = '/mnt/data01/Models/LLMs/mistral-ft-optimized-1218',
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True,
    )

    model = FastMistralModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Currently only supports dropout = 0
        bias = "none",    # Currently only supports bias = "none"
        use_gradient_checkpointing = True,
        max_seq_length = MAX_SEQ_LENGTH,
    )

    # logging.set_verbosity_info()
    logging.set_verbosity_warning()

    human_essays_dataset = PrepareDataset()

    kf = KFold(n_splits=5, shuffle=True)
    for fold, (train_index, val_index) in enumerate(kf.split(human_essays_dataset)):
        print(f"Training on fold {fold+1}")
        
        train_dataset = human_essays_dataset.iloc[train_index]
        val_dataset = human_essays_dataset.iloc[val_index]
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=Dataset.from_pandas(train_dataset),
            eval_dataset=Dataset.from_pandas(val_dataset),
            dataset_text_field="assignments_and_essays",
            max_seq_length=MAX_SEQ_LENGTH,
            tokenizer=tokenizer,
            args=TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                per_device_eval_batch_size=2,
                warmup_steps=50,
                num_train_epochs=epoch_count,
                # max_steps=100,
                learning_rate=learning_rate,
                fp16=not HAS_BFLOAT16,
                bf16=HAS_BFLOAT16,
                logging_steps=1,
                output_dir=f"Models/StudentImitators/{output_directory_name_prefix}_Fold{fold+1}",
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
            ),
        )
        
        trainer.train()
        metrics = trainer.evaluate()

        print(metrics)

        trainer.save_model(f"Models/StudentImitators/{output_directory_name_prefix}_Fold{fold+1}/final_model")

        break

if __name__ == "__main__":
    # TrainModel(learning_rate = 2e-4, epoch_count = 1, output_directory_name_prefix = 'mistral-ft-optimized-2e-4_1') # eval loss = 1.5844
    TrainModel(learning_rate = 1e-4, epoch_count = 2, output_directory_name_prefix = 'mistral-ft-optimized-1e-4_2') # eval loss = 1.5823

### Llama 2 13B:
# TrainModel(learning_rate = 5e-5, epoch_count = 1, output_directory_name_prefix = 'Llama-2-13B-fp16_5e-5_1') # eval loss = 1.5715
# TrainModel(learning_rate = 1e-4, epoch_count = 1, output_directory_name_prefix = 'Llama-2-13B-fp16_1e-4_1') # eval loss = 1.5592
# TrainModel(learning_rate = 2e-4, epoch_count = 1, output_directory_name_prefix = 'Llama-2-13B-fp16_2e-4_1') # eval loss = 1.5486
# TrainModel(learning_rate = 4e-4, epoch_count = 1, output_directory_name_prefix = 'Llama-2-13B-fp16_4e-4_1') # eval loss = 1.5389
# TrainModel(learning_rate = 8e-4, epoch_count = 1, output_directory_name_prefix = 'Llama-2-13B-fp16_8e-4_1') # eval loss = 1.5345
# TrainModel(learning_rate = 1.6e-3, epoch_count = 1, output_directory_name_prefix = 'Llama-2-13B-fp16_1.6e-3_1') # Diverged! LR too high!
# TrainModel(learning_rate = 8e-4, epoch_count = 2, output_directory_name_prefix = 'Llama-2-13B-fp16_8e-4_2') # eval loss = 1.5483
# TrainModel(learning_rate = 4e-4, epoch_count = 2, output_directory_name_prefix = 'Llama-2-13B-fp16_4e-4_2') # eval loss = 1.5352
# TrainModel(learning_rate = 2e-4, epoch_count = 2, output_directory_name_prefix = 'Llama-2-13B-fp16_2e-4_2') # eval loss = 1.5362

### Mistral 70 foundation or instruct v0.2:
# Baseline lr 2e-4
# Baseline template: f'### Assignment: {row["assignment"]}\n### Essay: {row["full_text"]}'
# Non-instruction tuned foundation model.
#   1 epoch:
#       EXAMPLE HOSTING COMMAND: "python -m vllm.entrypoints.api_server --model Models/StudentImitators/single_epoch_fold_1/final_model_merged/ --max-model-len 2048"
#       {'train_runtime': 6538.7879, 'train_samples_per_second': 3.18, 'train_steps_per_second': 0.199, 'train_loss': 1.6949349380255296, 'epoch': 1.0}                                                                                                                                           
#       {'eval_loss': 1.6394448280334473, 'eval_runtime': 492.4974, 'eval_samples_per_second': 10.558, 'eval_steps_per_second': 2.64, 'epoch': 1.0}
#   
#   3 epochs:
#       {'train_runtime': 19625.5474, 'train_samples_per_second': 3.179, 'train_steps_per_second': 0.199, 'train_loss': 1.558870387805374, 'epoch': 3.0}                                                                                                                                          
#       {'eval_loss': 1.6858057975769043, 'eval_runtime': 492.3037, 'eval_samples_per_second': 10.563, 'eval_steps_per_second': 2.641, 'epoch': 3.0}
#
# Instruct template: f'<s>[INST] Write an essay with wording similar to a middle school or high school student for the assignment:\n{row["assignment"]}\n[/INST] {row["full_text"]}'
# Instruction tuned foundation model.
#   1 epoch:
#       EXAMPLE HOSTING COMMAND: "python -m vllm.entrypoints.api_server --model Models/StudentImitators/single_epoch_Instruct-v0.2-start_fold_1/final_model_merged/ --max-model-len 2048"
#       {'train_runtime': 6673.0928, 'train_samples_per_second': 3.116, 'train_steps_per_second': 0.195, 'train_loss': 1.6699980771752667, 'epoch': 1.0}                                                                                                                                          
#       {'eval_loss': 1.5982861518859863, 'eval_runtime': 502.9257, 'eval_samples_per_second': 10.339, 'eval_steps_per_second': 2.585, 'epoch': 1.0}
#
# Baseline template
# Instruction tuned foundation model.
#   1 epoch:
#       {'train_runtime': 14034.7857, 'train_samples_per_second': 1.482, 'train_steps_per_second': 0.093, 'train_loss': 1.7183623830569166, 'epoch': 1.0}                                                                                                                                         
#       {'eval_loss': 1.6514554023742676, 'eval_runtime': 1072.6928, 'eval_samples_per_second': 4.848, 'eval_steps_per_second': 1.212, 'epoch': 1.0}
#
# Instruct template V2: f'<s>[INST] Write an essay with wording similar to a middle school or high school student for the assignment:\n{row["assignment"]}\n[/INST] {row["full_text"]}</s>'
# Instruction tuned foundation model.
#   1 epoch:
#       EXAMPLE HOSTING COMMAND: "python -m vllm.entrypoints.api_server --model Models/StudentImitators/single_epoch_Instruct-v0.2-start_fold_1/final_model_merged/ --max-model-len 2048"
#       {'train_runtime': 6731.2135, 'train_samples_per_second': 3.089, 'train_steps_per_second': 0.193, 'train_loss': 1.6701208174091013, 'epoch': 1.0}                                                                                                                                          
#       {'eval_loss': 1.5979669094085693, 'eval_runtime': 503.7484, 'eval_samples_per_second': 10.323, 'eval_steps_per_second': 2.581, 'epoch': 1.0}
#
# V2 instruct template, instruction tuned model, lr reduced to 1e-4, 2 epochs:
#   {'train_runtime': 13365.7289, 'train_samples_per_second': 3.112, 'train_steps_per_second': 0.194, 'train_loss': 1.6212755690362473, 'epoch': 2.0}
#   {'eval_loss': 1.5956535339355469, 'eval_runtime': 503.3407, 'eval_samples_per_second': 10.331, 'eval_steps_per_second': 2.583, 'epoch': 2.0}
#   Saved in Models/StudentImitators/1e-4_2_Fold1
