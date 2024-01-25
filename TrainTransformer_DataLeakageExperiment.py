from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import roc_auc_score

from MultiDataset import PileCompletionsDataset, PersuadeDataset, MultiDataset

def SetupDataLoaders(persuade_sampling_proportion, model_name, batch_size, max_sequence_length, cross_validation_fold_id):
    GENERATED_ESSAY_FILEPATH_PATTERNS = [
        'data/GeneratedEssays/airoboros-l2-70B-gpt4-1.4.1-AWQ_V0/*',
        'data/GeneratedEssays/GodziLLa2-70B-AWQ_V0/*',
        'data/GeneratedEssays/gpt-3.5-turbo_V0/*',
        'data/GeneratedEssays/gpt-3.5-turbo-16k_V0/*',
        'data/GeneratedEssays/gpt-4-1106-preview_V0/*',
        'data/GeneratedEssays/Llama-2-13B-chat-AWQ_V0/*',
        'data/GeneratedEssays/mistral-7b-platypus_V0/*',
        'data/GeneratedEssays/Mistral-7B-guanaco1k-ep2_V0/*',
        'data/GeneratedEssays/Mistral-7B-Instruct-v0.1_V0/*',
        'data/GeneratedEssays/Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-AWQ_V0/*',
    ]
    GENERATED_ESSAY_SUBSET_SAMPLING_PROPORTIONS = [
        20000,
        10000,
        20000, # gpt-3.5-turbo oversampled ~2x.
        1500, # gpt-3.5-turbo-16k oversampled ~2x.
        4000, # gpt-4 oversampled ~4x.
        20000,
        20000,
        300,
        20000,
        13000
    ]
    DATA_AUGMENTATION_STEPS = [
        {'name' : 'BuggySpellCheck', 'p' : 0.2 },
        {'name' : 'RemoveBlacklistedCharacters', 'p' : 0.3 },

        {'name' : 'char_swap', 'p' : 0.1 },
        {'name' : 'missing_char', 'p' : 0.1 },
        {'name' : 'extra_char', 'p' : 0.1 },
        {'name' : 'nearby_char', 'p' : 0.1 },
        {'name' : 'similar_char', 'p' : 0.1 },
        {'name' : 'skipped_space', 'p' : 0.1 },
        {'name' : 'random_space', 'p' : 0.1 },
        {'name' : 'repeated_char', 'p' : 0.1 },
        {'name' : 'unichar', 'p' : 0.1 },

        {'name' : 'char_swap', 'p' : 0.1 },
        {'name' : 'missing_char', 'p' : 0.1 },
        {'name' : 'extra_char', 'p' : 0.1 },
        {'name' : 'nearby_char', 'p' : 0.1 },
        {'name' : 'similar_char', 'p' : 0.1 },
        {'name' : 'skipped_space', 'p' : 0.1 },
        {'name' : 'random_space', 'p' : 0.1 },
        {'name' : 'repeated_char', 'p' : 0.1 },
        {'name' : 'unichar', 'p' : 0.1 },
    ]
    
    # CREATE TRAINING DATASETS.
    training_persuade_dataset = PersuadeDataset(
        human_essays_filepath = 'data/PERSUADE/persuade_2.0_human_scores_demo_id_github.csv',
        generated_essay_filepath_patterns = GENERATED_ESSAY_FILEPATH_PATTERNS, 
        generated_subset_sampling_proportions = GENERATED_ESSAY_SUBSET_SAMPLING_PROPORTIONS, 
        whitelisted_assignment_names = PersuadeDataset.COMPETITION_ASSIGNMENTS,
    )
    training_pile_dataset = PileCompletionsDataset(
        'data/PileCompletions/*/*.json', 
        [i for i in range(5) if i != cross_validation_fold_id])
    training_dataset = MultiDataset(
        datasets = [training_persuade_dataset, training_pile_dataset],
        sampling_proportions = [persuade_sampling_proportion, 1 - persuade_sampling_proportion],
        tokenizer = AutoTokenizer.from_pretrained(model_name),
        max_sequence_length = max_sequence_length,
        data_augmentation_steps = DATA_AUGMENTATION_STEPS,
        samples_per_epoch = 500
    )
    
    # CREATE TESTING DATSETS.
    testing_persuade_dataset = PersuadeDataset(
        human_essays_filepath = 'data/PERSUADE/persuade_2.0_human_scores_demo_id_github.csv',
        generated_essay_filepath_patterns = GENERATED_ESSAY_FILEPATH_PATTERNS, 
        generated_subset_sampling_proportions = GENERATED_ESSAY_SUBSET_SAMPLING_PROPORTIONS, 
        # whitelisted_assignment_names = PersuadeDataset.OTHER_ASSIGNMENTS,
        whitelisted_assignment_names = PersuadeDataset.COMPETITION_ASSIGNMENTS,
    )
    testing_pile_dataset = PileCompletionsDataset(
        'data/PileCompletions/*/*.json', 
        [cross_validation_fold_id])
    testing_dataset = MultiDataset(
        datasets = [testing_persuade_dataset, testing_pile_dataset],
        sampling_proportions = [0.5, 0.5],
        tokenizer = AutoTokenizer.from_pretrained(model_name),
        max_sequence_length = max_sequence_length,
        data_augmentation_steps = DATA_AUGMENTATION_STEPS,
        samples_per_epoch = 2000
    )

    # CREATE DATALOADERS.
    data_collator = DataCollatorWithPadding(tokenizer=training_dataset.Tokenizer)
    train_data_loader = DataLoader(
        training_dataset, 
        batch_size=batch_size, 
        num_workers=16, 
        prefetch_factor=20,
        shuffle=True,
        pin_memory=True,
        collate_fn=data_collator)
    
    test_data_loader = DataLoader(
        testing_dataset, 
        batch_size=batch_size, 
        num_workers=16, 
        prefetch_factor=20,
        shuffle=True,
        pin_memory=True,
        collate_fn=data_collator)
    
    return train_data_loader, test_data_loader

def TrainModel(persuade_sampling_proportion, max_learning_rate, save_models):
    # SETUP DATALOADER.
    # MODEL_NAME = 'allenai/longformer-base-4096'
    # MAX_SEQUENCE_LENGTH_TOKENS = 1024
    MODEL_NAME = 'microsoft/deberta-v3-base'
    MAX_SEQUENCE_LENGTH_TOKENS = 512
    BATCH_SIZE = 8
    train_data_loader, test_data_loader = SetupDataLoaders(
        persuade_sampling_proportion = persuade_sampling_proportion,
        model_name = MODEL_NAME, 
        batch_size = BATCH_SIZE, 
        max_sequence_length = MAX_SEQUENCE_LENGTH_TOKENS,
        cross_validation_fold_id = 0)

    # CREATE MODEL.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = 2).cuda()

    # SETUP TRAINING ALGORITHM.
    optimizer = optim.Adam(model.parameters(), lr=max_learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    EPOCH_COUNT = 20
    total_step_count = len(train_data_loader) * EPOCH_COUNT
    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=max_learning_rate,
        total_steps=total_step_count,
        pct_start=0.05,
        anneal_strategy='linear'
    )

    # TRAIN AND TEST THE MODEL.
    best_auroc = 0
    for epoch in range(EPOCH_COUNT):
        model.train()
        train_losses = []
        for batch in tqdm(train_data_loader):
            # Have shape (batch size, token count)
            token_sequences = batch.input_ids.cuda()
            attention_masks = batch.attention_mask.cuda()
            # Has shape (batch size)
            labels = batch.is_artificially_generated.cuda()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(token_sequences, attention_masks)
                loss = criterion(output.logits, labels)
            
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            lr_schedule.step()

            train_losses.append(loss.detach().cpu())

        model.eval()
        test_losses = []
        all_dataset_ids = []
        all_predictions = []
        all_actual_values = []
        with torch.no_grad():
            for batch in tqdm(test_data_loader):
                # Have shape (batch size, token count)
                token_sequences = batch.input_ids.cuda()
                attention_masks = batch.attention_mask.cuda()
                # Has shape (batch size)
                labels = batch.is_artificially_generated.cuda()

                with torch.cuda.amp.autocast():
                    raw_predictions = model(token_sequences, attention_masks).logits
                    loss = criterion(raw_predictions, labels)

                test_losses.append(loss.detach().cpu())

                scaled_predictions = raw_predictions.softmax(dim = 1)[:,1]
                all_predictions.extend(scaled_predictions.cpu().numpy())
                all_actual_values.extend(labels.cpu().numpy())

                all_dataset_ids.extend(batch.data_origin_id)

        all_dataset_ids, all_predictions, all_actual_values = np.array(all_dataset_ids), np.array(all_predictions), np.array(all_actual_values)
        auroc_scores_by_dataset = []
        for dataset_id in np.unique(all_dataset_ids):
            dataset_mask = (all_dataset_ids == dataset_id)
            predictions = all_predictions[dataset_mask]
            actual_values = all_actual_values[dataset_mask]

            auroc = roc_auc_score(actual_values, predictions)
            auroc_scores_by_dataset.append(auroc)

        np.set_printoptions(precision=4)
        auroc_scores_by_dataset = np.array(auroc_scores_by_dataset)
        print(f'Epoch {epoch + 1}: Train Loss = {np.mean(train_losses):.4f}, Test Loss = {np.mean(test_losses):.4f}, Test AUROCs = {auroc_scores_by_dataset}')

        # average_auroc = np.average(auroc_scores_by_dataset, weights=[2, 1])
        average_auroc = np.average(auroc_scores_by_dataset, weights=[1, 1])
        if average_auroc > best_auroc:
            best_auroc = average_auroc

            if save_models:
                torch.save(model.state_dict(), f'Models/Transformer/{int(persuade_sampling_proportion * 1000)}_{int(auroc_scores_by_dataset[0] * 1000)}_{int(auroc_scores_by_dataset[1] * 1000)}.pth')

    return best_auroc

if __name__ == '__main__':
    TrainModel(persuade_sampling_proportion=0.5, max_learning_rate=3e-5, save_models=False)