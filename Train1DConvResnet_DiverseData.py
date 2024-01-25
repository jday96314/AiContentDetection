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
import multiprocessing
import os

from MultiDataset import PileCompletionsDataset, PersuadeDataset, MultiDataset
from TokenResNet import BasicBlock, Bottleneck, ResNet_HF

def GetPilePath():
    local_path = '/mnt/data01/data/PileCompletions'
    possibly_remote_path = 'data/PileCompletions'
    if os.path.exists('/mnt/data01/data/PileCompletions'):
        path = local_path
    else:
        path = possibly_remote_path

    return path

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
    FIXED_ORDER_DATA_AUGMENTATION_STEPS = [
        {'name' : 'BuggySpellCheck', 'p' : [0.7, 0.2, 0.2] },
        {'name' : 'RemoveBlacklistedCharacters', 'p' : [0.7, 0.2, 0.2] },
    ]
    RAND_ORDER_DATA_AUGMENTATION_STEPS = [
        {'name' : 'DecapitalizeRandomLetter', 'p' : 0.1 },
        {'name' : 'DecapitalizeRandomLetter', 'p' : 0.1 },
        {'name' : 'CapitalizeRandomLetter', 'p' : 0.1 },

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
    pile_path = GetPilePath()
    training_pile_dataset = PileCompletionsDataset(
        f'{pile_path}/*/*.json', 
        [i for i in range(5) if i != cross_validation_fold_id],
        prefetch_data = False)
    
    tokenizer = f'Tokenizers/tokenmonster/vocabs/english-{24000}-strict-v1.vocab'
    training_dataset = MultiDataset(
        datasets = [training_persuade_dataset, training_pile_dataset],
        sampling_proportions = [persuade_sampling_proportion, 1 - persuade_sampling_proportion],
        tokenizer = tokenizer,
        max_sequence_length = max_sequence_length,
        fixed_order_data_augmentation_steps = FIXED_ORDER_DATA_AUGMENTATION_STEPS,
        rand_order_data_augmentation_steps = RAND_ORDER_DATA_AUGMENTATION_STEPS,
        samples_per_epoch = 50000
    )
    
    # CREATE TESTING DATSETS.
    testing_persuade_dataset = PersuadeDataset(
        human_essays_filepath = 'data/PERSUADE/persuade_2.0_human_scores_demo_id_github.csv',
        generated_essay_filepath_patterns = GENERATED_ESSAY_FILEPATH_PATTERNS, 
        generated_subset_sampling_proportions = GENERATED_ESSAY_SUBSET_SAMPLING_PROPORTIONS, 
        whitelisted_assignment_names = PersuadeDataset.OTHER_ASSIGNMENTS,
    )
    testing_pile_dataset = PileCompletionsDataset(
        f'{pile_path}/*/*.json', 
        [cross_validation_fold_id],
        prefetch_data = False)
    testing_dataset = MultiDataset(
        datasets = [testing_persuade_dataset, testing_pile_dataset],
        sampling_proportions = [0.5, 0.5],
        tokenizer = tokenizer,
        max_sequence_length = max_sequence_length,
        fixed_order_data_augmentation_steps = FIXED_ORDER_DATA_AUGMENTATION_STEPS,
        rand_order_data_augmentation_steps = RAND_ORDER_DATA_AUGMENTATION_STEPS,
        samples_per_epoch = 10000
    )

    # CREATE DATALOADERS.
    data_collator = DataCollatorWithPadding(tokenizer=training_dataset.Tokenizer) if type(training_dataset.Tokenizer) != str else None
    train_data_loader = DataLoader(
        training_dataset, 
        batch_size=batch_size, 
        num_workers=multiprocessing.cpu_count(), 
        prefetch_factor=2,
        shuffle=True,
        pin_memory=True,
        collate_fn=data_collator)
    
    test_data_loader = DataLoader(
        testing_dataset, 
        batch_size=batch_size, 
        num_workers=multiprocessing.cpu_count(), 
        prefetch_factor=2,
        shuffle=True,
        pin_memory=True,
        collate_fn=data_collator)
    
    return train_data_loader, test_data_loader

def TrainModel(persuade_sampling_proportion, max_learning_rate, layers, epoch_count, output_filename_prefix):
    # SETUP DATALOADER.
    MODEL_NAME = 'microsoft/deberta-v3-base'
    # MAX_SEQUENCE_LENGTH_TOKENS = 768
    MAX_SEQUENCE_LENGTH_TOKENS = 1024
    BATCH_SIZE = 32
    # BATCH_SIZE = 128
    train_data_loader, test_data_loader = SetupDataLoaders(
        persuade_sampling_proportion = persuade_sampling_proportion,
        model_name = MODEL_NAME, 
        batch_size = BATCH_SIZE, 
        max_sequence_length = MAX_SEQUENCE_LENGTH_TOKENS,
        cross_validation_fold_id = 0)

    # CREATE MODEL.   
    VOCAB_SIZE = 24000
    width = 256
    model = ResNet_HF(
        # +2 for padding & mask tokens.
        vocab_size=(VOCAB_SIZE + 2),
        block=BasicBlock, 
        layers=layers,
        width_coefs=[1, 1, 2, 2],
        num_classes=2, 
        token_embedding_dim_count = width,
        base_width = width).cuda()
    
    # model.load_state_dict(torch.load('Models/AutoEncoders/encoder_5e6_005_baseline.pth'), strict=False)
    # model.load_state_dict(
    #     torch.load('Models/AutoEncoders/encoder_BasicBlock_[2, 2, 2, 2]_250.pth', map_location=torch.device('cuda:0')), 
    #     strict=False)
    model.load_state_dict(
        torch.load('Models/AutoEncoders/encoder_BasicBlock_[2, 2, 2, 2]_500.pth', map_location=torch.device('cuda:0')), 
        strict=False)

    # SETUP TRAINING ALGORITHM.
    # optimizer = optim.Adam(model.parameters(), lr=max_learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=max_learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    scaler = torch.cuda.amp.GradScaler()

    total_step_count = len(train_data_loader) * epoch_count
    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=max_learning_rate,
        total_steps=total_step_count,
        pct_start=0.05,
        anneal_strategy='linear'
    )

    # TRAIN AND TEST THE MODEL.
    best_auroc = -99999999
    for epoch in range(epoch_count):
        model.train()
        train_losses = []
        for batch in tqdm(train_data_loader):
            # Have shape (batch size, token count)
            token_sequences = batch['input_ids'].cuda()
            attention_masks = batch['attention_mask'].cuda() if 'attention_mask' in batch.keys() else None
            # Has shape (batch size)
            labels = batch['is_artificially_generated'].cuda()

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
                token_sequences = batch['input_ids'].cuda()
                attention_masks = batch['attention_mask'].cuda() if 'attention_mask' in batch.keys() else None
                # Has shape (batch size)
                labels = batch['is_artificially_generated'].cuda()

                with torch.cuda.amp.autocast():
                    raw_predictions = model(token_sequences, attention_masks).logits
                    loss = criterion(raw_predictions, labels)

                test_losses.append(loss.detach().cpu())

                scaled_predictions = raw_predictions.softmax(dim = 1)[:,1]
                all_predictions.extend(scaled_predictions.cpu().numpy())
                all_actual_values.extend(labels.cpu().numpy())

                all_dataset_ids.extend(batch['data_origin_id'])

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

        average_auroc = np.average(auroc_scores_by_dataset, weights=[1, 1])
        if average_auroc > best_auroc:
            best_auroc = average_auroc

            if output_filename_prefix is not None:
                torch.save(model.state_dict(), f'Models/ResNetDiverse/{output_filename_prefix}_size={width}-{sum(layers)}_CTX{MAX_SEQUENCE_LENGTH_TOKENS}_{int(persuade_sampling_proportion * 1000)}_{int(auroc_scores_by_dataset[0] * 1000)}_{int(auroc_scores_by_dataset[1] * 1000)}.pth')

    return best_auroc

if __name__ == '__main__':
    # epoch_counts = [1, 4, 16, 32, 64]
    epoch_counts = [4, 16]
    persuade_sampling_proportions = [0.1, 0.5]
    for persuade_sampling_proportion in persuade_sampling_proportions:
        aurocs = []
        for epoch_count in epoch_counts:
            auroc = TrainModel(
                persuade_sampling_proportion=persuade_sampling_proportion, 
                max_learning_rate=0.0025, 
                layers = [2, 2, 2, 2], 
                epoch_count = epoch_count, 
                output_filename_prefix='Victim')
            aurocs.append(auroc)

    for epoch_count, auroc in zip(epoch_counts, aurocs):
        print(epoch_count, auroc)

    # Original finetuning data:
        ## No pretrained foundation.
        # 1 0.7235602938086809
        # 2 0.8329326120910492
        # 4 0.8980360814331405
        # 8 0.9218504406064243
        # 16 0.9414205690568065
        # 32 0.9487482450731854
        # 64 0.9474328180163856

        ## Baseline foundation (5e6 documents with lr=0.005, train loss ~0.86).
        # 1 0.9005817552846596
        # 4 0.94045541583563
        # 16 0.9498089784195376
        # 32 0.9543932554309691

        ## Scaled up foundation (5e7 documents with lr=0.0025, train loss ~0.75).
        # 1 0.9269692266214111
        # 4 0.9386772890486528
        # 16 0.9507141398255745
        # 32 0.9512040794211858

    # Added 50% more finetuning data (Pile).
        ## Baseline foundation model (lr=0.005, train loss ~0.86):
        # 1 0.9045005539949507
        # 4 0.9508262435397724
        # 16 0.9648445981954328
        # 32 0.9654757544835921

        ## Updated foundation model (lr=0.0025, train loss ~0.75):
        # 1 0.9112377062250128
        # 4 0.9569012955911917
        # 16 0.9617962013441415
        # 32 0.9647324964747832

        ## Updated foundation model (lr=0.005, train loss ~0.75):
        # 1 0.9239597790364622
        # 4 0.9524259833978883
        # 16 0.9636739824921482
        # 32 0.9647514965523102

        ## Updated foundation model (lr=0.005, train loss ~0.75), leaky relu:
        # 1 0.9153505358864278
        # 4 0.9479852775124935
        # 16 0.9633610727437125
        # 32 0.9650958348013524

        ## Updated foundation model (lr=0.005, train loss ~0.75), normal relu, dropout 0.25:
        # 1 0.930151520012728
        # 4 0.9527638139693567
        # 16 0.9642277373528856
        # 32 0.9652677464763769

    # Adding reddit data...
        ## 10% reddit data, updated foundation model (lr=0.005, train loss ~0.75):
        # 1 0.9389987719686463
        # 4 0.9534131628570512
        # 16 0.9693190029086697
        # 32 0.9683776284400603

        ## 5% reddit data, updated foundation model (lr=0.005, train loss ~0.75):
        # 1 0.9294562004032914
        # 4 0.9579636860112011
        # 16 0.9686613949860445
        # 32 0.9686851574435741