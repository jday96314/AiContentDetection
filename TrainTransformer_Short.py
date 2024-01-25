from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, DebertaV2Config, DebertaV2ForSequenceClassification
import numpy as np
from sklearn.metrics import roc_auc_score
import multiprocessing

from MultiDataset import PileCompletionsDataset, PersuadeDataset, MultiDataset, TrickyCrawlDataset

def SetupDataLoaders(persuade_sampling_proportion, model_name, batch_size, max_sequence_length, training_example_count, cross_validation_fold_id, random_crop_chars):
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
        'data/GeneratedEssays_StudentImitator/1e-4_2_Fold1_V2/*',
        'data/GeneratedEssays_StudentImitator/Llama-2-13B-fp16_1e-4_1_Fold1_V2/*',
        'data/GeneratedEssays_StudentImitator/Llama-2-13B-fp16_2e-4_1_Fold1_V2/*',
        'data/GeneratedEssays_StudentImitator/Llama-2-13B-fp16_4e-4_1_Fold1_V2/*',
        'data/GeneratedEssays_StudentImitator/Llama-2-13B-fp16_8e-4_1_Fold1_V2/*',
        'data/GeneratedEssays_StudentImitator/mistral-ft-optimized-1e-4_2_Fold1_V2/*',
        'data/GeneratedEssays_StudentImitator/mistral-ft-optimized-2e-4_1_Fold1_V2/*',
        'data/GeneratedEssays_AdversarialPersuade/*/*/*.json',
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
        13000,
        25000, # Mistral instruct v0.2 founcation.
        # Llama 2 13B based sampling proportions below. Undersampling because these are similar and many were trained with sub-optimnal hyperparameters.
        1000,
        1000,
        1000,
        22000,
        # Mistral ft optimized sampling proportions below.
        5000,
        20000,
        12000, # Adversarial persuade.
    ]
    FIXED_ORDER_DATA_AUGMENTATION_STEPS = [
        {'name' : 'BuggySpellCheck', 'p' : [0.7, 0.2, 0.2, 0.2] },
        {'name' : 'RemoveBlacklistedCharacters', 'p' : [0.7, 0.2, 0.2, 0.2] },
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
    training_tricky_crawl_dataset = TrickyCrawlDataset('data/TrickyCrawl/1500000_125192.csv')
    training_pile_dataset = PileCompletionsDataset(
        'data/PileCompletions_V3/*/*.json', 
        [i for i in range(10) if i != cross_validation_fold_id])
    training_pajama_dataset = PileCompletionsDataset('data/SlimPajamaCompletions/Completions/*/*.json', list(range(10)))

    total_completion_count = len(training_pile_dataset) + len(training_pajama_dataset)
    completions_sampling_ratio_ratios = np.array([len(training_pile_dataset), len(training_pajama_dataset)]) / total_completion_count
    CRAWL_SAMPLING_PROPORTION = 0.05
    normalized_completions_sampling_ratios = completions_sampling_ratio_ratios * (1 - persuade_sampling_proportion - CRAWL_SAMPLING_PROPORTION)

    print('Using train completion sampling ratios:', normalized_completions_sampling_ratios)

    training_dataset = MultiDataset(
        datasets = [training_persuade_dataset, training_pile_dataset, training_pajama_dataset, training_tricky_crawl_dataset],
        sampling_proportions = [persuade_sampling_proportion, *normalized_completions_sampling_ratios, CRAWL_SAMPLING_PROPORTION],
        tokenizer = AutoTokenizer.from_pretrained(model_name),
        max_sequence_length = max_sequence_length,
        fixed_order_data_augmentation_steps = FIXED_ORDER_DATA_AUGMENTATION_STEPS,
        rand_order_data_augmentation_steps = RAND_ORDER_DATA_AUGMENTATION_STEPS,
        samples_per_epoch = training_example_count,
        random_crop_length_chars = random_crop_chars
    )
    
    # CREATE TESTING DATSETS.
    testing_persuade_dataset = PersuadeDataset(
        human_essays_filepath = 'data/PERSUADE/persuade_2.0_human_scores_demo_id_github.csv',
        generated_essay_filepath_patterns = GENERATED_ESSAY_FILEPATH_PATTERNS, 
        generated_subset_sampling_proportions = GENERATED_ESSAY_SUBSET_SAMPLING_PROPORTIONS, 
        whitelisted_assignment_names = PersuadeDataset.OTHER_ASSIGNMENTS,
    )
    testing_pile_dataset = PileCompletionsDataset(
        'data/PileCompletions_V3/*/*.json', 
        [cross_validation_fold_id])
    testing_dataset = MultiDataset(
        datasets = [testing_persuade_dataset, testing_pile_dataset],
        sampling_proportions = [0.5, 0.5],
        tokenizer = AutoTokenizer.from_pretrained(model_name),
        max_sequence_length = max_sequence_length,
        fixed_order_data_augmentation_steps = FIXED_ORDER_DATA_AUGMENTATION_STEPS,
        rand_order_data_augmentation_steps = RAND_ORDER_DATA_AUGMENTATION_STEPS,
        samples_per_epoch = 15000,
        random_crop_length_chars = random_crop_chars
    )

    # CREATE DATALOADERS.
    data_collator = DataCollatorWithPadding(tokenizer=training_dataset.Tokenizer)
    train_data_loader = DataLoader(
        training_dataset, 
        batch_size=batch_size, 
        num_workers=multiprocessing.cpu_count(), 
        prefetch_factor=20,
        shuffle=True,
        pin_memory=True,
        collate_fn=data_collator)
    
    test_data_loader = DataLoader(
        testing_dataset, 
        batch_size=batch_size, 
        num_workers=multiprocessing.cpu_count(), 
        prefetch_factor=20,
        shuffle=True,
        pin_memory=True,
        collate_fn=data_collator)
    
    return train_data_loader, test_data_loader

def TestModel(test_data_loader, model, criterion):
    test_losses = []
    all_dataset_ids = []
    all_predictions = []
    all_actual_values = []
    with torch.no_grad():
        for batch in tqdm(test_data_loader):
        # for batch in test_data_loader:
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

    auroc_scores_by_dataset = np.array(auroc_scores_by_dataset)

    return auroc_scores_by_dataset, np.mean(test_losses)

def TrainModel(persuade_sampling_proportion, max_learning_rate, label_smoothing, training_example_count, output_subdir, random_crop_chars):
    # SETUP DATALOADER.
    MODEL_NAME = 'microsoft/deberta-v3-large'
    max_sequence_length_tokens = random_crop_chars * 3 // 8
    BATCH_SIZE = 16

    train_data_loader, test_data_loader = SetupDataLoaders(
        persuade_sampling_proportion = persuade_sampling_proportion,
        model_name = MODEL_NAME, 
        batch_size = BATCH_SIZE, 
        max_sequence_length = max_sequence_length_tokens,
        training_example_count = training_example_count,
        cross_validation_fold_id = 0,
        random_crop_chars = random_crop_chars)

    # CREATE MODEL.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels = 2,
        hidden_dropout_prob = 0,
        attention_probs_dropout_prob = 0
    ).cuda()
    
    model.train()

    #model.load_state_dict(torch.load('Models/Transformer/3090_1/E7_CTX1024_10_981_988.pth'))

    # SETUP TRAINING ALGORITHM.
    optimizer = optim.Adam(model.parameters(), lr=max_learning_rate)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.cuda.amp.GradScaler()

    total_step_count = len(train_data_loader)
    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=max_learning_rate,
        total_steps=total_step_count,
        pct_start=0.05,
        anneal_strategy='linear'
    )

    # TRAIN AND TEST THE MODEL.
    best_auroc = -99999999
    train_losses = []
    for batch_index, train_batch in enumerate(tqdm(train_data_loader)):
        # SEND DATA TO GPU.
        # Have shape (batch size, token count)
        token_sequences = train_batch.input_ids.cuda()
        attention_masks = train_batch.attention_mask.cuda()
        # Has shape (batch size)
        labels = train_batch.is_artificially_generated.cuda()

        # CLEAR GRADIENTS.
        optimizer.zero_grad()

        # FORWARD PASS.
        with torch.cuda.amp.autocast():
            output = model(token_sequences, attention_masks)
            loss = criterion(output.logits, labels)
        
        # BACKWARD PASS.
        scaler.scale(loss).backward()

        # UPDATE MODEL PARAMETERS, LR SCHEDULE, ETC.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        lr_schedule.step()

        # RECORD LOSS.
        train_losses.append(loss.detach().cpu())

        # POSSIBLY TEST MODEL.
        step_number = batch_index + 1
        train_steps_per_test = 50000 // BATCH_SIZE
        if (step_number % train_steps_per_test == 0) or (step_number == total_step_count):
            model.eval()
            auroc_scores_by_dataset, test_loss = TestModel(test_data_loader, model, criterion)
            model.train()

            np.set_printoptions(precision=4)
            print(f'Step {step_number}: Train Loss = {np.mean(train_losses):.4f}, Test Loss = {test_loss:.4f}, Test AUROCs = {auroc_scores_by_dataset}')

            average_auroc = np.average(auroc_scores_by_dataset, weights=[1, 1])
            if (average_auroc > best_auroc) or (max(auroc_scores_by_dataset) > 0.993):
                best_auroc = average_auroc
            
                if output_subdir is not None:
                    torch.save(model.state_dict(), f'Models/Transformer/{output_subdir}/S{step_number}_CTX{max_sequence_length_tokens}_{int(persuade_sampling_proportion * 1000)}_{int(auroc_scores_by_dataset[0] * 1000)}_{int(auroc_scores_by_dataset[1] * 1000)}.pth')

            train_losses = []

    return best_auroc

def RunExperiments(max_learning_rate, label_smoothing, output_subdir, random_crop_chars):
    aurocs = []
    for trial_id in range(1):
        auroc = TrainModel(
            persuade_sampling_proportion=0.01, 
            max_learning_rate=max_learning_rate, 
            label_smoothing=label_smoothing,
            # training_example_count = 28 * 50000, 
            training_example_count = 20 * 50000,
            output_subdir=output_subdir,
            random_crop_chars = random_crop_chars)
        aurocs.append(auroc)

    return min(aurocs), np.mean(aurocs), max(aurocs)

if __name__ == '__main__':
    ##  AUROCs = [0.8064 0.8308]
    # lr = 1e-5
    # label_smoothing = 0.03
    # random_crop_length = 64
    # output_subdir = '3090_0'

    ##  AUROCs = [0.9095 0.9063] (v1 pile)
    ##  AUROCs = [0.903  0.9057] (v3 pile & tricky crawl, bugged slim pajama)
    ##  AUROCs = [0.9095 0.8988] (v3 pile, tricky crawl, slim pajama)
    lr = 1e-5
    label_smoothing = 0.03
    random_crop_length = 128
    output_subdir = '4090_0'

    print(f'lr = {lr}, label_smoothing = {label_smoothing}, output_subdir = {output_subdir}')

    min_auroc, mean_auroc, max_auroc = RunExperiments(lr, label_smoothing, output_subdir, random_crop_length)
    print(f'AUROC (min, mean, max) = ({min_auroc:.4f}, {mean_auroc:.4f}, {max_auroc:.4f})')
    
    ##  AUROCs = [0.9584 0.9561]
    ##  AUROCs = [0.9532 0.954 ] (v3 pile & tricky crawl, bugged slim pajama)
    ##  AUROCs = [0.9515 0.9522] (v3 pile, tricky crawl, slim pajama)
    lr = 1e-5
    label_smoothing = 0.03
    random_crop_length = 256
    output_subdir = '4090_0'

    ## Test AUROCs = [0.980 0.978]
    # lr = 1e-5
    # label_smoothing = 0.03
    # random_crop_length = 512
    # output_subdir = '3090_1'

    print(f'lr = {lr}, label_smoothing = {label_smoothing}, output_subdir = {output_subdir}')

    min_auroc, mean_auroc, max_auroc = RunExperiments(lr, label_smoothing, output_subdir, random_crop_length)
    print(f'AUROC (min, mean, max) = ({min_auroc:.4f}, {mean_auroc:.4f}, {max_auroc:.4f})')