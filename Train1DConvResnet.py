from math import ceil
from AWP import AWP

from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from TokenResNet import BasicBlock, Bottleneck, ResNet, ResNet_HF
from PersuadeDataset import PersuadeDataset

def SetupDataLoaders(dataset_path, vocab_size, batch_size, max_sequence_length, cross_validation_fold_id):
    # CREATE DATASETS.
    train_dataset = PersuadeDataset(
        preprocessed_data_cache_filepath = dataset_path, 
        vocab_size = vocab_size, 
        output_seq_length = max_sequence_length, 
        fold_ids = [i for i in range(5) if i != cross_validation_fold_id])
    
    test_dataset = PersuadeDataset(
        preprocessed_data_cache_filepath = dataset_path, 
        vocab_size = vocab_size, 
        output_seq_length = max_sequence_length, 
        fold_ids = [cross_validation_fold_id])
    
    # CREATE DATALOADERS.
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=2,
        shuffle=True,
        pin_memory=True)
    
    test_data_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=2,
        shuffle=True,
        pin_memory=True)

    return train_data_loader, test_data_loader

def TrainModel(foundation_model_path, fold_id):
    # CREATE MODEL.
    VOCAB_SIZE = 24000
    DEVICE = torch.device('cuda:0')
    model = ResNet(
        vocab_size=VOCAB_SIZE + 2,
        block=BasicBlock, 
        layers=[2, 2, 2, 2],
        width_coefs=[1, 1, 2, 2],
        num_classes=2, 
        token_embedding_dim_count = 256,
        base_width = 256).to(DEVICE)

    if foundation_model_path is not None:
        model.load_state_dict(torch.load(foundation_model_path))

    # SETUP DATA LOADERS.
    CROSS_VALIDATION_FOLD = 0
    train_data_loader, test_data_loader = SetupDataLoaders(
        dataset_path = 'data/Cache/Persuade_V0_BasicPreprocessing_50kGenerated_MetadataIncluded.p', 
        vocab_size = VOCAB_SIZE, 
        batch_size = 128, 
        max_sequence_length = 1024, 
        cross_validation_fold_id = CROSS_VALIDATION_FOLD
    )

    # SETUP TRAINING ALGORITHM.
    MAX_LEARNING_RATE = 0.01
    optimizer = optim.Adam(model.parameters(), lr=MAX_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    EPOCH_COUNT = 1
    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=MAX_LEARNING_RATE,
        total_steps=len(train_data_loader) * EPOCH_COUNT,
        pct_start=0.1,
        anneal_strategy='linear'
    )

    awp = AWP(
        model,
        optimizer,
        adv_lr=0.001,
        adv_eps=0.001,
        start_epoch = 0,
        scaler=scaler)

    # TRAIN AND TEST THE MODEL.
    best_auroc = 0.5
    best_loss = 10000
    for epoch in range(EPOCH_COUNT):
        model.train()
        # train_losses = []
        # for batch_index, (token_ids, labels) in enumerate(tqdm(train_data_loader)):
        #     # SEND DATA TO GPU.
        #     token_ids = token_ids.int().to(DEVICE)
        #     labels = labels.type(torch.LongTensor).to(DEVICE)

        #     # FORWARD PASS.
        #     with torch.cuda.amp.autocast():
        #         predictions = model(token_ids)
        #         loss = criterion(predictions, labels)
            
        #     # BACKWARD PASS.
        #     scaler.scale(loss).backward()

        #     awp.attack_backward(
        #         token_ids,
        #         labels,
        #         attention_mask = None, 
        #         epoch = epoch)

        #     # RECORD STATISTICS.
        #     train_losses.append(loss.detach().cpu())
            
        #     # POSSIBLY UPDATE MODEL.
        #     scaler.step(optimizer)
        #     scaler.update()
            
        #     optimizer.zero_grad()
        #     lr_schedule.step()

        train_losses = [-1]

        model.eval()
        test_losses = []
        all_predictions = []
        all_actual_values = []
        with torch.no_grad():
            for batch_index, (token_ids, labels) in enumerate(tqdm(train_data_loader)):
                token_ids = token_ids.int().to(DEVICE)
                labels = labels.type(torch.LongTensor).to(DEVICE)

                with torch.cuda.amp.autocast():
                    predictions = model(token_ids)
                    loss = criterion(predictions, labels)

                test_losses.append(loss.detach().cpu())

                scaled_predictions = predictions.softmax(dim = 1)[:,1]
                all_predictions.extend(scaled_predictions.cpu().numpy())
                all_actual_values.extend(labels.cpu().numpy())

        auroc = roc_auc_score(all_actual_values, all_predictions)

        avg_test_loss = np.mean(test_losses)
        print(f'Epoch {epoch + 1}: Train Loss = {np.mean(train_losses):.4f}, Test Loss = {avg_test_loss:.4f}, Test AUROC = {auroc:.5f}')

        if (auroc > best_auroc) or ((auroc == best_auroc) and (avg_test_loss < best_loss)):
            best_auroc = auroc
            best_loss = avg_test_loss
            
            foundation_model_name = foundation_model_path.split('/')[-1].replace('.pth', '') if foundation_model_path is not None else 'NoFoundation'
            torch.onnx.export(
                model, 
                token_ids, 
                f"Models/ResNet50k/{foundation_model_name}_fold{fold_id}_auroc{int(best_auroc*10000)}_loss{int(best_loss*10000)}.onnx", 
                do_constant_folding=True,
                input_names=['token_ids'], 
                output_names=['predictions'],
                dynamic_axes= {'token_ids' : {0 : 'batch_size'}, 'predictions' : {0 : 'batch_size'}})
            
    return best_auroc

if __name__ == '__main__':
    foundation_model_paths = [
        # 'Models/ResNetDiverse/Baseline_size=256-8_CTX768_0_963_965.pth',
        # 'Models/ResNetDiverse/Baseline_size=256-8_CTX768_10_971_968.pth',
        # 'Models/ResNetDiverse/Baseline_size=256-8_CTX768_50_974_961.pth',
        # 'Models/ResNetDiverse/Baseline_size=256-8_CTX768_5_960_972.pth',
        'Models/ResNetDiverse/Victim_size=256-8_CTX1024_100_957_967.pth',
        'Models/ResNetDiverse/Victim_size=256-8_CTX1024_500_923_956.pth',
    ]
    for foundation_model_path in foundation_model_paths:
        auroc_scores = []
        for fold_id in range(1):
            auroc_score = TrainModel(foundation_model_path, fold_id)
            auroc_scores.append(auroc_score)

        print('Average score:', np.mean(auroc_scores))