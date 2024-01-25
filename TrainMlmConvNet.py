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
from torch.utils.tensorboard import SummaryWriter

from MultiDataset import SlimPajamaDataset, MultiDataset
from TokenResNet import BasicBlock, Bottleneck, ResNet, BasicDecoder, AutoEncoder

def GetPilePath():
    local_path = '/mnt/data01/data/PileCompletions'
    possibly_remote_path = 'data/PileCompletions'
    if os.path.exists('/mnt/data01/data/PileCompletions'):
        path = local_path
    else:
        path = possibly_remote_path

    return path

def SetupDataLoaders(batch_size, max_sequence_length, total_sample_count, mask_proportion):  
    # CREATE TRAINING DATASETS.
    slim_pajama = SlimPajamaDataset(total_sample_count)
    
    tokenizer = f'Tokenizers/tokenmonster/vocabs/english-{24000}-strict-v1.vocab'
    training_dataset = MultiDataset(
        datasets = [slim_pajama],
        sampling_proportions = [1],
        tokenizer = tokenizer,
        max_sequence_length = max_sequence_length,
        fixed_order_data_augmentation_steps = [],
        rand_order_data_augmentation_steps = [],
        mask_proportion = mask_proportion,
        samples_per_epoch = total_sample_count
    )

    # CREATE DATALOADERS.
    data_collator = DataCollatorWithPadding(tokenizer=training_dataset.Tokenizer) if type(training_dataset.Tokenizer) != str else None
    train_data_loader = DataLoader(
        training_dataset, 
        batch_size=batch_size, 
        num_workers=1,
        pin_memory=True,
        collate_fn=data_collator)
    
    return train_data_loader

def TrainModel(max_learning_rate, training_document_count, block_type = BasicBlock, layers = [2, 2, 2, 2], device = torch.device('cuda:0'), mask_proportion = 0.15):
    # SETUP DATALOADER.
    MAX_SEQUENCE_LENGTH_TOKENS = 768
    BATCH_SIZE = 64
    train_data_loader = SetupDataLoaders(
        batch_size=BATCH_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH_TOKENS,
        total_sample_count=training_document_count,
        mask_proportion = mask_proportion)

    # CREATE MODEL.   
    VOCAB_SIZE = 24000
    base_width = 256
    width_coefs=[1, 1, 2, 2]
    encoder = ResNet(
        # +2 for padding & mask tokens.
        vocab_size=(VOCAB_SIZE + 2),
        block=block_type, 
        layers=layers,
        width_coefs=width_coefs,
        num_classes=2, 
        token_embedding_dim_count = base_width,
        base_width = base_width,
        return_raw_features=True)
    decoder = BasicDecoder(in_channels = base_width * width_coefs[-1], out_channels = (VOCAB_SIZE + 2))
    auto_encoder = AutoEncoder(encoder, decoder).to(device)

    # LOAD PRETRAINED AUTOENCODER.
    auto_encoder.load_state_dict(torch.load('Models/AutoEncoders/temp_auto_encoder_BasicBlock_[2, 2, 2, 2]_2000_0.pth'))

    # SETUP TENSORBOARD.
    summary_writer = SummaryWriter()

    # SETUP TRAINING ALGORITHM.
    optimizer = optim.AdamW(auto_encoder.parameters(), lr=max_learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01, reduction='none')
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
    auto_encoder.train()
    batch_count = 0
    for batch in tqdm(train_data_loader):
        # Have shape (batch size, token count)
        masked_token_sequences = batch['masked_input_ids'].to(device)
        token_sequences = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device) if 'attention_mask' in batch.keys() else None

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = auto_encoder(masked_token_sequences)

            losses = criterion(output, token_sequences)
            masked_losses = losses * attention_masks
            loss = torch.sum(masked_losses) / torch.sum(attention_masks)

        if batch_count == 10000:
            print(token_sequences)
            print(masked_token_sequences)
            # print(output)
            print(torch.argmax(output, dim = 1))
            print(token_sequences.shape, output.shape)
        
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(auto_encoder.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        lr_schedule.step()
        
        summary_writer.add_scalar('Loss/Train', loss.detach().cpu(), batch_count)
        # summary_writer.add_scalar('Loss/Test', loss.detach().cpu(), batch_count)

        firm_predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
        correct_prediction_count = np.sum((firm_predictions == token_sequences.detach().cpu().numpy()) * attention_masks.detach().cpu().numpy())
        accuracy = correct_prediction_count / np.sum(attention_masks.detach().cpu().numpy())
        summary_writer.add_scalar('Accuracy/Train', accuracy, batch_count)
        # summary_writer.add_scalar('Accuracy/Test', accuracy, batch_count)

        batch_count += 1

    # # SAVE THE MODEL.
    name_suffix = f'{block_type.__name__}_{str(layers)}_{int(max_learning_rate*1e5)}_{int(mask_proportion * 100)}_{training_document_count}'
    torch.save(auto_encoder.state_dict(), f'Models/AutoEncoders/temp_auto_encoder_{name_suffix}.pth')
    torch.save(auto_encoder.encoder.state_dict(), f'Models/AutoEncoders/temp_encoder_{name_suffix}.pth')

if __name__ == '__main__':
    # TrainModel(max_learning_rate=0.01, layers = [2, 2, 2, 2], training_document_count = 5_000_000) # LR optimal within factor +/- 2.
    # TrainModel(max_learning_rate=0.01, layers=[2, 2, 2, 2], training_document_count=50_000_000) # Need lower lr for larger dataset.

    # TrainModel(max_learning_rate=0.005, layers=[2, 2, 2, 2], training_document_count=50_000_000) # Running on 3090 0
    # TrainModel(max_learning_rate=0.0025, layers=[2, 2, 2, 2], training_document_count=50_000_000, device=torch.device('cuda:1')) # Running on 3090 1


    # TrainModel(max_learning_rate=0.02, layers = [2, 2, 2, 2], training_document_count = 500_000, mask_proportion = 0)
    TrainModel(max_learning_rate=0.01, layers = [2, 2, 2, 2], training_document_count = 5_000_000, mask_proportion = 0.15)
    TrainModel(max_learning_rate=0.005, layers = [2, 2, 2, 2], training_document_count = 5_000_000, mask_proportion = 0.15)