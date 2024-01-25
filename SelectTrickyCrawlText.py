import glob
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

def LoadManyDocuments(pile_subset_names, document_count):
    documents = []
    pile_slice_filepaths = glob.glob('/mnt/data01/LLMs/Data/ThePile/Uncompressed/*.jsonl')
    for pile_slice_filepath in tqdm(pile_slice_filepaths):
        with open(pile_slice_filepath) as pile_slice_file:
            for line_index, line in enumerate(pile_slice_file):
                # CHECK IF THE DOCUMENT IS IN A SUBSET WE CARE ABOUT.
                pile_record = json.loads(line)
                subset_name = pile_record['meta']['pile_set_name']
                if subset_name not in pile_subset_names:
                    continue

                # RECORD TEXT.
                documents.append(pile_record['text'])

                # DISPLAY PROGRESS.
                if len(documents) % 1000 == 0:
                    print(f'Loaded {len(documents)}/{document_count} documents.')

                # RETURN IF WE HAVE ENOUGH DOCUMENTS.
                if len(documents) >= document_count:
                    return documents

    return documents

class SimpleTestDataset(Dataset):
    def __init__(self, strings, tokenizer, max_sequence_length):
        self.Strings = strings
        self.Tokenizer = tokenizer
        self.MaxSequenceLength = max_sequence_length

    def __len__(self):
        return len(self.Strings)
    
    def __getitem__(self, idx):
        string = self.Strings[idx].strip()
        token_ids = self.Tokenizer(string, max_length = self.MaxSequenceLength, truncation = True).input_ids

        return {
            'input_ids' : token_ids,
            'document_index' : idx,
        }

if __name__ == '__main__':
    classifiers = [
        # Trained on 1 million Pile & Persuade examples.
        AutoModelForSequenceClassification.from_pretrained(
            'microsoft/deberta-v3-large',
            state_dict = torch.load('DomainAdaptation/UnadaptedModels/E20_CTX1024_10_991_994_Submitted960.pth')).eval().cuda(),
        # Trained on 250k Pile, Persuade, and SlimPajama examples.
        AutoModelForSequenceClassification.from_pretrained(
            'microsoft/deberta-v3-xsmall',
            state_dict = torch.load('Models/Transformer/3090_0/S15625_CTX1024_10_983_986_ForFiltering.pth')).eval().cuda()
    ]

    DOCUMENT_COUNT = 1_500_000
    documents = LoadManyDocuments(['Pile-CC'], DOCUMENT_COUNT)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
    dataset = SimpleTestDataset(
        strings = documents, 
        tokenizer = tokenizer, 
        max_sequence_length = 1024)

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=1, 
        collate_fn=DataCollatorWithPadding(tokenizer))

    misclassified_documents = []
    victim_scores_by_model = [[] for _ in classifiers]
    for batch in tqdm(data_loader):
        token_sequences = batch.input_ids.cuda()
        attention_masks = batch.attention_mask.cuda()

        predictions_by_model = []
        with torch.no_grad():
            for classifier in classifiers:
                with torch.cuda.amp.autocast():
                    raw_predictions = classifier(token_sequences, attention_masks).logits
                
                scaled_predictions = raw_predictions.softmax(dim = 1)[:,1].cpu().numpy()
                predictions_by_model.append(scaled_predictions)

        stored_document_count = 0
        for batch_document_index, global_document_index in enumerate(batch.document_index):
            document_predictions = [model_predictions[batch_document_index] for model_predictions in predictions_by_model]
            if max(document_predictions) > 0.5:
                misclassified_documents.append(documents[global_document_index])

                for model_index, model_predictions in enumerate(predictions_by_model):
                    victim_scores_by_model[model_index].append(model_predictions[batch_document_index])
                
                stored_document_count += 1

    tricky_documents = pd.DataFrame({
        'text' : misclassified_documents,
        'classifier_0_score' : victim_scores_by_model[0],
        'classifier_1_score' : victim_scores_by_model[1]
    })
    tricky_documents.to_csv(f'TrickyCrawl/{DOCUMENT_COUNT}_{len(misclassified_documents)}.csv', index = False)