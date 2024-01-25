import tokenmonster
import pickle
import numpy as np
import tqdm
from torch.utils.data import Dataset

class PersuadeDataset(Dataset):
    def __init__(self, preprocessed_data_cache_filepath, vocab_size, output_seq_length, fold_ids):
        self.VocabSize = vocab_size
        self.Tokenizer = None
        self.OutputSequenceLength = output_seq_length

        with open('data/Cache/Persuade_V0_BasicPreprocessing_50kGenerated_MetadataIncluded.p', 'rb') as train_data_file:
            essays_text, essays_metadata, essay_labels = pickle.load(train_data_file)

        self.EssaysText, self.EssaysMetadata, self.EssayLabels = [], [], []
        for essay_index in range(len(essays_text)):
            if (essay_index % 5) not in fold_ids:
                continue

            if len(essays_text[essay_index]) == 0:
                continue

            self.EssaysText.append(essays_text[essay_index])
            self.EssaysMetadata.append(essays_metadata[essay_index])
            self.EssayLabels.append(essay_labels[essay_index])

    def __len__(self):
        return len(self.EssaysText)
    
    def __getitem__(self, index):
        # POSSIBLY INITIALIZE TOKENIZER.
        # This isn't done when the dataset is first created because initializing it before forking subprocesses seems to cause problems 
        # (data corruption when using load, hanging when using load_multiprocess_save).
        if self.Tokenizer is None:
            self.Tokenizer = tokenmonster.load(f'Tokenizers/tokenmonster/vocabs/english-{self.VocabSize}-strict-v1.vocab')

        # LOAD TOKENIZED ESSAY.
        essay = self.EssaysText[index]
        token_ids = self.Tokenizer.tokenize(essay)

        # PAD OR TRUNCATE ESSAY.
        # The vocab size is used as the padding token ID because it is one more than
        # the max value that can be used in the actual text.
        padded_token_ids = np.ones(self.OutputSequenceLength) * self.VocabSize
        output_token_count = min(self.OutputSequenceLength, len(token_ids))
        padded_token_ids[:output_token_count] = token_ids[:output_token_count]

        # LOAD ESSAY LABEL.
        label = np.int32(self.EssayLabels[index])

        return padded_token_ids, label

if __name__ == '__main__':
    dataset = PersuadeDataset(
        preprocessed_data_cache_filepath = 'data/Cache/Persuade_V0_BasicPreprocessing_50kGenerated_MetadataIncluded.p', 
        vocab_size = 24000, 
        output_seq_length = 1024, 
        fold_ids = [0, 1, 2, 3])
    
    for padded_token_ids, label in dataset:
        print(padded_token_ids[:5], padded_token_ids[-5:], label)
