import numpy as np
from torch.utils.data import Dataset

class PersuadeDataset(Dataset):
    def __init__(self, tokenizer, max_sequence_length):
        self.EssaysDf = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
        self.EssaysText = self.EssaysDf['text']
        self.Ids = np.array(self.EssaysDf['id'])
        
        self.MaxSequenceLength = max_sequence_length
        self.Tokenizer = tokenizer
        
    def __len__(self):
        return len(self.EssaysText)
    
    def __getitem__(self, index):
        text = self.EssaysText[index]
        #essay_id = self.Ids[index]
        
        #start_index = len(text) // 10
        #end_index = 9 * len(text) // 10
        #text = text[start_index:end_index]
        
        text = text.strip()
        
        token_ids = self.Tokenizer(text, max_length = self.MaxSequenceLength, truncation = True).input_ids
        
        return {
            'input_ids' : token_ids,
            'essay_index' : index
            #'essay_id' : essay_id,
        }
