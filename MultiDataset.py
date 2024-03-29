# TODO: 
#   1) Strip leading/trailing whitespace off of text! Might overfit to that.
#   2) Normalize placeholders.
#   3) Add data augmentation for removal of leading/trailing lines (trailing line == everything after last period).
#   4) Investigate why cross-validation loss is so much higher than train.
#       - Is it reproducible when only testing on the pile?
#       - Are you sure unique samples are being selected?

import glob
import json
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from functools import lru_cache
import re
from random import random
from transformers import AutoTokenizer
import datetime
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from spellchecker import SpellChecker
import calendar
import jamspell
from multiprocessing import Pool
import multiprocessing
import tokenmonster
import time
from datasets import load_dataset

# Shut up pointless repetitive warnings ("You're using a LongformerTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a 
# method to encode the text followed by a call to the `pad` method to get a padded encoding.")
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

def LoadJson(json_filepath):
    with open(json_filepath) as json_file:
        data = json.loads(json_file.read())

    return data

def PrefetchJson(json_filepaths):
    with Pool(16) as pool:
        # Use the multiprocessing pool to load JSON data in parallel
        json_data_list = pool.map(LoadJson, json_filepaths)
    
    # Create a lookup of filepaths to decoded JSON
    json_lookup = dict(zip(json_filepaths, json_data_list))
    return json_lookup

class PileCompletionsDataset(Dataset):
    def __init__(self, data_filepath_pattern, fold_ids, prefetch_data=False):
        all_data_filepaths = sorted(glob.glob(data_filepath_pattern))
        self.Filepaths = [
            filepath
            for filepath_index, filepath in enumerate(all_data_filepaths)
            if (filepath_index % 10) in fold_ids
        ]
        np.random.shuffle(self.Filepaths)

        if prefetch_data:
            self.JsonLookup = PrefetchJson(self.Filepaths)
        else:
            self.JsonLookup = None

    def __len__(self):
        # 2 completions per file, one organic + one generated.
        return len(self.Filepaths) * 2
    
    # Returns (text from last part of document, completion label)
    # Where label is 1 if the text was generated by a LLM, 0 if it originated
    # from The Pile.
    def __getitem__(self, index):
        completions_data_filepath = self.Filepaths[(index // 2) % len(self.Filepaths)]

        if self.JsonLookup:
            completions_data = self.JsonLookup[completions_data_filepath]
        else:
            completions_data = LoadJson(completions_data_filepath)
        
        use_generated_completion = ((index % 2) == 0)
        if use_generated_completion:
            return completions_data['GeneratedCompletion'], 1
        else:
            return completions_data['RealCompletion'], 0

class TrickyCrawlDataset(Dataset):
    def __init__(self, filepath):
        self.Data = pd.read_csv(filepath, lineterminator='\n')
    
    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        row = self.Data.iloc[index]

        full_document = row['text']
        completion_start_index = int(len(full_document) * np.random.uniform(0.15, 0.75))
        truncated_document = full_document[completion_start_index:]

        return truncated_document, 0

class PersuadeDataset(Dataset):
    # "The RDizzl3 Seven" - Assignments believed to be used to calculate the leaderboard scores.
    COMPETITION_ASSIGNMENTS = [
        '"A Cowboy Who Rode the Waves"', 
        'Car-free cities', 
        'Facial action coding system', 
        'Exploring Venus', 
        'The Face on Mars', 
        'Driverless cars', 
        'Does the electoral college work?'
    ]

    # Everything other than "The RDizzl3 Seven".
    OTHER_ASSIGNMENTS = [
        'Summer projects', 
        'Phones and driving', 
        'Mandatory extracurricular activities', 
        'Seeking multiple opinions', 
        'Community service', 
        'Cell phones at school', 
        'Distance learning', 
        'Grades for extracurricular activities', 
    ]

    def __init__(
            self, 
            human_essays_filepath,
            generated_essay_filepath_patterns, 
            generated_subset_sampling_proportions, 
            whitelisted_assignment_names,
            samples_per_epoch = None):
        all_human_essays_df = pd.read_csv(human_essays_filepath)
        filtered_essays_df = all_human_essays_df[all_human_essays_df['prompt_name'].isin(whitelisted_assignment_names)]
        self.OrganicEssaysText = filtered_essays_df['full_text']
        
        self.WhitelistedAssignmentNames = whitelisted_assignment_names
        
        if samples_per_epoch is not None:
            self.SamplesPerEpoch = samples_per_epoch
        else:
            self.SamplesPerEpoch = len(self.OrganicEssaysText) * 2

        # Normalized to be probabilities (0 - 1).
        self.SubsetSamplingProportions = np.array(generated_subset_sampling_proportions) / sum(generated_subset_sampling_proportions)

        self.GeneratedEssayFilepathsBySubset = []
        for data_filepath_pattern in generated_essay_filepath_patterns:
            subset_filepaths = glob.glob(data_filepath_pattern)
            assert len(subset_filepaths) > 0, f'No files found matching pattern "{data_filepath_pattern}"'
            self.GeneratedEssayFilepathsBySubset.append(subset_filepaths)

    def __len__(self):
        return self.SamplesPerEpoch
    
    def __GetOrganicEssay(self):
        essay = np.random.choice(self.OrganicEssaysText)
        return self.__NormalizeIntroAndEnd(essay)
    
    def __GetGeneratedEssay(self):
        subset_count = len(self.GeneratedEssayFilepathsBySubset)
        subset_id = np.random.choice(list(range(subset_count)), p = self.SubsetSamplingProportions)
        subset_filepaths = self.GeneratedEssayFilepathsBySubset[subset_id]
        essay_filepath = np.random.choice(subset_filepaths)

        essay_data = LoadJson(essay_filepath)
        essay_text = essay_data['EssayText']

        return self.__NormalizeIntroAndEnd(essay_text)

    def __TruncateEssay(self, essay):
        start_index = np.random.randint(0, len(essay) // 4)
        end_index = np.random.randint(3 * len(essay) // 4, len(essay))

        essay = essay[start_index : end_index]
        return essay

    def __RemoveAfterLastPeriod(self, essay):
        last_period_index = essay.rfind('.')
        if last_period_index != -1:
            return essay[:last_period_index]
        
        return essay

    def __DetectLineEndings(self, essay):
        if '\r\n' in essay:
            return '\r\n'
        elif '\n' in essay:
            return '\n'
        elif '\r' in essay:
            return '\r'
        else:
            return ''

    def __CheckIfContainsWhitelistedTerm(self, line, whitelist):
        for term in whitelist:
            if term in line:
                return True
            
        return False

    def __NormalizeIntroAndEnd(self, essay):
        if np.random.choice([True, False], p = [0.5, 0.5]):
            # Short cuircuite randomly to avoid overfitting to imperfect preprocessing.
            return essay

        # NORMALIZE INTRO.
        first_line_text = essay.splitlines()[0]
        intro_normalization_necessary = first_line_text[-1] not in ['.', '?', '!']
        if intro_normalization_necessary:
            line_endings = self.__DetectLineEndings(essay)
            truncated_essay = line_endings.join(essay.splitlines()[1:])

            # TODO: The competition organizers sometimes remove intros containing "to" or "dear", so this isn't a perfect imitation. Additional investigation may be beneficial.
            whitelisted_terms = ['dear', 'to'] + list(calendar.month_name)[1:]
            include_first_line_without_newline = self.__CheckIfContainsWhitelistedTerm(first_line_text, whitelisted_terms) or (first_line_text[-1] == '"')
            if include_first_line_without_newline:
                essay = first_line_text + " " + truncated_essay
            else:
                essay = truncated_essay

        # NORMALIZE END.
        essay = self.__RemoveAfterLastPeriod(essay)

        return essay

    def __getitem__(self, index):
        use_generated_essay = np.random.choice([True, False])
        if use_generated_essay:
            MAX_RETRY_COUNT = 5
            essay = None
            for _ in range(MAX_RETRY_COUNT - 1):
                try:
                    # The generation script got interrupted a few times, so a tiny fraction of the
                    # files are invalid. As a result, this sometimes needs to be retried with a different
                    # randomly selected data file. The probability of it failing 5 times in a row
                    # should be astrinomically low.
                    essay = self.__GetGeneratedEssay()
                except:
                    continue

            if essay is None:
                essay = self.__GetGeneratedEssay()

            return essay, 1
        else:
            return self.__GetOrganicEssay(), 0

class SlimPajamaDataset(Dataset):
    def __init__(self, samples_per_epoch, cache_dir = '/mnt/data02/datasets/SlimPajama'):
        if not os.path.exists(cache_dir):
            print('WARNING: Cache directory does not exist. Ignoring user specified path.')
            cache_dir = None

        self.Dataset = iter(load_dataset("cerebras/SlimPajama-627B", cache_dir=cache_dir, streaming=True, split='test'))
        self.SamplesPerEpoch = samples_per_epoch

    def __len__(self):
        return self.SamplesPerEpoch
    
    def __getitem__(self, index):
        # return "This is a test." * 100, 0

        try:
            text = next(self.Dataset)['text']
        except:
            # One retry to gracefully handle HTTP connection errors while streaming.
            time.sleep(1)
            text = next(self.Dataset)['text']

        return text, 0

class MultiDataset:
    # Data augmentation steps like:
    # [
    #     {'name' : 'BuggySpellCheck', 'p' : 0.05 },
    #     {'name' : 'RemoveBlacklistedCharacters', 'p' : 0.3 },
    #     {'name' : 'char_swap', 'p' : 0.1 },
    #     {'name' : 'missing_char', 'p' : 0.1 },
    #     {'name' : 'extra_char', 'p' : 0.1 },
    #     {'name' : 'nearby_char', 'p' : 0.1 },
    #     {'name' : 'similar_char', 'p' : 0.1 },
    #     {'name' : 'skipped_space', 'p' : 0.1 },
    #     {'name' : 'random_space', 'p' : 0.1 },
    #     {'name' : 'repeated_char', 'p' : 0.1 },
    #     {'name' : 'unichar', 'p' : 0.1 },
    # ]
    # 
    # Can list same step multiple times to sometimes do it repeatedly. Data augmentation steps will be executed
    # in a random order.
    def __init__(
            self, 
            datasets, 
            sampling_proportions, 
            tokenizer, 
            max_sequence_length, 
            fixed_order_data_augmentation_steps, 
            rand_order_data_augmentation_steps, 
            mask_proportion = 0,
            samples_per_epoch = None,
            random_crop_length_chars = None):
        self.Datasets = datasets
        self.SampleCountsPerRound = (np.array(sampling_proportions) * 1000).astype(np.int64)

        self.FixedOrderDataAugmentationSteps = fixed_order_data_augmentation_steps
        self.RandOrderDataAugmentationSteps = rand_order_data_augmentation_steps
        self.MaskProportion = mask_proportion
        self.SlowSpellChecker = SpellChecker()

        self.FastSpellChecker = jamspell.TSpellCorrector()
        assert self.FastSpellChecker.LoadLangModel('data/en.bin')

        self.UsingTokenmonster = False
        self.Tokenizer = tokenizer

        self.MaxSequenceLength = max_sequence_length
        self.RandomCropLengthChars = random_crop_length_chars
        
        if samples_per_epoch is not None:
            self.SamplesPerEpoch = samples_per_epoch
        else:
            lengths = [len(dataset) for dataset in self.Datasets]
            self.SamplesPerEpoch = sum(lengths)


    def __len__(self):
        return self.SamplesPerEpoch

    def __GetCorrectedWord(self, word):
        candidates = self.FastSpellChecker.GetCandidates([word], 0)
        if len(candidates) == 0:
            return None

        return np.random.choice(list(candidates)[:5])

    # Based on https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/456142
    # but uses better regex that appears to handle ' the same way as the competition organizers & limits the number
    # of replacements to avoid ever taking a horrendously long time to run (pyspellchecker is slow).
    # Intentionally buggy to imitate how the competition organizers appear to have done their preprocessing.
    def __BuggyCorrectTypos(self, text):   
        # Tokenize the text into words
        words = re.findall(r"\b[\w|']+\b", text)

        # Find misspelled words. Truncated for runtime reasons (not the bug).
        misspelled = self.SlowSpellChecker.unknown(words)
        corrected_typo_count = min(10, len(misspelled))
        misspelled = np.random.choice(list(misspelled), corrected_typo_count, replace=10)

        # Correct the misspelled words
        corrected_text = text
        for word in misspelled:
            correction = self.__GetCorrectedWord(word)
            if correction:
                # Doing replace-alls like this is an intentional bug. Sometimes single letters get detected
                # as words and replaced throughout the document, thereby resulting in garbled text.
                corrected_text = corrected_text.replace(word, correction)

        return corrected_text

    # Based on https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452172
    # Removes anything in PERSUADE dataset that doesn't appear in "train.csv" from the competition organizers.
    def __RemoveBlacklistedCharacters(self, text):
        blacklisted_characters = ['\x94', ']', 'á', '¹', '`', 'å', '~', '}', 'Ö', '\\', '=', '\x97', '(', '©', '²', ')', '\x91', '>', '®', ';', '<', '£', '+', '#', '¶', '\xa0', '{', '^', '\x80', '[', '|', '\x93', '-', '\x85', 'Ó', '*', '/', '$', 'é', 'ó', '\x99']
        translation_table = str.maketrans('', '', ''.join(blacklisted_characters))
        cleaned_text = text.translate(translation_table)

        return cleaned_text
    
    def __DecapitalizeRandomLetter(self, text):
        capital_indices = [i for i, char in enumerate(text) if char.isupper()]
        if len(capital_indices) == 0:
            return text
        
        random_index = np.random.choice(capital_indices)

        modified_text = text[:random_index] + text[random_index].lower() + text[random_index + 1:]
        
        return modified_text

    def __CapitalizeRandomLetter(self, text):
        lower_indices = [i for i, char in enumerate(text) if char.islower()]
        if len(lower_indices) == 0:
            return text
        
        random_index = np.random.choice(lower_indices)

        modified_text = text[:random_index] + text[random_index].upper() + text[random_index + 1:]
        
        return modified_text

    def __getitem__(self, item_index):
        if type(self.Tokenizer) == str:
            self.UsingTokenmonster = True
            self.VocabSize = int(self.Tokenizer.split('-')[1]) # Not safe. Assumes "-" only appears in filename & vocap size is second compoent.
            self.Tokenizer = tokenmonster.load(self.Tokenizer)

        round_index = item_index // self.SampleCountsPerRound.sum()
        intra_round_index = item_index % self.SampleCountsPerRound.sum()
        
        cumulative_sample_sum = 0
        for dataset_index in range(len(self.Datasets)):
            cumulative_sample_sum += self.SampleCountsPerRound[dataset_index]
            if intra_round_index < cumulative_sample_sum:
                break

        dataset = self.Datasets[dataset_index]
        intra_dataset_item_index = (round_index * self.SampleCountsPerRound[dataset_index]) + (intra_round_index - (cumulative_sample_sum - self.SampleCountsPerRound[dataset_index]))
        text, label = dataset[intra_dataset_item_index]

        # print(f'INFO: Using dataset {dataset_index} (item {intra_dataset_item_index}) for global item {item_index}')

        text = text.strip()

        np.random.shuffle(self.RandOrderDataAugmentationSteps)
        for augmentation_step in (self.FixedOrderDataAugmentationSteps + self.RandOrderDataAugmentationSteps):
            augmentation_prob = augmentation_step['p']
            if type(augmentation_prob) is list:
                augmentation_prob = augmentation_prob[dataset_index]

            skip_step = np.random.uniform(0, 1) > augmentation_prob
            if skip_step:
                continue

            if augmentation_step['name'] == 'BuggySpellCheck':
                text = self.__BuggyCorrectTypos(text)
            elif augmentation_step['name'] == 'RemoveBlacklistedCharacters':
                text = self.__RemoveBlacklistedCharacters(text)
            elif augmentation_step['name'] == 'DecapitalizeRandomLetter':
                text = self.__DecapitalizeRandomLetter(text)
            elif augmentation_step['name'] == 'CapitalizeRandomLetter':
                text = self.__CapitalizeRandomLetter(text)
            else:
                error_type_name = augmentation_step['name']
                try:
                    text = eval(f'typo.StrErrer(text).{error_type_name}().result')
                except:
                    # Sometimes randomly fails for characters that don't have a keyboard neighbor.
                    pass

        if self.RandomCropLengthChars is not None:
            max_start_index = max(1, len(text) - self.RandomCropLengthChars//2)
            start_index = np.random.randint(0, max_start_index)
            end_index = start_index + self.RandomCropLengthChars

            text = text[start_index:end_index]

        if not self.UsingTokenmonster:
            tokenization_result = self.Tokenizer(text, max_length = self.MaxSequenceLength, truncation = True)
            attention_mask = tokenization_result['attention_mask']
            token_ids = tokenization_result['input_ids']
        else:
            try:
                token_ids = self.Tokenizer.tokenize(text)[:self.MaxSequenceLength]
                attention_mask = np.ones_like(token_ids, dtype=np.int64)
            except:
                token_ids = []
                attention_mask = []

            padded_token_ids = np.ones(self.MaxSequenceLength, dtype = np.int64) * self.VocabSize
            output_token_count = min(self.MaxSequenceLength, len(token_ids))
            padded_token_ids[:output_token_count] = token_ids[:output_token_count]

            padded_attention_mask = np.zeros(self.MaxSequenceLength, dtype = np.int64)
            padded_attention_mask[:output_token_count] = attention_mask[:output_token_count]

            token_ids = padded_token_ids

        outputs = {
            'input_ids' : token_ids, 
            'data_origin_id' : dataset_index,
            'is_artificially_generated' : label,
            'index' : item_index,
            'attention_mask' : padded_attention_mask if self.UsingTokenmonster else attention_mask
        }

        # print(token_ids)
        if self.MaskProportion > 0:
            non_padding_token_count = np.sum(attention_mask)
            mask_count = int(non_padding_token_count * self.MaskProportion)
            mask_indices = np.random.choice(list(range(non_padding_token_count)), mask_count, replace=False)
            
            masked_token_ids = np.array(token_ids)
            masked_token_ids[mask_indices] = self.VocabSize + 1

            outputs['masked_input_ids'] = masked_token_ids

        return outputs

def PrintStats(dataset, length_limit = None):    
    organic_text_lengths = []
    generated_text_lengths = []
    for completion, label in tqdm(dataset):
        if label == 1:
            generated_text_lengths.append(len(completion))
        else:
            organic_text_lengths.append(len(completion))

        if (length_limit is not None) and (len(organic_text_lengths) >= length_limit//2):
            break

    print('Organic (min, 5th, median, 95th, max): {:.1f},  {:.1f},  {:.1f},  {:.1f},  {:.1f}'.format(
        min(organic_text_lengths), 
        np.percentile(organic_text_lengths, 5),
        np.percentile(organic_text_lengths, 50),
        np.percentile(organic_text_lengths, 95),
        max(organic_text_lengths)))

    print('Generated (min, 5th, median, 95th, max): {:.1f},  {:.1f},  {:.1f},  {:.1f},  {:.1f}'.format(
        min(generated_text_lengths), 
        np.percentile(generated_text_lengths, 5),
        np.percentile(generated_text_lengths, 50),
        np.percentile(generated_text_lengths, 95),
        max(generated_text_lengths)))

def OldTestMain():
    print('Creating pile dataset...')
    pile_dataset = PileCompletionsDataset('data/PileCompletions/*/*.json', [0, 1, 2, 3])
    
    print('Creating persuade dataset...')
    persuade_dataset = PersuadeDataset(
        human_essays_filepath = 'data/PERSUADE/persuade_2.0_human_scores_demo_id_github.csv',
        generated_essay_filepath_patterns = [
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
        ], 
        generated_subset_sampling_proportions = [
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
        ], 
        whitelisted_assignment_names = PersuadeDataset.COMPETITION_ASSIGNMENTS,
        # whitelisted_assignment_names = PersuadeDataset.OTHER_ASSIGNMENTS,
        samples_per_epoch = 50000
    )

    print('Creating multi dataset...')
    
    FIXED_ORDER_DATA_AUGMENTATION_STEPS = [
        {'name' : 'BuggySpellCheck', 'p' : [0.7, 0.2] },
        {'name' : 'RemoveBlacklistedCharacters', 'p' : [0.7, 0.2] },
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
    training_persuade_dataset = PersuadeDataset(
        human_essays_filepath = 'data/PERSUADE/persuade_2.0_human_scores_demo_id_github.csv',
        generated_essay_filepath_patterns = GENERATED_ESSAY_FILEPATH_PATTERNS, 
        generated_subset_sampling_proportions = GENERATED_ESSAY_SUBSET_SAMPLING_PROPORTIONS, 
        whitelisted_assignment_names = PersuadeDataset.COMPETITION_ASSIGNMENTS,
    )

    local_path = '/mnt/data01/data/PileCompletions'
    possibly_remote_path = 'data/PileCompletions'
    if os.path.exists('/mnt/data01/data/PileCompletions'):
        path = local_path
    else:
        path = possibly_remote_path
    training_pile_dataset = PileCompletionsDataset(
        f'{path}/*/*.json', 
        [i for i in range(10) if i != 0])
    
    training_dataset = MultiDataset(
        datasets = [training_persuade_dataset, training_pile_dataset],
        sampling_proportions = [0.02, 0.98],
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', Use_fast=False),
        # tokenizer = f'Tokenizers/tokenmonster/vocabs/english-{24000}-strict-v1.vocab',
        max_sequence_length = 768,
        fixed_order_data_augmentation_steps = FIXED_ORDER_DATA_AUGMENTATION_STEPS,
        rand_order_data_augmentation_steps = RAND_ORDER_DATA_AUGMENTATION_STEPS,
        samples_per_epoch = 50000
    )
    data_collator = DataCollatorWithPadding(tokenizer=training_dataset.Tokenizer) if type(training_dataset.Tokenizer) != str else None
    train_data_loader = DataLoader(
        training_dataset, 
        batch_size=4, 
        num_workers=1, #multiprocessing.cpu_count(), 
        # prefetch_factor=20,
        shuffle=False,
        pin_memory=True,
        collate_fn=data_collator 
    )

    # element_count = 0
    # for tokenized_essay in tqdm(training_dataset):
    #     element_count += 1

    #     if element_count > 10000:
    #         break

    batch_count = 0
    #indices = []
    for batch in tqdm(train_data_loader):
        #indices.extend(batch['index'].numpy())
        batch_count += 1

    # print('\n\nRESTARTING...\n\n')
    # batch_count = 0
    # #indices = []
    # for batch in train_data_loader:
    #     #indices.extend(batch['index'].numpy())
    #     batch_count += 1
    #     break

if __name__ == '__main__':
    test_dataset = TrickyCrawlDataset('data/TrickyCrawl/1500000_125192.csv')
    print(len(test_dataset))
    print(test_dataset[0])
    print(test_dataset[1])