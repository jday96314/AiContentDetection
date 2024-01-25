import numpy as np
import pandas as pd
#import language_tool_python
from concurrent.futures import ProcessPoolExecutor
import csv
import glob
from spellchecker import SpellChecker
import re
from typing import List
import json
from tqdm import tqdm
import os
import pickle
from multiprocessing import Pool
from functools import lru_cache
import string

spell_checker = SpellChecker()
@lru_cache(maxsize=30000)
def GetCorrectWord(word):
    return spell_checker.correction(word)

# Based on https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/456142
# but uses better regex that appears to handle ' the same way as the competition organizers.
def BuggyCorrectTypos(text):   
    # Tokenize the text into words
    words = re.findall(r"\b[\w|']+\b", text)

    # Find misspelled words
    misspelled = spell_checker.unknown(words)

    # Correct the misspelled words
    corrected_text = text
    for word in misspelled:
        correction = GetCorrectWord(word)
        if correction:
            corrected_text = corrected_text.replace(word, correction)

    return corrected_text

# Based on https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452172
# Removes anything in PERSUADE dataset that doesn't appear in "train.csv" from the competition organizers.
def RemoveBlacklistedCharacters(text):
    blacklisted_characters = ['\x94', ']', 'á', '¹', '`', 'å', '~', '}', 'Ö', '\\', '=', '\x97', '(', '©', '²', ')', '\x91', '>', '®', ';', '<', '£', '+', '#', '¶', '\xa0', '{', '^', '\x80', '[', '|', '\x93', '-', '\x85', 'Ó', '*', '/', '$', 'é', 'ó', '\x99']
    translation_table = str.maketrans('', '', ''.join(blacklisted_characters))
    cleaned_text = text.translate(translation_table)

    return cleaned_text

def PreprocessEssay(essay):
    essay = essay.strip()
    essay = BuggyCorrectTypos(essay)
    essay = RemoveBlacklistedCharacters(essay)

    return essay

def CreateCacheFile(output_filepath):
    preprocessed_essays = []
    raw_essay_data = []
    labels = []
    with open('data/PERSUADE/persuade_2.0_human_scores_demo_id_github.csv') as human_essays_file:
        essays_reader = csv.DictReader(human_essays_file)
        for row in tqdm(essays_reader):
            preprocessed_essays.append(row['full_text'])
            raw_essay_data.append(row)
            labels.append(0)

    generated_essay_filepaths = glob.glob('data/GeneratedEssays/*/*.json')
    selected_essay_filepaths = np.random.choice(generated_essay_filepaths, 50000, replace=False)
    for generated_essay_filepath in tqdm(selected_essay_filepaths):
        with open(generated_essay_filepath) as essay_file:
            try:
                essay_data = json.loads(essay_file.read())
            except:
                print(f'ERROR: Failed to load data from {generated_essay_filepath}')

        preprocessed_essays.append(essay_data['EssayText'])
        raw_essay_data.append(essay_data)
        labels.append(1)

    with Pool(16) as worker_pool:
        preprocessed_essays = worker_pool.map(PreprocessEssay, preprocessed_essays)

    with open(output_filepath, 'wb') as cache_file:
        pickle.dump((preprocessed_essays, raw_essay_data, labels), cache_file)

    return preprocessed_essays, raw_essay_data, labels

def CreateCsvFile(output_filepath):
    column_names_to_values = {
        'Assigment': [],
        'AssignmentName': [],
        'SourceTextIncluded': [],
        'ShotCount': [],
        'RequestPrefix': [],
        'SamplingConfig': [],
        'EssayText': [],
        'OriginalFilepath': []
    }

    generated_essay_filepaths = glob.glob('data/GeneratedEssays_AdversarialPersuade/**/*.json', recursive=True) + glob.glob('data/GeneratedEssays/*/*.json')
    print(len(generated_essay_filepaths))
    for generated_essay_filepath in tqdm(generated_essay_filepaths):
        with open(generated_essay_filepath) as essay_file:
            try:
                essay_data = json.loads(essay_file.read())
            except:
                print(f'ERROR: Failed to load data from {generated_essay_filepath}')
                continue

        # assert len(essay_data.keys()) == len(column_names_to_values.keys())

        for column_name in [key for key in column_names_to_values.keys() if key != 'OriginalFilepath']:
            try:
                column_names_to_values[column_name].append(essay_data[column_name])
            except:
                try:
                    column_names_to_values[column_name].append(essay_data['SamplingConfig'][column_name])
                except:
                    column_names_to_values[column_name].append(essay_data['PromptDetails'][column_name])
        
        column_names_to_values['OriginalFilepath'].append(generated_essay_filepath)

    # FIX TYPO IN COLUMN NAME.
    # Propigated to initial list for simplicity & consistency with underlying data files, but should be fixed in output files.
    column_names_to_values['Assignment'] = column_names_to_values['Assigment']
    del column_names_to_values['Assigment']

    for column_name in column_names_to_values.keys():
        print(f'{column_name}: {len(column_names_to_values[column_name])}')

    # SAVE TO DISK.
    df = pd.DataFrame.from_dict(column_names_to_values)
    df.to_csv(output_filepath, index=False)

def GenerateFakeTestFile(document_count, output_filepath):
    dataset = {
        'id' : [],
        'prompt_id' : [],
        'text' : []
    }

    filepaths = np.random.choice(glob.glob('data/PileCompletions/*/*.json'), document_count, replace=False)
    for data_file in tqdm(filepaths):
        with open(data_file) as file:
            data = json.loads(file.read())

        possible_id_chars = list(string.ascii_lowercase + string.digits)
        id = ''.join(np.random.choice(possible_id_chars, size = 8))
        dataset['id'].append(id)

        prompt_id = np.random.randint(1, 8)
        dataset['prompt_id'].append(prompt_id)

        text = data['GeneratedCompletion'] if (prompt_id % 2 == 1) else data['RealCompletion']
        dataset['text'].append(text)

    dataset = pd.DataFrame.from_dict(dataset)
    dataset.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    #CreateCsvFile('data/Cache/Persuade_V0_NoPreprocessing_100kGenerated_MetadataIncluded.csv')

    #GenerateFakeTestFile(10000, 'data/Cache/FakeTestFile.csv')

    df = pd.read_csv('data/Cache/Persuade_V0_NoPreprocessing_100kGenerated_MetadataIncluded.csv', lineterminator='\n')
    print(len(df))
    print(df.head())