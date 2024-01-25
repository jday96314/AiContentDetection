import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import make_scorer, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
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
from pqdm.threads import pqdm
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import optuna

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

def LoadData(cache_filepath):
    use_cache = (cache_filepath is not None) and (os.path.exists(cache_filepath))
    if use_cache:
        with open(cache_filepath, 'rb') as cache_file:
            essays, labels = pickle.load(cache_file)
            return essays, labels

    essays = []
    labels = []

    with open('data/PERSUADE/persuade_2.0_human_scores_demo_id_github.csv') as human_essays_file:
        essays_reader = csv.DictReader(human_essays_file)
        for row in tqdm(essays_reader):
            essays.append(row['full_text'])
            labels.append(0)

    generated_essay_filepaths = glob.glob('data/GeneratedEssays/*/*.json')
    selected_essay_filepaths = np.random.choice(generated_essay_filepaths, 30000, replace=False)
    for generated_essay_filepath in tqdm(selected_essay_filepaths):
        with open(generated_essay_filepath) as essay_file:
            try:
                essay_data = json.loads(essay_file.read())
            except:
                print(f'ERROR: Failed to load data from {generated_essay_filepath}')

        essays.append(essay_data['EssayText'])
        labels.append(1)

    with Pool(16) as worker_pool:
        essays = worker_pool.map(PreprocessEssay, essays)

    with open(cache_filepath, 'wb') as cache_file:
        pickle.dump((essays, labels), cache_file)

    return essays, labels

def RunExperiment(hparams):
    # print('Loading data...')
    essays, labels = LoadData('data/Cache/Persuade_V0_BasicPreprocessing_30kGenerated.p')

    # vectorizer = TfidfVectorizer(ngram_range=(3, 7),sublinear_tf=True)
    vectorizer = TfidfVectorizer(sublinear_tf=True)

    # model = LogisticRegression(max_iter=1000, C = 0.5)
    model = LogisticRegression(max_iter=1000)
    # model = XGBClassifier(
    #     max_depth = 6, 
    #     n_estimators = 100, 
    #     subsample = 1,
    #     reg_alpha = 0,
    #     reg_lambda = 0)

    # model = VotingClassifier(
    #     estimators = [
    #         ('lr', LogisticRegression(
    #             max_iter = 1000, 
    #             C = hparams['lr']['C'])),
    #         ('xgb', XGBClassifier(
    #             max_depth = hparams['xgb']['max_depth'], 
    #             n_estimators = hparams['xgb']['n_estimators'], 
    #             learning_rate = hparams['xgb']['learning_rate'],
    #             subsample = hparams['xgb']['subsample'],
    #             reg_alpha = hparams['xgb']['reg_alpha'],
    #             reg_lambda = hparams['xgb']['reg_lambda']))
    #     ],
    #     weights = hparams['weights'],
    #     voting = 'soft'
    # )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_index, val_index) in enumerate(skf.split(essays, labels), 1):
        # print(f'Processing fold {fold}/5...')

        train_essays, validation_essays = np.array(essays)[train_index], np.array(essays)[val_index]
        train_labels, validation_labels = np.array(labels)[train_index], np.array(labels)[val_index]

        # print('\tFitting vectorizer...')
        # vectorizer.fit(validation_essays)
        vectorizer.fit(train_essays)

        # print('\tFitting model...')
        train_features = vectorizer.transform(train_essays)

        model.fit(train_features, train_labels)

        # print('\tMaking predictions...')
        validation_features = vectorizer.transform(validation_essays)
        predicted_labels = model.predict_proba(validation_features)[:, 1]

        # print('\tScoring...')
        auroc = roc_auc_score(validation_labels, predicted_labels)
        scores.append(auroc)

        print(f'\tFold {fold} - AUROC: {auroc:.4f}')
    
    # print('Average AUROC:', np.mean(scores))
    return np.mean(scores)

def Objective(trial):
    hparams = {
        'lr' : {
            'C' : trial.suggest_float('lr_C', 0.2, 1.5),
        },
        'xgb' : {
            'max_depth' : trial.suggest_int('xgb_max_depth', 3, 10),
            'n_estimators' : trial.suggest_int('xgb_n_estimators', 10, 1000, log = True),
            'learning_rate' : trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log = True),
            'subsample' : trial.suggest_float('xgb_subsample', 0.5, 1),
            'reg_alpha' : trial.suggest_float('xgb_reg_alpha', 0, 0.2),
            'reg_lambda' : trial.suggest_float('xgb_reg_lambda', 0, 0.2),
        },
        'weights' : [
            trial.suggest_float('weight_0', 0, 1),
            trial.suggest_float('weight_1', 0, 1),
        ]
    }

    return RunExperiment(hparams)

def DetermineOptimalHyperparameters():
    study = optuna.create_study(direction='maximize')
    study.optimize(Objective, n_trials=100)

    print("Best trial:", study.best_trial)
    print('Best AUROC:', study.best_trial.value)
    print('Best params:')
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Best AUROC: 0.9908222337647526
    # Best params:
    #     lr_C: 1.4530927509206837
    #     xgb_max_depth: 4
    #     xgb_n_estimators: 675
    #     xgb_learning_rate: 0.15598487328256594
    #     xgb_subsample: 0.7410024764460655
    #     xgb_reg_alpha: 0.1845353588224741
    #     xgb_reg_lambda: 0.07470974306084215
    #     weight_0: 0.7270227518540957
    #     weight_1: 0.5005508443771808

    # real	843m9.116s
    # user	10257m19.657s
    # sys	361m29.211s

def Debug():
    with open('data/Cache/Persuade_V0_BasicPreprocessing_30kGenerated.p', 'rb') as data_file:
        essays, labels = pickle.load(data_file)

    vectorizer = TfidfVectorizer(sublinear_tf=True)
    model = LogisticRegression(max_iter=1000)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_index, val_index) in enumerate(skf.split(essays, labels), 1):
        # print(f'Processing fold {fold}/5...')

        train_essays, validation_essays = np.array(essays)[train_index], np.array(essays)[val_index]
        train_labels, validation_labels = np.array(labels)[train_index], np.array(labels)[val_index]

        # print('\tFitting vectorizer...')
        vectorizer.fit(train_essays)

        # print('\tFitting model...')
        train_features = vectorizer.transform(train_essays)

        model.fit(train_features, train_labels)

        # print('\tMaking predictions...')
        validation_features = vectorizer.transform(validation_essays)
        predicted_labels = model.predict_proba(validation_features)[:, 1]

        # print('\tScoring...')
        auroc = roc_auc_score(validation_labels, predicted_labels)
        scores.append(auroc)

        print(f'\tFold {fold} - AUROC: {auroc:.4f}')

if __name__ == '__main__':
    Debug()

    # mean_auroc = RunExperiment({
    #     'lr' : {
    #         'C' : 1.4530927509206837,
    #     },
    #     'xgb' : {
    #         'max_depth' : 4,
    #         'n_estimators' : 675,
    #         'learning_rate' : 0.15598487328256594,
    #         'subsample' : 0.7410024764460655,
    #         'reg_alpha' : 0.1845353588224741,
    #         'reg_lambda' : 0.07470974306084215,
    #     },
    #     'weights' : [
    #         0.7270227518540957,
    #         0.5005508443771808,
    #     ]
    # })

    # print('Average score:', mean_auroc)