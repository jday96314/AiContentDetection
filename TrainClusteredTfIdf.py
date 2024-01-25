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
from sklearn.cluster import KMeans, SpectralClustering

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

if __name__ == '__main__':
    print('Loading data...')
    essays, labels = LoadData('data/Cache/Persuade_V0_BasicPreprocessing_30kGenerated.p')

    essays = np.random.choice(essays, 5000)
    labels = np.random.choice(labels, 5000)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_index, val_index) in enumerate(skf.split(essays, labels), 1):
        print(f'Processing fold {fold}/5...')

        train_essays, validation_essays = np.array(essays)[train_index], np.array(essays)[val_index]
        train_labels, validation_labels = np.array(labels)[train_index], np.array(labels)[val_index]

        print('\tFitting vectorizer...')
        # vectorizer = HashingVectorizer(ngram_range=(3, 7))
        vectorizer = TfidfVectorizer(ngram_range=(3, 7),sublinear_tf=True)
        vectorizer.fit(validation_essays)

        print('\tGrouping training examples...')
        train_features = vectorizer.transform(train_essays)

        CLUSTER_COUNT = 15
        clustering_algo = KMeans(n_clusters = CLUSTER_COUNT, n_init = 5)
        train_cluster_ids = clustering_algo.fit_predict(train_features)

        print('\tTraining models...')
        models_by_cluster = []
        for cluster_id in range(CLUSTER_COUNT):
            cluster_mask = (train_cluster_ids == cluster_id)
            cluster_train_features = train_features[cluster_mask]
            cluster_train_labels = train_labels[cluster_mask]

            model = LogisticRegression(n_jobs = -1, max_iter=1000)
            model.fit(cluster_train_features, cluster_train_labels)
            models_by_cluster.append(model)

        print('\tMaking predictions...')
        validation_features = vectorizer.transform(validation_essays)
        validation_cluster_ids = clustering_algo.predict(validation_features)
        all_clusters_predictions = []
        all_clusters_labels = []
        for cluster_id in range(CLUSTER_COUNT):
            cluster_mask = (validation_cluster_ids == cluster_id)
            cluster_validation_features = validation_features[cluster_mask]
            predicted_labels = model.predict_proba(cluster_validation_features)[:, 1]
            all_clusters_predictions.append(predicted_labels)

            cluster_validation_labels = validation_labels[cluster_mask]
            all_clusters_labels.append(cluster_validation_labels)
            
        all_clusters_predictions = np.concatenate(all_clusters_predictions)
        all_clusters_labels = np.concatenate(all_clusters_labels)

        print('\tScoring...')
        print(np.shape(all_clusters_labels), np.shape(all_clusters_predictions))
        auroc = roc_auc_score(all_clusters_labels, all_clusters_predictions)
        scores.append(auroc)

        print(f'\tFold {fold} - AUROC: {auroc:.4f}')
    
    print('Average AUROC:', np.mean(scores))
