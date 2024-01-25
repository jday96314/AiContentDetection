import pickle

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

from MultiDataset import PileCompletionsDataset, PersuadeDataset, MultiDataset

def LoadTfIdfData(use_preprocessed_data, use_d7_only):
    with open('data/Cache/Persuade_V0_BasicPreprocessing_50kGenerated_MetadataIncluded.p', 'rb') as train_data_file:
        preprocessed_essays_text, essays_metadata, essay_labels = pickle.load(train_data_file)

    if use_d7_only:
        filtered_essays_text = []
        filtered_essay_labels = []
        for essay_index, label in enumerate(essay_labels):
            # GET ASSIGNMENT NAME.
            essay_metadata = essays_metadata[essay_index]

            if 'prompt_name' in essay_metadata.keys():
                assignment_name = essay_metadata['prompt_name']
            else:
                assignment_name = essay_metadata['AssignmentName']

            # EXCLUDE IRRELEVANT ASSIGNMENTS.
            if assignment_name not in PersuadeDataset.COMPETITION_ASSIGNMENTS:
                continue

            # RECORD ESSAY & LABEL.
            filtered_essay_labels.append(label)

            if use_preprocessed_data:
                filtered_essays_text.append(preprocessed_essays_text[essay_index])
            else:
                raw_essay_text = essay_metadata['EssayText'] if 'EssayText' in essay_metadata.keys() else essay_metadata['full_text']
                filtered_essays_text.append(raw_essay_text)

        return filtered_essays_text, filtered_essay_labels  
    else:
        if use_preprocessed_data:
            return preprocessed_essays_text, essay_labels
        else:
            raw_essay_text = [
                essay_data['EssayText'] if 'EssayText' in essay_data.keys() else essay_data['full_text']
                for essay_data in essays_metadata
            ]
            return raw_essay_text, essay_labels

def TruncateEssays(essays_text, min_desired_length_characters):
    truncated_essays = []
    for essay in essays_text:
        lower_bound_length = min(min_desired_length_characters, len(essay))
        upper_bound_length = len(essay)
        truncation_length = int(np.random.uniform(lower_bound_length, upper_bound_length))

        truncated_essays.append(essay[:truncation_length])

    return truncated_essays

def TrainTfIdfModels(root_output_directory_path):
    classifiers = [
        (
            'words_logistic-regression', 
            Pipeline(steps = [
                ('tf-idf', TfidfVectorizer(sublinear_tf=True)),
                ('logistic-regression', LogisticRegression(max_iter=1000))])
        ),
        (
            'ngram15_naive-bayes', 
            Pipeline(steps = [
                ('tf-idf', TfidfVectorizer(ngram_range=(1, 5))),
                ('multinomial-naive-bayes', MultinomialNB())])
        ),
        (
            'words_xgboost', 
            Pipeline(steps = [
                ('tf-idf', TfidfVectorizer()),
                ('xgboost-classifier', XGBClassifier())])
        )
    ]

    preprocessing_configs = [True, False]
    d7_filtering_configs = [True, False]
    combinations = itertools.product(preprocessing_configs, d7_filtering_configs, classifiers)
    for preprocessing_enabled, d7_filtering_enabled, classification_approach in combinations:
        # LOG ONGOING WORK.
        classification_approach_name, classifier = classification_approach
        print('Preprocessing enabled = {}, d7 filtering enabled = {}, Classifier = "{}"'.format(
            preprocessing_enabled, d7_filtering_enabled, classification_approach_name))
        
        # LOAD DATA.
        essays_text, labels = LoadTfIdfData(preprocessing_enabled, d7_filtering_enabled)
        essays_text = TruncateEssays(essays_text, min_desired_length_characters=50)

        # TRAIN CLASSIFIER.
        classifier.fit(essays_text, labels)

        # VALIDATE CLASSIFIER.
        # This is just intented to check for UNDERfitting, so we test on the training set.
        predictions = classifier.predict_proba(essays_text)[:,1]
        auroc = roc_auc_score(labels, predictions)
        print('\tAUROC:', auroc)

        # SAVE PIPELINE.
        output_filename = f'tf-idf_{classification_approach_name}_preprocessing={preprocessing_enabled}_filtering={d7_filtering_enabled}.p'
        output_file_path = os.path.join(root_output_directory_path, output_filename)
        with open(output_file_path, 'wb') as output_file:
            pickle.dump(classifier, output_file)

if __name__ == '__main__':
    TrainTfIdfModels('Models/VictimClassifiers')