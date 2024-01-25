import traceback
import os
import sys
import hashlib
import numpy as np
import json
import csv
from typing import List
from dataclasses import dataclass
import random
import requests
import multiprocessing
import time
import glob
import pickle
from transformers import AutoTokenizer
import re
import concurrent.futures
import threading
import onnxruntime as ort
import tokenmonster
from scipy.special import softmax
import multiprocessing

# 2x3090s
#   vllm
#   python -m vllm.entrypoints.api_server --host 0.0.0.0 --model ehartford/dolphin-2.1-mistral-7b --gpu-memory-utilization 0.90
#   CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8001 --model mistralai/Mistral-7B-Instruct-v0.1 --gpu-memory-utilization 0.90


# TODO: Load source_text & student essays
def LoadAssignments(esssays_filepath):
    assignments = set()
    with open(esssays_filepath, 'r') as essays_file:
        csv_dict_reader = csv.DictReader(essays_file)

        # assignments_to_sources = {}
        for row in csv_dict_reader:
            # assignments_to_sources[row['assignment']] = (row['prompt_name'], row['source_text'])
            assignments.add((row['assignment'], row['prompt_name']))

    return list(assignments)

def GeneratePromptResponse(host, port, prompt, sampling_temperature, frequency_penalty, top_k, top_p, desired_response_count, max_tokens):
    request_payload = {
        'prompt' : prompt,
        'temperature' : sampling_temperature,
        'frequency_penalty' : frequency_penalty,
        'top_k' : top_k if top_k is not None else -1,
        'top_p' : top_p if top_p is not None else 1,
        'max_tokens' : max_tokens,
        'n' : desired_response_count,
    }
    request_url = f"http://{host}:{port}/generate"
    response = requests.post(request_url, json = request_payload)

    if response.status_code != 200:
        print(f'ERROR: Failed to generate response to prompt (status code = {response.status_code})!')
        return [""]
    
    responses_text = []
    for prompt_and_response in json.loads(response.text)['text']:
        response_text = prompt_and_response[len(prompt):]
        responses_text.append(response_text)

    return responses_text

def GetExample(esssays_filepath, assignment_name):
    filtered_rows = []
    with open(esssays_filepath, 'r') as essays_file:
        csv_dict_reader = csv.DictReader(essays_file)

        for row in csv_dict_reader:
            if (assignment_name is not None) and (row['prompt_name'] != assignment_name):
                continue

            filtered_rows.append(row)

    selected_row = np.random.choice(filtered_rows)

    return selected_row['assignment'], selected_row['full_text']

def FormMistralInstructPrompt(system_prompt, messages):
    if system_prompt is None:
        STANDARDS_SYSTEM_PROMPTS = [
            # Empty string scores best on MT Bench.
            "",
            # Mistral's "system prompt for guardrailing." Taken from https://docs.mistral.ai/usage/guardrailing/
            "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. "
            # Workload-specific system prompt.
            "You are an AI writing assistant who is very good at following instructions. Write essays, stories, letters, and other content that appropriately completes user requests. ",
            # WizardLM system prompt.
            "Below is an instruction that describes a task. Write a response that appropriately completes the request. "
        ]
        system_prompt = random.choice(STANDARDS_SYSTEM_PROMPTS)
    
    prompt = f'<s> {system_prompt}'

    for message in messages:
        role, message_body = message['role'], message['content']
        if role == 'user':
            prompt += f' </s><s>[INST] {message_body} [/INST] '
        elif role == 'assistant':
            prompt += f'{message_body}'
        else:
            print(f'ERROR: Invalid role ({role}) encountered while forming prompt!')

    return prompt

def FormChatMLPrompt(system_prompt, messages):
    if system_prompt is None:
        STANDARDS_SYSTEM_PROMPTS = [
            "",
            "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. "
            "You are an AI writing assistant who is very good at following instructions. Write essays, stories, letters, and other content that appropriately completes user requests. ",
            "Below is an instruction that describes a task. Write a response that appropriately completes the request. "
        ]
        system_prompt = random.choice(STANDARDS_SYSTEM_PROMPTS)

    prompt = f'<|im_start|>system\n{system_prompt}'

    for message in messages:
        role, message_body = message['role'], message['content']
        if role == 'user':
            prompt += f'<|im_end|>\n<|im_start|>user\n{message_body}<|im_end|>\n<|im_start|>assistant\n'
        elif role == 'assistant':
            prompt += f'{message_body}'
        else:
            print(f'ERROR: Invalid role ({role}) encountered while forming prompt!')

    return prompt

def GetInitialConversationChain(configuration):
    # FORM SYSTEM PROMPT.
    assignments = LoadAssignments(configuration['HumanEssaysFilepath'])
    assignment_text, assignment_name = random.choice(assignments)
    
    source_text_filepath = os.path.join(configuration['SourceTextDirectoryPath'], assignment_name)
    source_text_exists = os.path.exists(source_text_filepath)
    include_source_text = source_text_exists and (random.random() < configuration['ProbIncludeSourceText'])
    if include_source_text:
        with open(source_text_filepath, 'r') as source_text_file:
            source_text = source_text_file.read()
        system_prompt = f'You are an AI writing assistant. If appropriate, use the following as added context when processing the user\'s requests.\n\n```{source_text}```\n'
    else:
        system_prompt = None

    # GET FEW-SHOT PROMPTING EXAMPLE.
    few_shot_examples = []
    example_count = int(np.random.choice(configuration['FewShotExampleCounts']))
    for _ in range(example_count):
        example_assignment_name = np.random.choice([assignment_name, None])
        example_assignment, example_essay = GetExample(configuration['HumanEssaysFilepath'], example_assignment_name)
        few_shot_examples += [{'role':'user', 'content':example_assignment}]
        few_shot_examples += [{'role':'assistant', 'content': example_essay}]
    
    # FORM CONVERSATION CHIAN.
    if random.random() < configuration['ProbIncludeSuppelementalInstructions']:
        instructions = np.random.choice(configuration['SuppelementalInstructions'])
        prompt_content = f'{instructions} "{assignment_text}"'
    else:
        instructions = None
        prompt_content = assignment_text

    prompt_metadata = {
        'Assigment' : assignment_text,
        'AssignmentName' : assignment_name,
        'SourceTextIncluded' : include_source_text,
        'ShotCount' : example_count,
        'RequestPrefix' : instructions,
    }

    messages = few_shot_examples + [{'role':'user', 'content':prompt_content}]

    return system_prompt, messages, prompt_metadata

def PickSamplingSettings(configuration):
    sampling_config = configuration['SamplingConfigurations']
    sampling_temperature = np.random.uniform(
        low = sampling_config['MinSamplingTemperature'], 
        high = sampling_config['MaxSamplingTemperature'])
    frequency_penalty = np.random.uniform(
        low = sampling_config['MinFrequencyPenalty'], 
        high = sampling_config['MaxFrequencyPenalty'])
    top_k = int(np.random.choice(sampling_config['TopKValues']))
    top_p = np.random.uniform(
        low = sampling_config['MinTopPValue'], 
        high = sampling_config['MaxTopPValue'])
    
    return sampling_temperature, frequency_penalty, top_k, top_p

def FormPrompt(template_name, system_prompt, messages):
    if template_name == 'MistralInstruct':
        prompt = FormMistralInstructPrompt(system_prompt,  messages)
    elif template_name == 'ChatML':
        prompt = FormChatMLPrompt(system_prompt,  messages)
    else:
        print(f"ERROR: Invalid prompt template ({template_name})!")
        assert False

    return prompt

def CountTokens(tokenizer, text):
    # 1-2 extra tokens from string start token + some unknown factor, maybe vLLM not stopping generation at a very precise time.
    return len(tokenizer(text)['input_ids']) - 1

def GetLeadingWords(configuration, assignment_name):
    _, example_human_essay = GetExample(configuration['HumanEssaysFilepath'], assignment_name)

    return example_human_essay[:np.random.randint(30, 100)]
    
class Conv1DPipeline:
    def __init__(self, model_path, vocab_size, max_sequence_length):
        self.InferenceSession = ort.InferenceSession(model_path)
        self.VocabSize = vocab_size
        self.Tokenizer = tokenmonster.load_multiprocess_safe(f'Tokenizers/tokenmonster/vocabs/english-{vocab_size}-strict-v1.vocab')
        self.MaxSeqLen = max_sequence_length

    def predict_proba(self, essays):
        tokenized_essays = self.Tokenizer.tokenize(essays)
        padded_essays = np.ones((len(essays), self.MaxSeqLen), dtype=np.int32) * self.VocabSize
        for essay_index, tokenized_essay in enumerate(tokenized_essays):
            usable_token_count = min(self.MaxSeqLen, len(tokenized_essay))
            padded_essays[essay_index][:usable_token_count] = tokenized_essay[:usable_token_count]

        raw_predictions = self.InferenceSession.run(None, {'token_ids': padded_essays})[0]

        predictions = softmax(raw_predictions, axis = 1)
        return predictions

def GenerateEssay(configuration, min_error_threshold, victim_classifier_name, victim_classifier):
    seed = (threading.get_ident() * int(time.time() * 100)) % 123456789
    np.random.seed(seed)
    random.seed(seed)

    system_prompt, initial_messages, prompt_metadata = GetInitialConversationChain(configuration)
    sampling_temperature, frequency_penalty, top_k, top_p = PickSamplingSettings(configuration)

    # print('DEBUG:', (sampling_temperature, frequency_penalty, top_k, top_p))
    # print('DEBUG 2:', initial_messages)

    tokenizer = AutoTokenizer.from_pretrained(configuration['Tokenizer'])

    # GENERATE ESSAY.
    essay = GetLeadingWords(configuration, prompt_metadata['AssignmentName'])
    consecutive_failure_count = 0
    total_falure_count = 0
    failures_since_last_backtrack = 0
    MAX_FAILURE_BEFORE_BACKTRACKING = 10
    MAX_FAILURES_BEFORE_EXITING = 20 * 16
    TOKENS_PER_EXTENSION = 30
    MAX_TOKENS_PER_ESSAY = 1500
    while True:
        # ENSURE WE DON'T GET STUCK IN AN INFINITE LOOP.
        if total_falure_count > MAX_FAILURES_BEFORE_EXITING:
            print('WARNING: Failed to conduct attack!')
            return None
        
        if (consecutive_failure_count > MAX_FAILURE_BEFORE_BACKTRACKING) and (failures_since_last_backtrack > MAX_FAILURE_BEFORE_BACKTRACKING//2):
            # Backtrack slightly & regenerate the tail end of the essay.
            backtrack_amount = 5 + (total_falure_count // 10)
            essay = essay[:-backtrack_amount]

            failures_since_last_backtrack = 0

            print('Backtracking...')
            print('Essay:', essay)

        # FORM PROMPT.
        prompt = FormPrompt(
            template_name=configuration['PromptTemplateName'],
            system_prompt=system_prompt,
            messages=initial_messages + [{'role':'assistant', 'content': essay}])

        # GENERATE CANDIDATE CONTINUATIONS.
        candidate_continuations = GeneratePromptResponse(
            host = configuration['LlmHostname'],
            port = configuration['VllmPort'],
            prompt = prompt,
            sampling_temperature = sampling_temperature,
            frequency_penalty = frequency_penalty,
            top_k = top_k,
            top_p = top_p,
            desired_response_count = 1,
            max_tokens = TOKENS_PER_EXTENSION)
        
        # SELECT CONTINUATION THAT FOOLS THE CLASSIFIER.
        updated_essays = [essay + continuation for continuation in candidate_continuations]
        p_generated_scores = victim_classifier.predict_proba(updated_essays)[:, 1]

        print(p_generated_scores)

        selected_continuation = None
        for continuation_index, p_generated_score in enumerate(p_generated_scores):
            if p_generated_score < (1 - min_error_threshold):
                selected_continuation = candidate_continuations[continuation_index]
                essay = updated_essays[continuation_index]
                break

        # UPDATE FAILURE COUNT.
        if selected_continuation is None:
            consecutive_failure_count += 1
            total_falure_count += 1
            failures_since_last_backtrack += 1
            continue

        consecutive_failure_count = 0
        failures_since_last_backtrack = 0

        # POSSIBLY END GENERATION.
        # Only considered done if the essay is significantly shorter than the desired length because
        # vLLM doesn't seem to stop at a precise time.
        extension_length = CountTokens(tokenizer, selected_continuation)
        continuation_cut_short = extension_length < TOKENS_PER_EXTENSION
        
        ending_is_valid = re.search(r'[\.!?]"?$', essay) or ('sincerely' in essay.lower()[3*len(essay)//4:])
        if continuation_cut_short and ending_is_valid:
            break

        essay_length = CountTokens(tokenizer, essay)
        essay_hit_length_limit = (essay_length > MAX_TOKENS_PER_ESSAY)
        if essay_hit_length_limit:
            print('WARNING: Essay hit length limit. This should be rare!')
            break

        print(essay_length)

    print('Yay (1)!', essay_length)

    # PICK OUTPUT FILEPATH.
    root_output_directory_path = configuration['RootOutputDirectoryPath']
    output_directory_path = os.path.join(root_output_directory_path, victim_classifier_name)

    os.makedirs(output_directory_path, exist_ok=True)

    essay_hash = hashlib.md5()
    essay_hash.update(essay.encode('utf-8'))
    essay_hash_hex = essay_hash.hexdigest()
    output_filepath = os.path.join(output_directory_path, f'{essay_hash_hex}.json')

    # SAVE ESSAY.
    with open(output_filepath, 'w') as output_file:
        output_json = json.dumps({
            'PromptDetails' : prompt_metadata,
            'SamplingConfig': {
                'SamplingTemperature' : sampling_temperature,
                'TopK' : top_k,
                'TopP' : top_p,
                'FrequencyPenalty' : frequency_penalty,
            },
            'MinErrorThreshold' : min_error_threshold,
            'EssayText' : essay
        })

        output_file.write(output_json)

    print('Yay (2)!', essay_length)

def DebuGenerateEssay(configuration, min_error_threshold, victim_classifier_name, victim_classifier):
    try:
        GenerateEssay(configuration, min_error_threshold, victim_classifier_name, victim_classifier)
    except Exception as e:
        traceback.print_exc()

def GenerateEssays(configuration):
    # LOAD VICTIM CLASSIFIERS.
    victim_classifiers = []
    # for classifier_path in glob.glob('Models/VictimClassifiers/*.p'):
    #     with open(classifier_path, 'rb') as classifier_file:
    #         classifier = pickle.load(classifier_file)
    for classifier_path in glob.glob('Models/VictimClassifiers/*.onnx'):
        classifier = Conv1DPipeline(
            model_path = classifier_path,
            vocab_size = 24000,
            max_sequence_length = 1024)

        victim_classifier_name, _ = os.path.splitext(os.path.basename(classifier_path))
        victim_classifiers.append((victim_classifier_name, classifier))

    # GENERATE ESSAYS.
    thread_count = configuration['ThreadCount']
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        generation_tasks = []

        for _ in range(configuration['EssayCount']):
            min_error_threshold = random.choice([0.125, 0.25, 0.5, 0.75])
            victim_classifier_name, victim_classifier = random.choice(victim_classifiers)

            task = executor.submit(
                GenerateEssay,
                configuration,
                min_error_threshold,
                victim_classifier_name,
                victim_classifier
            )
            generation_tasks.append(task)

        concurrent.futures.wait(generation_tasks)

if __name__ == '__main__':
    # LOAD CONFIGURATION.
    config_filepath = sys.argv[1]
    with open(config_filepath, 'r') as file:
        configuration = json.load(file)

    worker_configs = []
    for _ in range(configuration['ThreadCount']):
        worker_config = configuration.copy()
        worker_config['EssayCount'] = configuration['EssayCount'] // configuration['ThreadCount']
        worker_config['ThreadCount'] = 1
        worker_configs.append(worker_config)

    # GENERATE ESSAYS.
    with multiprocessing.Pool(processes=configuration['ThreadCount']) as pool:
        pool.map(GenerateEssays, worker_configs)