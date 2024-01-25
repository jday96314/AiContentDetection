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

# 2x3090s 
# cwd Shared/ExpansionDrive/AiContentDetection/
#   python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8000 --model Models/StudentImitators/single_epoch_Instruct-v0.2-start_fold_1/final_model_merged/ --gpu-memory-utilization 0.90 --max-model-len 2048
#   CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8001 --model Models/StudentImitators/single_epoch_fold_1/final_model_merged/ --gpu-memory-utilization 0.90 --max-model-len 2048
#
#   python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8000 --model Models/StudentImitators/1e-4_2_Fold1/final_model_merged/ --gpu-memory-utilization 0.90 --max-model-len 2048
#   CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8000 --model Models/StudentImitators/1e-4_2_Fold1/final_model_merged/ --gpu-memory-utilization 0.90 --max-model-len 2048
#   python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8000 --model Models/StudentImitators/Llama-2-13B-fp16_5e-5_1_Fold1/final_model_merged/ --gpu-memory-utilization 0.90 --max-model-len 2048 --tensor-parallel-size 2
#
#   nohup python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8000 --model Models/StudentImitators/mistral-ft-optimized-2e-4_1_Fold1/final_model_merged/ --gpu-memory-utilization 0.90 --max-model-len 2048 &> vllm_log1.txt &
#   nohup CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8001 --model Models/StudentImitators/mistral-ft-optimized-2e-4_1_Fold1/final_model_merged/ --gpu-memory-utilization 0.90 --max-model-len 2048 &> vllm_log2.txt &

def LoadAssignments(esssays_filepath):
    assignments = set()
    with open(esssays_filepath, 'r') as essays_file:
        csv_dict_reader = csv.DictReader(essays_file)

        for row in csv_dict_reader:
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
        'stop' : ['</s>']
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

def GenerateEssay(configuration):
    seed = (os.getpid() * int(time.time() * 100)) % 123456789
    np.random.seed(seed)
    random.seed(seed)

    # PICK ASSIGNMENT.
    assignments = LoadAssignments(configuration['HumanEssaysFilepath'])
    assignment_text, assignment_name = random.choice(assignments)
    
    # PICK SAMPLING CONFIG.
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

    # GENERATE ESSAY.
    template_name = configuration['PromptTemplateName']
    if template_name == 'Baseline':
        prompt = f'### Assignment: {assignment_text}\n### Essay: '
    elif template_name == 'Instruct_V1':
        prompt = f'<s>[INST] Write an essay with wording similar to a middle school or high school student for the assignment:\n{assignment_text}\n[/INST] '
    elif template_name == 'Instruct_V2':
        prompt = f'<s>[INST] Write an essay with wording similar to a middle school or high school student for the assignment:\n{assignment_text}\n[/INST]'
    else:
        assert False, f'Unknown template name: {template_name}'

    response = GeneratePromptResponse(
        host = configuration['LlmHostname'],
        port = configuration['VllmPort'],
        prompt = prompt,
        sampling_temperature = sampling_temperature,
        frequency_penalty = frequency_penalty,
        top_k = top_k,
        top_p = top_p,
        desired_response_count = 1,
        max_tokens = 1500
    )
    essay = response[0].strip()

    if len(response) == 0:
        # Try again. Sometimes there are random server errors.
        time.sleep(5)
        response = GeneratePromptResponse(
            host = configuration['LlmHostname'],
            port = configuration['VllmPort'],
            prompt = prompt,
            sampling_temperature = sampling_temperature,
            frequency_penalty = frequency_penalty,
            top_k = top_k,
            top_p = top_p,
            desired_response_count = 1,
            max_tokens = 1500
        )
        essay = response[0].strip()

    if len(essay) == 0:
        print(f'WARNING: Generated empty essay!')
        return

    # PICK OUTPUT FILEPATH.
    root_output_directory_path = configuration['RootOutputDirectoryPath']

    essay_hash = hashlib.md5()
    essay_hash.update(essay.encode('utf-8'))
    essay_hash_hex = essay_hash.hexdigest()
    output_filepath = f'{root_output_directory_path}/{essay_hash_hex}.json'

    # SAVE ESSAY.
    with open(output_filepath, 'w') as output_file:
        output_json = json.dumps({
            'Assigment' : assignment_text,
            'AssignmentName' : assignment_name,
            'SourceTextIncluded' : False,
            'ShotCount' : 0,
            'RequestPrefix' : None,
            'SamplingConfig': {
                'SamplingTemperature' : sampling_temperature,
                'TopK' : top_k,
                'TopP' : top_p,
                'FrequencyPenalty' : frequency_penalty,
            },
            'EssayText' : essay
        })

        output_file.write(output_json)

if __name__ == '__main__':
    # LOAD CONFIGURATION.
    config_filepath = sys.argv[1]
    with open(config_filepath, 'r') as file:
        configuration = json.load(file)
    
    # ENSURE OUTPUT DIRECTORY EXISTS.
    root_output_directory_path = configuration['RootOutputDirectoryPath']
    os.makedirs(root_output_directory_path, exist_ok=True)

    # CREATE WORK.
    generation_configs = []
    for _ in range(configuration['EssayCount']):
        generation_configs.append(configuration)

    # GENERATE ESSAYS.
    worker_count = configuration['ParallelRequestCount']
    with multiprocessing.Pool(worker_count) as worker_pool:
        worker_pool.map(GenerateEssay, generation_configs)