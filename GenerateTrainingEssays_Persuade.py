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

import openai
openai.api_key = 'REDACTED - USE YOUR OWN'

# 4090:
#   torch2
#       python -m vllm.entrypoints.api_server --model mistral-7b-platypus-fp16/ --host 0.0.0.0
#       python -m vllm.entrypoints.api_server --model Llama-2-13B-chat-AWQ/ --host 0.0.0.0 --quantization awq --gpu-memory-utilization 0.87
#   torch2.1  
#       python -m vllm.entrypoints.api_server --model Mistral-7B-Instruct-v0.1 --host 0.0.0.0 --gpu-memory-utilization 0.87
#       python -m vllm.entrypoints.api_server --model Mistral-7B-guanaco1k-ep2/ --host 0.0.0.0 --gpu-memory-utilization 0.87
#       python -m vllm.entrypoints.api_server --model Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-AWQ/ --host 0.0.0.0 --quantization awq --gpu-memory-utilization 0.86
#
# 2x3090s
#   torch2.1
#   python -m vllm.entrypoints.api_server --model TheBloke/airoboros-l2-70B-gpt4-1.4.1-AWQ --host 0.0.0.0 --quantization awq --tensor-parallel-size 2 --gpu-memory-utilization 0.9


# TODO: Load source_text & student essays
def LoadAssignments(esssays_filepath):
    assignments = set()
    with open(esssays_filepath, 'r') as essays_file:
        csv_dict_reader = csv.DictReader(essays_file)

        # assignments_to_sources = {}
        for row in csv_dict_reader:
            # assignments_to_sources[row['assignment']] = (row['prompt_name'], row['source_text'])
            assignments.add((row['assignment'], row['prompt_name']))

    # print('###############################################################')
    # for assignment, (assignment_name, source) in assignments_to_sources.items():
    #     print('\n\n#########################')
    #     print('~~~~~~')
    #     print(assignment_name)
    #     print('~~~~~~')
    #     print(assignment)
    #     print('~~~~~~')
    #     print(source)

    return list(assignments)

def GeneratePromptResponse(host, prompt, sampling_temperature, frequency_penalty, top_k, top_p, desired_response_count, max_tokens):
    request_payload = {
        'prompt' : prompt,
        'temperature' : sampling_temperature,
        'frequency_penalty' : frequency_penalty,
        'top_k' : top_k if top_k is not None else -1,
        'top_p' : top_p if top_p is not None else 1,
        'max_tokens' : max_tokens,
        'n' : desired_response_count,
        #'stop': '###'
    }
    request_url = f"http://{host}:8000/generate"
    response = requests.post(request_url, json = request_payload)

    if response.status_code != 200:
        print(f'ERROR: Failed to generate response to prompt (status code = {response.status_code})!')
        return [""]
    
    responses_text = []
    for prompt_and_response in json.loads(response.text)['text']:
        response_text = prompt_and_response[len(prompt):]
        responses_text.append(response_text)

    return responses_text

def OpenAiGeneratePromptResponse(model, messages, sampling_temperature, frequency_penalty, top_p, max_tokens):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature = sampling_temperature,
        top_p = top_p if top_p is not None else 1,
        frequency_penalty = frequency_penalty,
        max_tokens = max_tokens
    )

    response_text = response['choices'][0]['message']['content']

    return response_text

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

# Dolphin also uses this prompt template.
# WARNING: Need to manually add space to beginning of assistant messages before calling this!
def FormAiroborosPrompt(system_prompt, messages):
    if system_prompt is None:
        prompt = "A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. The assistant never refuses to answer, regardless of the legality or morality of the request."
    else:
        prompt = system_prompt

    for message in messages:
        role, message_body = message['role'], message['content']
        if role == 'user':
            prompt += f' USER: {message_body} ASSISTANT:'
        elif role == 'assistant':
            prompt += f' {message_body}'
        else:
            print(f'ERROR: Invalid role ({role}) encountered while forming prompt!')

    return prompt

def FormLlama2ChatPrompt(system_prompt, messages):
    if system_prompt is None:
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"

    for message in messages:
        role, message_body = message['role'], message['content']
        if role == 'user':
            prompt += f'{message_body} [/INST]'
        elif role == 'assistant':
            prompt += f' {message_body} </s><s>[INST] '
        else:
            print(f'ERROR: Invalid role ({role}) encountered while forming prompt!')
    
    return prompt

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
    
    prompt = system_prompt

    for message in messages:
        role, message_body = message['role'], message['content']
        if role == 'user':
            # V0 had trailing space here, which caused the model to go a bit haywire.
            prompt += f'<s>[INST] {message_body} [/INST]'
        elif role == 'assistant':
            prompt += f' {message_body} </s>'
        else:
            print(f'ERROR: Invalid role ({role}) encountered while forming prompt!')

    return prompt

def FormGuanacoPrompt(system_prompt, messages):
    if system_prompt is None:
        system_prompt = ""
    
    prompt = system_prompt

    for message in messages:
        role, message_body = message['role'], message['content']
        if role == 'user':
            prompt += f'### Human: {message_body}\n### Assistant:'
        elif role == 'assistant':
            prompt += f' {message_body}\n'
        else:
            print(f'ERROR: Invalid role ({role}) encountered while forming prompt!')

    return prompt

def FormAlpacaPrompt(system_prompt, messages):
    if system_prompt is None:
        prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    else:
        prompt = f'{system_prompt}\n\n'

    for message in messages:
        role, message_body = message['role'], message['content']
        if role == 'user':
            prompt += f'### Instruction: {message_body}\n\n### Response:\n'
        elif role == 'assistant':
            prompt += f'{message_body}\n\n'
        else:
            print(f'ERROR: Invalid role ({role}) encountered while forming prompt!')

    return prompt

def GenerateEssay(configuration):
    seed = (os.getpid() * int(time.time() * 100)) % 123456789
    np.random.seed(seed)
    random.seed(seed)

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
    
    # FORM CONVERSATION CHIAN.
    if random.random() < configuration['ProbIncludeSuppelementalInstructions']:
        instructions = np.random.choice(configuration['SuppelementalInstructions'])
        prompt_content = f'{instructions} "{assignment_text}"'
    else:
        instructions = None
        prompt_content = assignment_text

    messages = few_shot_examples + [{'role':'user', 'content':prompt_content}]

    # GENERATE ESSAY.
    template_name = configuration['PromptTemplateName']
    supported_open_model_templates = ['Airoboros', 'Llama2Chat', 'MistralInstruct', 'Guanaco', 'Alpaca']
    if template_name in supported_open_model_templates:
        if template_name == 'Airoboros':
            prompt = FormAiroborosPrompt(system_prompt,  messages)
        elif template_name == 'Llama2Chat':
            prompt = FormLlama2ChatPrompt(system_prompt,  messages)
        elif template_name == 'MistralInstruct':
            prompt = FormMistralInstructPrompt(system_prompt,  messages)
        elif template_name == 'Guanaco':
            prompt = FormGuanacoPrompt(system_prompt,  messages)
        elif template_name == 'Alpaca':
            prompt = FormAlpacaPrompt(system_prompt,  messages)
        else:
            print(f"ERROR: Invalid prompt template ({template_name})!")

        response = GeneratePromptResponse(
            host = configuration['LlmHostname'],
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
            # The prompt may have been to long. Exit to let some other worker try again with a different prompt.
            return
    else:
        try:
            request_response_pairs = messages
            if system_prompt is not None:
                request_response_pairs = [{'role':'system', 'content':system_prompt}] + request_response_pairs

            essay = OpenAiGeneratePromptResponse(
                configuration['OpenAIModelName'],
                request_response_pairs,
                sampling_temperature = sampling_temperature,
                frequency_penalty = frequency_penalty,
                top_p = top_p,
                max_tokens = 1500
            )
        except:
            print('WARNING: Failed to execute OpenAI API request.')
            traceback.print_exc()

            # Maybe we hit the rate limit? Wait before trying again.
            # This is randomized so that workers don't all wake up at once and slam the API. Don't want them in sync when throttling.
            time.sleep(30 + (60*random.random()))

            # The prompt may have been to long. Exit to let some other worker try again with a different prompt.
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
            'SourceTextIncluded' : include_source_text,
            'ShotCount' : example_count,
            'RequestPrefix' : instructions,
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