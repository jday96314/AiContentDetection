import hashlib
import glob
import requests
import json
import random
import numpy as np
import time
import os
from math import ceil
import multiprocessing
import sys

# python -m vllm.entrypoints.api_server --host 0.0.0.0 --model mistralai/Mistral-7B-Instruct-v0.1
# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8001 --model mistralai/Mistral-7B-v0.1
# python -m vllm.entrypoints.api_server --model Llama-2-13B-chat-AWQ/ --host 0.0.0.0 --quantization awq
# python -m vllm.entrypoints.api_server --host 0.0.0.0 --model TheBloke/Llama-2-13B-AWQ --gpu-memory-utilization 0.90 --quantization awq
# python -m vllm.entrypoints.api_server --model CodeLlama-34B-AWQ/ --host 0.0.0.0 --quantization awq --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.90
# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8001 --model TheBloke/CodeLlama-34B-AWQ --quantization awq --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.90
# python -m vllm.entrypoints.api_server --model WizardCoder-Python-34B-V1.0-AWQ/ --host 0.0.0.0 --quantization awq --dtype float16 --max-model-len 2560 --gpu-memory-utilization 0.90
# python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8001 --model TheBloke/CodeLlama-34B-AWQ --quantization awq --dtype float16 --max-model-len 3072 --gpu-memory-utilization 0.90 --tensor-parallel-size 2
# python -m vllm.entrypoints.api_server --host 0.0.0.0 --model OpenHermes-2.5-Mistral-7B/ --gpu-memory-utilization 0.90
# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 8001 --model HuggingFaceH4/zephyr-7b-beta
# python -m vllm.entrypoints.api_server --host 0.0.0.0 --model TheBloke/Airoboros-L2-13B-2.1-AWQ --gpu-memory-utilization 0.87 --quantization awq --dtype float16
# python -m vllm.entrypoints.api_server --host 0.0.0.0 --model TheBloke/Llama-2-70B-AWQ --gpu-memory-utilization 0.9 --quantization awq --dtype float16 --tensor-parallel-size 2

# https://huggingface.co/TheBloke/StableBeluga2-70B-AWQ
# https://huggingface.co/openchat/openchat_3.5
# https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b

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

def FormPrompt(prompt_format, context):
    if prompt_format == 'Foundation':
        return context
    
    payload_cuttoff = int(len(context) * np.random.uniform(0.1, 0.9))
    prompt_payload = context[:payload_cuttoff]

    user_request = np.random.choice([
        f'Complete the following document.\n\n"{prompt_payload}"',
        f'Finish writing the following document.\n\n"{prompt_payload}"',
        f'Help me finish writing the following.\n\n"{prompt_payload}"',
        f'Help me complete this: "{prompt_payload}"',
        f'Finish writing the following (be careful not to stop prematurely): "{prompt_payload}"',
    ])

    leading_words = np.random.choice([
        context[payload_cuttoff:],
        f"Here's a plausible continuation for the document: {context[payload_cuttoff:]}",
        f"Sure! Here's the rest of the document: \"{context[payload_cuttoff:]}"
    ])

    if prompt_format == 'MistralInstruct':
        return f'<s>[INST] {user_request} [/INST]\n{leading_words}'
    elif prompt_format == 'Llama2Chat':
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n {user_request} [/INST] {leading_words}"
    elif prompt_format == 'Alpaca':
        system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        return f"{system_prompt}\n\n### Instruction:\n{user_request}\n\n### Response: {leading_words}"
    elif prompt_format == 'OpenHermes':
        system_prompt = 'You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.'
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_request}<|im_end|>\n<|im_start|>assistant\n{leading_words}"
    elif prompt_format == 'ZephyrBeta':
        system_prompt = 'You are a helpful essay writing assistant who follows instructions very well. Use wording similar to a typical middle school or high school student.'
        return f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_request}</s>\n<|assistant|>\n{leading_words}"
    elif prompt_format == 'Airoboros':
        return f"A chat.\nUSER: {user_request}\nASSISTANT: {leading_words}" 
    elif prompt_format == 'Orca-Hashes':
        system_prompt = "You are a document completion assistant. Provide plausible continuations with a writing style similar to the provided sample text."
        return f"### System:\n{system_prompt}\n\n### User:\n{user_request}\n\n### Assistant:\n{leading_words}"
    elif prompt_format == 'Vicuna':
        system_prompt = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.'
        return f"{system_prompt} USER: {user_request} ASSISTANT: {leading_words}" 
    else:
        print(f'ERROR: Invalid prompt format ({prompt_format})!')

def GenerateCompletion(prompt, vllm_host, vllm_port):
    # PICK SAMPLING CONFIG.
    # sampling_temperature = np.random.uniform(low = 0, high = 2)
    sampling_temperature = np.clip(np.random.normal(loc = 1, scale = 0.2), a_min=0, a_max=2) # Centered around 1 because that's what's hardest for downstream classification models.
    frequency_penalty = np.random.uniform(low = 0, high = 0.5)
    top_k = int(np.random.choice([-1, 20, 40]))
    top_p = np.random.uniform(low = 0.5, high = 1)

    response = GeneratePromptResponse(
        host = vllm_host,
        port = vllm_port,
        prompt = prompt,
        sampling_temperature = sampling_temperature,
        frequency_penalty = frequency_penalty,
        top_k = top_k,
        top_p = top_p,
        desired_response_count = 1,
        max_tokens = 1300
    )

    if len(response) != 1:
        return None
    
    return {
        'GeneratedCompletion' : response[0],
        'SamplingConfig': {
            'SamplingTemperature' : sampling_temperature,
            'TopK' : top_k,
            'TopP' : top_p,
            'FrequencyPenalty' : frequency_penalty,
        },
    }

def GeneratePileCompletions(slice_path, prompt_format, blacklisted_pile_subsets, vllm_host, vllm_port, desired_completion_count, output_directory_path):
    # SEED RNG.
    seed = (os.getpid() * int(time.time() * 100)) % 123456789
    np.random.seed(seed)
    random.seed(seed)

    # ITERATE OVER SLICE.
    generated_completion_count = 0
    with open(slice_path) as pile_slice_file:
        for line_index, line in enumerate(pile_slice_file):
            # SKIP TO RANDOMLY SELECTED DOCUMENT.
            # Most of data is skipped so that we can take multiple passes over the slice
            # without a ton of overlap.
            # There are roughly 7 million documents per slice, so grabbing 0.1% works out to
            # roughly 7k completions per pass over a slice.
            SLECTION_PROB = 0.001
            if random.random() > SLECTION_PROB:
                continue

            document = json.loads(line)
            document_text = json.loads(line)['text']
            pile_subset = json.loads(line)['meta']['pile_set_name']

            # CHECK IF PILE SUBSET IS BLACKLISTED.
            if pile_subset in blacklisted_pile_subsets:
                continue
            
            # PARTITION DOCUMENT.
            context_len = int(len(document_text) * np.random.uniform(0.25, 0.75))
            context = document_text[:context_len]
            real_completion = document_text[context_len:]

            # GENERATE FAKE COMPLETION.
            prompt = FormPrompt(prompt_format, context)
            completion_and_metadata = GenerateCompletion(prompt, vllm_host, vllm_port)

            if (completion_and_metadata is None) or (len(completion_and_metadata['GeneratedCompletion']) == 0):
                print('WARNING: Failed to generate completion. Skipping to next prompt.')
                continue

            # SAVE RESULT.
            completion_and_metadata['RealCompletion'] = real_completion
            completion_and_metadata['ContextCharacterCount'] = context_len
            completion_and_metadata['PileSubset'] = pile_subset

            completion_hash = hashlib.md5()
            completion_hash.update(str(completion_and_metadata).encode('utf-8'))
            completion_hash_hex = completion_hash.hexdigest()
            output_filepath = f'{output_directory_path}/{completion_hash_hex}.json'

            with open(output_filepath, 'w') as output_file:
                output_file.write(json.dumps(completion_and_metadata))

            # POSSIBLY EXIT.
            generated_completion_count += 1
            if generated_completion_count >= desired_completion_count:
                break

if __name__ == '__main__':
    # LOAD CONFIGURATION.
    config_filepath = sys.argv[1]
    with open(config_filepath, 'r') as file:
        config = json.load(file)
    
    # ENSURE OUTPUT DIRECTORY EXISTS.
    output_directory_path = config['OutputDirectoryPath']
    os.makedirs(output_directory_path, exist_ok=True)

    # FORM GENERATION CONFIGS.
    pile_slices = glob.glob('/mnt/data01/LLMs/Data/ThePile/Uncompressed/*.jsonl')

    generation_configs = []
    for completion_id in range(config['WorkerCount']):
        generation_configs.append((
            pile_slices[completion_id % len(pile_slices)],
            config['PromptFormat'],
            config['BlacklistedPileSubsets'],
            config['VllmHostname'],
            config['VllmPort'],
            ceil(config['CompletionCount'] / config['WorkerCount']),
            config['OutputDirectoryPath']
        ))

    # GENERATE COMPLETIONS.
    with multiprocessing.Pool(config['WorkerCount']) as worker_pool:
        worker_pool.starmap(GeneratePileCompletions, generation_configs)