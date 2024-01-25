import shutil
import json
import glob
import os
from tqdm import tqdm
from multiprocessing import Pool

def FilterV1Directory(input_directory_path, output_directory_path, blacklisted_subsets):
    # ENSURE OUTPUT DIRECTORY EXIST.
    os.makedirs(output_directory_path, exist_ok=True)

    # LOAD DATA TO FILTER.
    filtered_documents = []
    for input_filepath in glob.glob(f'{input_directory_path}/*.json'):
        output_filepath = input_filepath.replace(input_directory_path, output_directory_path)
        output_directory_path = os.path.dirname(output_filepath)
        
        with open(input_filepath, 'r') as input_file:
            # Keys: ['GeneratedCompletion', 'SamplingConfig', 'RealCompletion', 'ContextCharacterCount']
            input_data = json.load(input_file)
        
        filtered_documents.append(input_data)
    
    # FILTER DATA.
    pile_slice_filepaths = glob.glob('/mnt/data01/LLMs/Data/ThePile/Uncompressed/*.jsonl')
    for pile_slice_filepath in tqdm(pile_slice_filepaths):
        with open(pile_slice_filepath) as pile_slice_file:
            for line_index, line in enumerate(pile_slice_file):
                # CHECK IF THE DOCUMENT IS IN A BLACKLISTED SUBSET.
                pile_record = json.loads(line)
                subset_name = pile_record['meta']['pile_set_name']
                if subset_name not in blacklisted_subsets:
                    continue

                # ELIMINATE ANYTHING MATCHING THE COPYRIGHTED TEXT.
                possibly_copyrighted_text = pile_record['text']
                filtered_documents = [
                    document
                    for document in filtered_documents
                    if not possibly_copyrighted_text.endswith(document['RealCompletion'])
                ]

                # if line_index > 10000:
                #     break

    # SAVE FILTERED DATA.
    for filtered_document_index, filtered_document in enumerate(filtered_documents):
        output_filepath = f'{output_directory_path}/{filtered_document_index}.json'
        with open(output_filepath, 'w') as output_file:
            json.dump(filtered_document, output_file)

def FilterV2Directory(input_directory_path, output_directory_path, blacklisted_subsets):
    # ENSURE OUTPUT DIRECTORY EXIST.
    os.makedirs(output_directory_path, exist_ok=True)

    # LOAD DATA TO FILTER.
    filtered_documents = []
    for input_filepath in glob.glob(f'{input_directory_path}/*.json'):
        # READ INPUT.
        with open(input_filepath, 'r') as input_file:
            # Keys: ['GeneratedCompletion', 'SamplingConfig', 'RealCompletion', 'ContextCharacterCount']
            input_data = json.load(input_file)

        # CHECK IF THE DOCUMENT IS FROM A BLACKLISTED SUBSET.
        if input_data['PileSubset'] in blacklisted_subsets:
            continue

        # COPY TO OUTPUT.
        output_filepath = input_filepath.replace(input_directory_path, output_directory_path)       
        shutil.copy(input_filepath, output_filepath)

if __name__ == '__main__':
    ROOT_INPUT_DIRECTORY_PATH = 'data/PileCompletions_V2'
    input_directory_paths = sorted(glob.glob(f'{ROOT_INPUT_DIRECTORY_PATH}/*'))

    BLACKLISTED_SUBSETS = ['Books3', 'BookCorpus2', 'OpenSubtitles', 'YoutubeSubtitles', 'OpenWebText2']
    ROOT_OUTPUT_DIRECTORY_PATH = 'data/PileCompletions_V3'

    with Pool(16) as worker_pool:
        worker_pool.starmap(
            # FilterV1Directory, 
            FilterV2Directory, 
            [
                (
                    input_directory_path, 
                    input_directory_path.replace(ROOT_INPUT_DIRECTORY_PATH, ROOT_OUTPUT_DIRECTORY_PATH), 
                    BLACKLISTED_SUBSETS
                ) 
                for input_directory_path in input_directory_paths
            ]
        )