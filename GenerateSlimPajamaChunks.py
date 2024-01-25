from datasets import load_dataset
from tqdm import tqdm
import os
import json

# CREATE STREAMING DATASET.
# Slim pajama's test dataset is used because it is plenty large & faster to initialize.
CHUNK_SIZE = 25000
CHUNK_COUNT = 20
dataset = load_dataset("cerebras/SlimPajama-627B", cache_dir='/mnt/data02/datasets/SlimPajama', streaming=True, split='test')

# GENERATE CHUNKS.
chunk_documents = []
for document_index, document in tqdm(enumerate(dataset)):
    chunk_documents.append(document)
    
    # CHECK IF CHUNK IS COMPLETE.
    if len(chunk_documents) >= CHUNK_SIZE:
        # SAVE CHUNK.
        chunk_filepath = f'data/SlimPajamaCompletions/SourceDocuments/{document_index // CHUNK_SIZE}.json'
        chunk_directory_path = os.path.dirname(chunk_filepath)
        os.makedirs(chunk_directory_path, exist_ok=True)
        
        with open(chunk_filepath, 'w') as chunk_file:
            json.dump(chunk_documents, chunk_file)
        
        # RESET CHUNK.
        chunk_documents = []

    # CHECK IF ALL CHUNKS ARE COMPLETE.
    document_count = CHUNK_SIZE * CHUNK_COUNT
    if document_index >= document_count:
        break
