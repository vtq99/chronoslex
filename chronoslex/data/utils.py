import os
import pandas as pd
import re

# import gdown
import torch
from transformers import AutoTokenizer, GPT2Tokenizer
from datasets import load_dataset


def initialize_gpt2_transform(max_token_length):
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    def transform(text):
        # tokens = tokenizer(
        #     text,
        #     padding='max_length',
        #     truncation=True,
        #     max_length=max_token_length,
        #     return_tensors='pt'
        # )
        tokens = tokenizer(
            ' '.join(text),
            padding='max_length',
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        )
        x = torch.stack((tokens['input_ids'], tokens['attention_mask']), dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform

    
# def download_gdrive(url, save_path, is_folder):
#     """ Download the preprocessed data from Google Drive. """
#     if not is_folder:
#         gdown.download(url=url, output=save_path, quiet=False)
#     else:
#         gdown.download_folder(url=url, output=save_path, quiet=False)


def download_eurlex(data_dir, dataset_file):
    # Load the English part of the dataset
    if 'eurlex21' in dataset_file:
        dataset = load_dataset('multi_eurlex', language='en', label_level='level_1')
    else:
        dataset = load_dataset('multi_eurlex', language='en', label_level='level_2')
    splits = ['train', 'validation', 'test']
    df = pd.DataFrame(columns=['celex_id', 'text', 'labels', 'year'])

    for x in splits:
        for i in dataset[x]:
            valid_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                            "October", "November", "December"]
            pattern = r'(\d{1,2})\s*(' + '|'.join(valid_months) + ')\s*(\d{4})'
            # matches = list(re.finditer(pattern, i['title'], re.IGNORECASE))

            # if len(matches) > 0:
            #     year = matches[0].group(3)
            # else:
            #
            matches = list(re.finditer(pattern, i['text'], re.IGNORECASE))
            if len(matches) > 0:
                year = matches[0].group(3)
            # else:
                # print(i['celex_id'], i['text'][:100], '\n')
            elif i['celex_id'] == '31988R0091':
                year = 1988
            elif i['celex_id'] in ['31987D0594', '31987D0593']:
                year = 1987

            if i['celex_id'] == '32014R0607':
                year = 2014
            if i['celex_id'] == '32013R0589':
                year = 2013

            df.loc[len(df)] = list(i.values()) + [int(year)]
    df = df.sort_values(by=['year'])
    if 'eurlex21' in dataset_file:
        df.to_json(os.path.join(data_dir, 'eurlex21.jsonl'), orient='records', lines=True)
    else:
        df.to_json(os.path.join(data_dir, 'eurlex127.jsonl'), orient='records', lines=True)


def download_detection(data_dir, dataset_file):
    if os.path.isfile(data_dir):
        raise RuntimeError('Save path should be a directory!')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, dataset_file)):
        pass
    else:
        if 'eurlex' in dataset_file:
            download_eurlex(data_dir, dataset_file)
        # elif 'drug' in dataset_file:
        #     download_drug(data_dir)
        # elif 'fmow' in dataset_file:
        #     download_fmow(data_dir)
        # elif 'huffpost' in dataset_file:
        #     download_huffpost(data_dir)
        # elif 'yearbook' in dataset_file:
        #     download_yearbook(data_dir)
        # else:
        #     raise RuntimeError(f'The dataset {dataset_file} does not exist in WildTime!')
