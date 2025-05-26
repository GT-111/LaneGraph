import os
import easydict as edict
import json
import yaml
import re

def get_files_with_extension(folder_path = '', file_extension = '.parquet'):

    files = []
    for root, _, file in os.walk(folder_path):
        for f in file:
            if f.endswith(file_extension):
                files.append(os.path.join(root, f))

    return files

def get_last_file_with_extension(folder_path = '', file_extension = '.pth'):
    files = []
    
    for root, _, file in os.walk(folder_path):
        for f in file:
            if f.endswith(file_extension):
                files.append(os.path.join(root, f))
    files.sort(key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)) if re.search(r'epoch_(\d+)', x) else -1)
    return files[-1] if len(files) > 0 else None



def get_config (config_file = ''):
    # Load config file
    # if config is end with .json
    if config_file.endswith('.json'):
        with open(config_file) as f:
            config = edict.EasyDict(json.load(f))
    elif config_file.endswith('.yaml'):
        # if config is end with .yaml
        with open(config_file) as f:
            config = edict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))
            
    return config






