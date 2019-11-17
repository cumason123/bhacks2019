import os
from shutil import copyfile

import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    np.random.seed(0)

    valid_split = 0.1
    test_split = 0.1
    data_folder = 'data'
    destination_folder = 'balanced_data'
    image_folder = os.path.join(data_folder, 'images')

    metadata_file = os.path.join(data_folder, 'HAM10000_metadata.csv')

    metadata = pd.read_csv(metadata_file)

    classes = metadata['dx'].unique()

    metadata['split'] = 'nothing'

    for class_name in classes:
        locations = metadata.loc[metadata['dx'] == class_name, 'split']
        count = len(locations)

        num_val = int(count * valid_split)
        num_test = int(count * test_split)
        num_train = count - num_val - num_test

        splits = ['train'] * num_train + \
                 ['validation'] * num_val + \
                 ['test'] * num_test

        np.random.shuffle(splits)

        metadata.loc[metadata['dx'] == class_name, 'split'] = splits

    for split_folder in ['train', 'validation', 'test']:
        for class_name in classes:
            path = os.path.join(destination_folder, split_folder, class_name)
            if not os.path.exists(path):
                os.makedirs(path)

    for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file_name = row['image_id'] + '.jpg'
        class_name = row['dx']
        split = row['split']
        source_file = os.path.join(image_folder, file_name)
        destination_file = os.path.join(destination_folder, split, class_name, file_name)
        try:
            copyfile(source_file, destination_file)
        except FileNotFoundError:
            print('Could not copy file', source_file, 'to', destination_file)
