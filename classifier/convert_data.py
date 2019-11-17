import os
from shutil import copyfile
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    valid_split = 0.1
    test_split = 0.1
    data_folder = 'data'
    destination_folder = 'balanced_data'
    image_folder = os.path.join(data_folder, 'images')

    metadata_file = os.path.join(data_folder, 'HAM10000_metadata.csv')

    metadata = pd.read_csv(metadata_file)

    classes = metadata['dx'].unique()

    metadata['split'] = ''

    for class_name in classes:
        count = len(metadata.loc[metadata['dx'] == class_name])

        splits = ['train'] * int(count * (1 - test_split - valid_split)) + \
                 ['validation'] * int(count * valid_split) + \
                 ['test'] * int(count * test_split)

        np.random.shuffle(splits)

        metadata.loc[metadata['dx'] == class_name]['path'] = splits

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

    # for class_name in classes:
    #     path = os.path.join(destination_folder, class_name)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #
    # for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
    #     file_name = row['image_id'] + '.jpg'
    #     classification = row['dx']
    #     source_file = os.path.join(image_folder, file_name)
    #     destination_file = os.path.join(destination_folder, classification, file_name)
    #     try:
    #         copyfile(source_file, destination_file)
    #     except FileNotFoundError:
    #         print('Could not copy file', source_file, 'to', destination_file)
    #
    # for folder in ['train', 'validation', 'test']:
    #     split_folder = os.path.join(destination_folder, folder)
    #     if not os.path.exists(split_folder):
    #         os.makedirs(split_folder)
    #
    # for class_name in os.listdir(destination_folder):
    #     class_folder = os.path.join(destination_folder, class_name)
    #     dst_class_folder = os.path.join(destination_folder, split_folder, class_name)
    #     files = os.listdir(class_folder)
    #     count = len(files)
    #     i = int((1 - valid_split - test_split) * count)
    #     j = int((1 - test_split) * count)
    #
    #     np.random.shuffle(files)
    #     for index, file in enumerate(files):
    #         if index < i:
    #             split_folder = 'train'
    #         elif index < j:
    #             split_folder = 'validation'
    #         else:
    #             split_folder = 'test'
    #
    #         if not os.path.exists(dst_class_folder):
    #             os.makedirs(dst_class_folder)
    #
    #         src_path = os.path.join(class_folder, file)
    #         dst_path = os.path.join(dst_class_folder, file)
    #
    #         os.rename(src_path, dst_path)
