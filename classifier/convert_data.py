import os
from shutil import copyfile
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    data_folder = 'data'
    destination_folder = 'formatted_data'
    image_folder = os.path.join(data_folder, 'images')

    metadata_file = os.path.join(data_folder, 'HAM10000_metadata.csv')

    metadata = pd.read_csv(metadata_file)

    classes = metadata['dx'].unique()

    for class_name in classes:
        path = os.path.join(destination_folder, class_name)
        if not os.path.exists(path):
            os.makedirs(path)

    for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file_name = row['image_id'] + '.jpg'
        classification = row['dx']
        source_file = os.path.join(image_folder, file_name)
        destination_file = os.path.join(destination_folder, classification, file_name)
        try:
            copyfile(source_file, destination_file)
        except FileNotFoundError:
            print('Could not copy file', source_file, 'to', destination_file)

