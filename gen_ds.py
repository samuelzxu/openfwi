import os
from tqdm import tqdm
import numpy as np
import pandas as pd

data_dir = '/home/ziggy/dev/openfwi_unpacked'
files = os.listdir(data_dir)

all_data = list(os.walk(data_dir))
data_file_tups = list(filter(lambda x: len(x[2]) == 2 and x[2][0].endswith('.npy'), all_data))

def test_load_data(data_dir, data_files):
    assert len(data_files) == 2
    absolute_x_path = os.path.join(data_dir, data_files[1])
    absolute_y_path = os.path.join(data_dir, data_files[0])
    assert np.load(absolute_y_path).shape == (70, 70)
    assert np.load(absolute_x_path).shape == (5, 1000, 70)

    return absolute_x_path, absolute_y_path

ds_dict = {
    'id': [],
    'absolute_x_path': [],
    'absolute_y_path': [],
}

for i, data_file_tup in tqdm(enumerate(data_file_tups)):
    data_dir = data_file_tup[0]
    data_files = data_file_tup[2]
    assert test_load_data(data_dir, data_files)
    id = '/'.join(data_dir.split('/')[-3:])
    absolute_y_path = os.path.join(data_dir, data_files[0])
    absolute_x_path = os.path.join(data_dir, data_files[1])
    ds_dict['id'].append(id)
    ds_dict['absolute_y_path'].append(absolute_y_path)
    ds_dict['absolute_x_path'].append(absolute_x_path)

df = pd.DataFrame(ds_dict)
df.to_csv('generated_ds.csv', index=False)

    
    



