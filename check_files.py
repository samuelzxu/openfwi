import os
from tqdm import tqdm
import numpy as np

data_dir = '/home/ziggy/dev/openfwi_unpacked'
files = os.listdir(data_dir)

all_data = list(os.walk(data_dir))
data_file_tups = list(filter(lambda x: len(x[2]) == 2 and x[2][0].endswith('.npy'), all_data))

def test_load_data(data_dir, data_files):
    assert len(data_files) == 2
    assert np.load(os.path.join(data_dir, data_files[0])).shape == (70, 70)
    assert np.load(os.path.join(data_dir, data_files[1])).shape == (5, 1000, 70)
    return True

for data_file_tup in tqdm(data_file_tups):
    data_dir = data_file_tup[0]
    data_files = data_file_tup[2]
    assert test_load_data(data_dir, data_files)


