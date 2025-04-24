# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import transforms as T

class FineFWIDataset(Dataset):
    ''' FWI dataset
    For convenience, in this class, a batch refers to a npy file 
    instead of the batch used during training.

    Args:
        anno: path to annotation file
        sample_ratio: downsample ratio for seismic data
        transform_data|label: transformation applied to data or label
    '''
    def __init__(self, anno, sample_ratio=1, 
                    transform_data=None, transform_label=None, expand_label_zero_dim=True, expand_data_zero_dim=False, squeeze=False):
        if not os.path.exists(anno):
            print(f'Annotation file {anno} does not exists')
        self.sample_ratio = sample_ratio
        self.transform_data = transform_data
        self.transform_label = transform_label
        self.expand_label_zero_dim = expand_label_zero_dim
        self.expand_data_zero_dim = expand_data_zero_dim
        self.squeeze = squeeze
        assert anno[-4:] == '.csv'
        self.df = pd.read_csv(anno)
        assert set(self.df.columns).intersection({'id', 'absolute_x_path', 'absolute_y_path'}) == {'id', 'absolute_x_path', 'absolute_y_path'}
        assert self.df['absolute_x_path'].apply(lambda x: x[-4:] == '.npy').all()
        assert self.df['absolute_y_path'].apply(lambda x: x[-4:] == '.npy').all()
        
    def __getitem__(self, idx):
        data = np.load(self.df.iloc[idx]['absolute_x_path'])[ :, ::self.sample_ratio, :].astype(np.float32)
        
        label = np.load(self.df.iloc[idx]['absolute_y_path']).astype(np.float32)
        if self.expand_label_zero_dim:
            label = np.expand_dims(label, axis=0)
        if self.expand_data_zero_dim:
            data = np.expand_dims(data, axis=0)
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label and label is not None:
            label = self.transform_label(label)
        if self.squeeze:
            label = label.squeeze()
            data = data.squeeze()
        return data, label if label is not None else np.array([])
        
    def __len__(self):
        return len(self.df)

if __name__ == '__main__':
    transform_data = Compose([
        T.LogTransform(k=1),
        T.MinMaxNormalize(T.log_transform(-61, k=1), T.log_transform(120, k=1))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(2000, 6000)
    ])
    dataset = FineFWIDataset(f'generated_ds.csv', transform_data=transform_data, transform_label=transform_label)
    data, label = dataset[0]
    print(data.shape)
    print(label is None)
