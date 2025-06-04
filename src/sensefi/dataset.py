import numpy as np
import glob
import os
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader


def UT_HAR_dataset(root_dir):
    data_list = glob.glob(root_dir+'/data/*.csv')
    label_list = glob.glob(root_dir+'/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data


# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp'):
        """
        Args:
            root_dir (string): Directory with all the .mat files organized in subfolders.
            modal (CSIamp/CSIphase): CSI data modality key in the .mat files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.modal = modal

        # Build a sorted list of class folders and a mapping to integer labels
        classes = sorted(glob.glob(root_dir + '/*/'))
        class_to_idx = {os.path.basename(c[:-1]): i for i, c in enumerate(classes)}

        # Build a flat list of (file_path, label) pairs
        self.samples = []
        for c in classes:
            label = class_to_idx[os.path.basename(c[:-1])]
            for mat_file in glob.glob(c + '*.mat'):
                self.samples.append((mat_file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Support slicing: return a batch of samples
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
            samples = [self[i] for i in indices]
            xs, ys = zip(*samples)
            return torch.stack(xs, dim=0), ys

        # Handle a single index
        path, label = self.samples[idx]
        mat_data = sio.loadmat(path)
        x = mat_data[self.modal]

        # Pre-process: normalize and downsample
        x = (x - 42.3199) / 4.9802
        x = x[:, ::4]
        x = x.reshape(3, 114, 500)

        x = torch.from_numpy(x).float()
        return x, label


class Widar_Dataset(Dataset):
    def __init__(self, root_dir):       
        self.samples = []
        classes = sorted(glob.glob(root_dir+'/*/'))
        class_to_idx = {os.path.basename(c[:-1]): i for i, c in enumerate(classes)}
        for c in classes: 
            label = class_to_idx[os.path.basename(c[:-1])]
            for csv_file in glob.glob(c + '*.csv'):
                self.samples.append((csv_file, label))
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):        
            # Compute the actual indices this slice refers to
            indices = list(range(*idx.indices(len(self))))
            samples = [self[i] for i in indices]
            
            # Separate inputs and labels
            xs, ys = zip(*samples)
            # ys = list(ys)
                       
            return torch.stack(xs, dim=0), ys
        
        path, label = self.samples[idx]
        x = np.genfromtxt(path, delimiter=',')
        
        # Pre-process and convert to tensor
        x = (x - 0.0025)/0.0119
        x = x.reshape(22, 20, 20)
        x = torch.from_numpy(x).float()
        
        return x, label
    