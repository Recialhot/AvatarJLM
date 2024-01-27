import numpy as np
import random
import torch
import glob
import pickle
from torch.utils.data import Dataset

#ysq
class OdtData(Dataset):
    """Odt dataset"""

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = opt['dataloader_batch_size']
        phase = self.opt['phase']
        dataroot = opt['dataroot']
        dataset_type = opt['dataset_type']
        assert dataset_type in ['odt'] and phase == 'test'
        self.filename_list = glob.glob(f'./{dataroot}/*.pkl')
        self.filename_list.sort()
        print('-------------------------------number of {} data is {}'.format(phase, len(self.filename_list)))

    def __len__(self):
        return len(self.filename_list)


    def __getitem__(self, idx):
        filename = self.filename_list[idx]
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        seq_len = data['hmd_position_global_full_gt_list'].shape[0]
        return {'input_signal': data['hmd_position_global_full_gt_list'].reshape(seq_len, -1).float(),
                'global_head_trans': data['head_global_trans_list'],
                }