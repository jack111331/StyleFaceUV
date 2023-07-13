import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
import multiprocessing
import lmdb

class StyleCode3DMMParamsDataset(Dataset):
    def __init__(self, data_dir):
        self.coeffs = torch.load(os.path.join(data_dir, '3DMMparam-new.pkl'))
        self.stylecodes = torch.load(os.path.join(data_dir, 'tensor-new.pkl'))
        self.data = []
        for i in range(int(self.coeffs.size()[0])):
            self.data.append([self.stylecodes[i], self.coeffs[i]])

    def __len__(self):
        return int(self.coeffs.size(0) * 1)

    def __getitem__(self, idx):
        return self.data[idx]


def dispatch_work(index):
    return index

class StyleCodeImage3DMMParamsPoseDirDataset(Dataset):
    def __init__(self, data_dir, clean=False):
        # TODO
        self.stylecodes = torch.load(os.path.join(data_dir, 'tensor-new.pkl'))
        self.coeffs = torch.load(os.path.join(data_dir, '3DMMparam-new.pkl'))
        self.env = lmdb.open(
            os.path.join(data_dir, 'lmdb'),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.transform = transforms.Compose(
            [
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True), #(-1, 1)
            ]
        )
        with self.env.begin(write=False) as txn:
            self.size = int(txn.get('size'.encode('utf-8')).decode('utf-8'))

        yaws = self.coeffs[:, 225].detach().numpy()

        if clean == True:
            indices = np.where(np.abs(yaws) < 0.5236)[0] # 30 degrees in radian
            self.indices_mapping = {}
            for idx, clean_idx in enumerate(indices):
                self.indices_mapping[idx] = clean_idx
        else:
            self.indices_mapping = np.arange(yaws.shape[0])
        self.cnt = len(self.indices_mapping)

    def __len__(self):
        return self.cnt

    def __getitem__(self, idx):
        clean_idx = self.indices_mapping[idx]
        with self.env.begin(write=False) as txn:
            key = f'{str(clean_idx).zfill(5)}'.encode('utf-8')
            img = np.frombuffer(txn.get(key), dtype=np.uint8)
        img = Image.fromarray(img.reshape(self.size, self.size, 3))
        img = self.transform(img)
        img = (img+1)/2
        return self.stylecodes[clean_idx], img, self.coeffs[clean_idx]

if __name__ == '__main__':
    dataset = StyleCodeImage3DMMParamsPoseDirDataset('./data/', clean=True)
