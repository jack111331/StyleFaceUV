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
        self.stylecodes = torch.load(os.path.join(data_dir, 'tensor-new.pkl'))
        self.coeffs = torch.load(os.path.join(data_dir, '3DMMparam-new.pkl'))
        self.pose_direction = torch.load(os.path.join(data_dir, 'pose_direction-new.pkl')).view(14, 512).type(torch.FloatTensor)
        # FIXME Refactor
        self.env = lmdb.open(
            os.path.join(data_dir, 'lmdb'),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        transform = transforms.Compose(
            [
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True), #(-1, 1)
            ]
        )
        n_worker = 8
        cnt = 0
        with self.env.begin(write=False) as txn:
            size = int(txn.get('size'.encode('utf-8')).decode('utf-8'))

        yaws = self.coeffs[:, 225].detach().numpy()
        if clean == True:
            indices = np.where(np.abs(yaws) < 0.5236)[0] # 30 degrees in radian
        else:
            indices = np.arange(yaws.shape[0])
        self.data = {}


        with multiprocessing.Pool(n_worker) as pool:
            for i in pool.imap_unordered(dispatch_work, indices):
                with self.env.begin(write=False) as txn:
                    key = f'{str(i).zfill(5)}'.encode('utf-8')
                    img = np.frombuffer(txn.get(key), dtype=np.uint8)
                    img = Image.fromarray(img.reshape(size, size, 3))
                    img = transform(img)
                    img = (img+1)/2
                    self.data[i] = [self.stylecodes[i], img, self.coeffs[i], self.pose_direction]
        self.data = list(self.data.values())
        self.cnt = len(self.data)

    def __len__(self):
        return self.cnt
        #return int(self.img.size(0)*1)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    dataset = StyleCodeImage3DMMParamsPoseDirDataset('./data/', clean=True)
