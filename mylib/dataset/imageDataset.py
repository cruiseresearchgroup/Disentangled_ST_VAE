import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        img1 = self.data_tensor[index1]
        img2 = self.data_tensor[index2]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

    def __len__(self):
        return self.data_tensor.size(0)

def return_data(name, dset_dir, image_size):
    # assert image_size == 64, 'currently only image size of 64 is supported'

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),])

    # print(name)

    if name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    elif name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='latin1')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset
    elif name.lower() == 'taxibj_13_t':
        root = os.path.join(dset_dir, 'TaxiBJ_13_t.npz')
        data = np.load(root)
        data = torch.from_numpy(data['X']).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset
    elif name == 'TaxiBJ_13_image':
        root = os.path.join(dset_dir, 'TaxiBJ_13_image.npz')
        data = np.load(root)
        print(data['X'].shape)
        data = torch.from_numpy(data['X']).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset
    
    else:
        raise NotImplementedError


    train_dset = dset(**train_kwargs)
    return train_dset