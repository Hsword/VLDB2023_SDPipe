from hetu.gpu_ops.Node import Op
from hetu import ndarray
import hetu as ht

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import os.path as osp

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

def tensor2ndarray(x):
    return ndarray.array(x.numpy(), ctx=ndarray.cpu())

class CIFAR100Dataset():
    def __init__(self, train=True, image=True):
        directory = osp.expanduser("~/.cache/hetu/datasets/CIFAR_100")
        train_set_x, train_set_y, valid_set_x, valid_set_y = ht.data.cifar100(directory=directory)
        if train and image:
            x = train_set_x
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
            ])
        elif not train and image:
            x = valid_set_x
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
            ])
        elif train and not image:
            x = train_set_y
            self.transform = transforms.Compose([])
        else:
            x = valid_set_y
            self.transform = transforms.Compose([])
        self.data = torch.Tensor(x)

    def __getitem__(self, index):
        return self.transform(self.data[index])

    def __len__(self):
        return self.data.shape[0]

    def init_state(self, rank, nrank):
        cur_size = self.data.shape[0] // nrank
        start = cur_size * rank
        ending = start + cur_size
        self.data = self.data[start:ending]

class CIFAR100DataLoader(Op):

    def __init__(self, batch_size, image, tondarry=True):
        super().__init__(CIFAR100DataLoader, [], ndarray.cpu(0))
        self.on_gpu = True
        self.on_cpu = False
        self.name = "CIFAR100DataLoader"
        self.desc = self.name
        self.batch_size = batch_size
        self.train_data = CIFAR100Dataset(train=True, image=image)
        self.test_data = CIFAR100Dataset(train=False, image=image)
        self.image = image # image or label
        self.tondarry=tondarry

    def get_batch_num(self, name):
        if name=="train":
            return len(self.dl_train) // self._data_split
        else:
            return len(self.dl_test)

    def get_arr(self, name):
        if name=="train":
            return next(self.dl_train_gen)
        else:
            return next(self.dl_test_gen)

    def get_cur_shape(self, name):
        if self.image:
            return (self.batch_size, 3, 32, 32)
        else:
            return (self.batch_size, 100)

    def infer_shape(self, input_shapes):
        raise NotImplementedError

    def gradient(self, output_grad):
        return None

    def backward_hook(self, config):
        if config.pipeline:
            rank, nrank = config.pipeline_dp_rank, config.nrank // config.pipeline_nrank
        elif config.context_launch:
            rank, nrank = config.rank, config.nrank
        else:
            rank, nrank = 0, 1
        self._data_split = nrank
        self.test_data.init_state(rank, nrank)
        gen = torch.Generator()
        gen.manual_seed(rank)
        self.dl_train = DataLoader(
            self.train_data, shuffle=True, drop_last=True,
            num_workers=4, batch_size=self.batch_size, generator=gen)
        self.dl_test = DataLoader(
            self.test_data, shuffle=False, drop_last=True,
            num_workers=4, batch_size=self.batch_size)
        self.dl_train_gen = self.get_generator(self.dl_train)
        self.dl_test_gen = self.get_generator(self.dl_test)

    def get_generator(self, dataloader):
        while True:
            for batch_idx, x in enumerate(dataloader):
                if self.tondarry:
                    yield tensor2ndarray(x)
                else:
                    yield x

