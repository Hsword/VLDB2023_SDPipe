from torch.utils.data import DataLoader
import torch
from hetu.gpu_ops.Node import Op
from hetu import ndarray
import hetu as ht

import numpy as np
import os.path as osp

from more_itertools import peekable

def tensor2ndarray(x):
    return ndarray.array(x.numpy(), ctx=ndarray.cpu())

class BookCorpusDataset():
    def __init__(self, data_name, doc_num=200, save_gap=200):
        directory = osp.expanduser("~/.cache/hetu/datasets/bookcorpus/")
        self.data = []
        for i in range(0, doc_num,save_gap):
            start, end = i, i+save_gap-1
            if end > doc_num-1:
                end = doc_num-1
            range_name = '_%d_%d.npy'%(start,end)
            self.data.append(np.load(directory+data_name+range_name))
        self.data = np.concatenate(self.data, axis=0)
        self.data = torch.Tensor(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

    def shape(self):
        return self.data[0].shape

    def init_state(self, rank, nrank):
        cur_size = self.data.shape[0] // nrank
        start = cur_size * rank
        ending = start + cur_size
        self.data = self.data[start:ending]

class BookCorpusDataLoader(Op):
    def __init__(self, batch_size, data_name, transform=None, tondarry=True):
        super().__init__(BookCorpusDataLoader, [], ndarray.cpu(0))
        self.on_gpu = True
        self.on_cpu = False
        self.name = "BookCorpusDataLoader"
        self.desc = self.name
        self.batch_size = batch_size
        self.train_data = BookCorpusDataset(data_name)
        self.tondarry=tondarry
        self.transform = transform

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
        return self.dl_train_gen.peek().shape

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
        self.train_data.init_state(rank, nrank)
        # self.test_data.init_state(rank, nrank)
        gen = torch.Generator()
        gen.manual_seed(rank)
        self.dl_train = DataLoader(
            self.train_data, shuffle=False, drop_last=True,
            num_workers=1, batch_size=self.batch_size, generator=gen)
        # self.dl_test = DataLoader(
        #     self.test_data, shuffle=False, drop_last=True,
        #     num_workers=1, batch_size=self.batch_size)
        self.dl_train_gen = peekable(self.get_generator(self.dl_train))
        # self.dl_test_gen = self.get_generator(self.dl_test)

    def get_generator(self, dataloader):
        while True:
            for _, x in enumerate(dataloader):
                if self.transform:
                    x = self.transform(x)
                if self.tondarry:
                    yield tensor2ndarray(x)
                else:
                    yield x


