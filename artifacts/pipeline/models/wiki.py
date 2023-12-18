from torch.utils.data import DataLoader
import torch
from hetu.gpu_ops.Node import Op
from hetu import ndarray

import numpy as np
import os.path as osp
import h5py

from more_itertools import peekable

def tensor2ndarray(x):
    return ndarray.array(x.numpy(), ctx=ndarray.cpu())

class WikiCorpusDataset():
    def __init__(self, data_name, rank, nrank):
        self.data_name = data_name
        self.rank = rank
        self.nrank = nrank
        self.data_id = rank
        self.reload_data()

    def reload_data(self):
        directory = osp.expanduser("~/.cache/hetu/datasets/wikicorpus_en/")
        fname = directory + "wikicorpus_en_training_{}.hdf5".format(self.data_id)
        if not osp.exists(fname):
            self.data_id = self.rank
            fname = directory + "wikicorpus_en_training_{}.hdf5".format(self.data_id)
        assert osp.exists(fname)
        self.data_id += self.nrank
        f = h5py.File(fname, mode='r')
        if self.data_name == "input_ids":
            self.data = f["input_ids"][:]
        elif self.data_name == "token_type_ids":
            self.data = f["segment_ids"][:]
        elif self.data_name == "masked_lm_labels":
            masked_lm_positions = f["masked_lm_positions"][:]
            masked_lm_ids = f["masked_lm_ids"][:]
            self.data = np.ones(f["input_ids"].shape, dtype=np.int64) * -1
            # store number of masked tokens in index
            n, max_pred_len = masked_lm_positions.shape
            x = np.arange(n).repeat(max_pred_len).reshape(n, max_pred_len)
            self.data[(x, masked_lm_positions)] = masked_lm_ids
            self.data[:, 0] = -1
        elif self.data_name == "next_sentence_label":
            self.data = f["next_sentence_labels"][:]
        elif self.data_name == "attention_mask":
            self.data = f["input_mask"][:]
        else:
            raise NameError("Data name not correct.")
        f.close()
        self.data = torch.Tensor(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

    def shape(self):
        return self.data[0].shape

class WikiCorpusDataLoader(Op):
    def __init__(self, batch_size, data_name, transform=None, tondarry=True):
        super().__init__(WikiCorpusDataLoader, [], ndarray.cpu(0))
        self.on_gpu = True
        self.on_cpu = False
        self.name = "WikiCorpusDataLoader"
        self.desc = self.name
        self.batch_size = batch_size
        self.tondarry=tondarry
        self.transform = transform
        self.data_name = data_name

    def get_batch_num(self, name):
        if name=="train":
            return 100
        else:
            assert False

    def get_arr(self, name):
        if name=="train":
            return next(self.dl_train_gen)
        else:
            assert False

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
        self.train_data = WikiCorpusDataset(self.data_name, rank, nrank)
        gen = torch.Generator()
        gen.manual_seed(rank)
        self.dl_train = DataLoader(
            self.train_data, shuffle=False, drop_last=True,
            num_workers=1, batch_size=self.batch_size, generator=gen)
        self.dl_train_gen = peekable(self.get_generator(self.dl_train))

    def get_generator(self, dataloader):
        while True:
            for _, x in enumerate(dataloader):
                if self.transform:
                    x = self.transform(x)
                if self.tondarry:
                    yield tensor2ndarray(x)
                else:
                    yield x
            self.train_data.reload_data()


