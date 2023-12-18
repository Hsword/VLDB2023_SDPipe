from hetu.gpu_ops.Node import Op
from hetu import ndarray

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import numpy as np

IMAGENET_TRAIN_MEAN = (0.485, 0.456, 0.406)
IMAGENET_TRAIN_STD = (0.229, 0.224, 0.225)
IMAGENET_TRAIN_ROOT="~/.cache/hetu/datasets/imagenet/train"
IMAGENET_VAL_ROOT="~/.cache/hetu/datasets/imagenet/val"

def tensor2ndarray(x):
    return ndarray.array(x.numpy(), ctx=ndarray.cpu())

def imagenet_onehot_label(x):
    one_hot_vals = np.zeros((len(x), 1000))
    one_hot_vals[np.arange(len(x)), x] = 1
    return ndarray.array(one_hot_vals, ctx=ndarray.cpu())

class DDPSampler(torch.utils.data.Sampler):
    # we want different worker to use different data in validation run
    def __init__(self, rank, nrank, data_len):
        length = data_len // nrank
        self.start = length * rank
        self.end = length * (rank + 1)

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self) -> int:
        return self.end - self.start

class ImageNetDataLoader(Op):

    def __init__(self, batch_size=32, image=True):
        super().__init__(ImageNetDataLoader, [], ndarray.cpu(0))
        self.on_gpu = True
        self.on_cpu = False
        self.name = "ImageNetDataLoader"
        self.desc = self.name
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_MEAN),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_MEAN),
        ])
        self.image = image # image or label

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
            return (self.batch_size, 3, 224, 224)
        else:
            return (self.batch_size, 1000)

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
        self.train_data = ImageFolder(IMAGENET_TRAIN_ROOT, self.transform)
        self.test_data = ImageFolder(IMAGENET_VAL_ROOT, self.transform_test)
        gen = torch.Generator()
        gen.manual_seed(rank)
        self.dl_train = DataLoader(
            self.train_data, shuffle=True, drop_last=True,
            num_workers=4, batch_size=self.batch_size, generator=gen)
        self.dl_test = DataLoader(
            self.test_data,
            sampler=DDPSampler(rank, nrank, len(self.test_data)),
            num_workers=4, batch_size=self.batch_size, drop_last=True)
        self.dl_train_gen = self.get_generator(self.dl_train)
        self.dl_test_gen = self.get_generator(self.dl_test)

    def get_generator(self, dataloader):
        while True:
            for batch_idx, (image, label) in enumerate(dataloader):
                if self.image:
                    yield tensor2ndarray(image)
                else:
                    yield imagenet_onehot_label(label)
