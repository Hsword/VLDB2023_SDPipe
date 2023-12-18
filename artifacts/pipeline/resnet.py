import hetu as ht
from hetu import init
from hetu import ndarray

from datasets.cifar100 import CIFAR100DataLoader
from datasets.imagenet import ImageNetDataLoader

def conv2d(x, in_channel, out_channel, stride=1, padding=1, kernel_size=3, name=''):
    limit = 1 / (kernel_size * kernel_size * in_channel) ** 0.5
    weight = init.random_uniform(
        shape=(out_channel, in_channel, kernel_size, kernel_size),
        minval=-limit, maxval=limit, name=name+'_weight')
    x = ht.conv2d_op(x, weight, stride=stride, padding=padding)
    return x


def batch_norm_with_relu(x, hidden, name):
    scale = init.ones(shape=(1, hidden, 1, 1), name=name+'_scale')
    bias = init.zeros(shape=(1, hidden, 1, 1), name=name+'_bias')
    x = ht.batch_normalization_op(x, scale, bias, momentum=0.1, eps=1e-5)
    x = ht.relu_op(x)
    return x

def batch_norm(x, hidden, name):
    scale = init.ones(shape=(1, hidden, 1, 1), name=name+'_scale')
    bias = init.zeros(shape=(1, hidden, 1, 1), name=name+'_bias')
    x = ht.batch_normalization_op(x, scale, bias, momentum=0.1, eps=1e-5)
    return x

def bottleneck(x, input_channel, channel, stride=1, name=''):
    # bottleneck architecture used when layer > 34
    # there are 3 block in reset that should set stride to 2
    # when channel expands, use 11 conv to expand identity
    output_channel = 4 * channel
    x = ht.relu_op(x)
    shortcut = x
    x = conv2d(x, input_channel, channel, stride=stride, kernel_size=1, padding=0, name=name+'_conv11a')
    x = batch_norm_with_relu(x, channel, name+'_bn1')

    x = conv2d(x, channel, channel, kernel_size=3, padding=1, name=name+'_conv33')
    x = batch_norm_with_relu(x, channel, name+'_bn2')

    x = conv2d(x, channel, output_channel, kernel_size=1, padding=0, name=name+'_conv11b')
    x = batch_norm(x, output_channel, name+'_bn2')

    if input_channel != output_channel:
        shortcut = conv2d(shortcut, input_channel, output_channel,
            kernel_size=1, stride=stride, padding=0, name=name+'_conv11c')
        shortcut = batch_norm(shortcut, output_channel, name+'_bn3')

    x = x + shortcut
    # x = ht.relu_op(x)

    return x, output_channel

def basic_block(x, input_channel, output_channel, stride=1, name=''):
    # there are 3 block in reset that should set stride to 2
    # when channel expands, use 11 conv to expand identity
    x = ht.relu_op(x)
    shortcut = x
    x = conv2d(x, input_channel, output_channel, stride=stride, kernel_size=3, name=name+'_conv33a')
    x = batch_norm_with_relu(x, output_channel, name+'_bn1')

    x = conv2d(x, output_channel, output_channel, stride=1, kernel_size=3, name=name+'_conv33b')
    x = batch_norm(x, output_channel, name+'_bn2')

    if input_channel != output_channel or stride > 1:
        shortcut = conv2d(shortcut, input_channel, output_channel,
            kernel_size=1, stride=stride, padding=0, name=name+'_conv11')
        shortcut = batch_norm(shortcut, output_channel, name+'_bn3')

    x = x + shortcut
    # x = ht.relu_op(x)

    return x, output_channel

def fc(x, shape, name):
    limit = 1 / shape[0] ** 0.5
    weight = init.random_uniform(shape=shape, minval=-limit, maxval=limit, name=name+'_weight')
    bias = init.zeros(shape=shape[-1:], name=name+'_bias')
    x = ht.matmul_op(x, weight)
    x = x + ht.broadcastto_op(bias, x)
    return x


def resnet(dataset, batch_size, num_layers, device_list):
    '''
    Parameters:
        dataset: cifar100, imagenet
        num_layers: 18, 34, 50, 101, 152
        device_list: a list that decide the pipeline partition
            each item in the list is like [rgpu("xxx", 0), rgpu("xxx", 1)]
            it should be same length as block number in resnet
    Return:
        loss: Variable(hetu.gpu_ops.Node.Node), shape (1,)
        y: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    cur_channel = 64
    channels = [64, 128, 256, 512]
    if num_layers == 18:
        layers = [2, 2, 2, 2]
    elif num_layers == 34:
        layers = [3, 4, 6, 3]
    elif num_layers == 50:
        layers = [3, 4, 6, 3]
    elif num_layers == 101:
        layers = [3, 4, 23, 3]
    elif num_layers == 152:
        layers = [3, 8, 36, 3]
    else:
        assert False
    if num_layers > 34:
        block = bottleneck
    else:
        block = basic_block
    assert len(device_list) == sum(layers)

    layer_id = 0
    with ht.context(device_list[layer_id]):
        if dataset == "cifar100":
            x = CIFAR100DataLoader(batch_size, image=True)
            x = conv2d(x, 3, cur_channel, stride=1, padding=1,
                    name='resnet_initial_conv')
            x = batch_norm(x, cur_channel, 'resnet_initial_bn')
        else:
            x = ImageNetDataLoader(batch_size, image=True)
            x = conv2d(x, 3, cur_channel, kernel_size=7, stride=2, padding=1, name='resnet_initial_conv')
            x = batch_norm_with_relu(x, cur_channel, 'resnet_initial_bn')
            x = ht.max_pool2d_op(x, 3, 3, padding=1, stride=2)

    for i in range(len(layers)):
        for k in range(layers[i]):
            with ht.context(device_list[layer_id]):
                stride = 2 if k == 0 and i > 0 else 1
                x, cur_channel = block(
                    x, cur_channel, channels[i], stride=stride,
                    name='resnet_block_{}_{}'.format(i, k)
                )
                layer_id += 1

    num_class = 100 if dataset=="cifar100" else 1000
    with ht.context(device_list[-1]):
        x = ht.relu_op(x)
        x = ht.reduce_mean_op(x, [2, 3]) # H, W
        y = fc(x, (cur_channel, num_class), name='resnet_final_fc')
        if dataset == "cifar100":
            y_ = CIFAR100DataLoader(batch_size, image=False)
        else:
            y_ = ImageNetDataLoader(batch_size, image=False)
        loss = ht.softmaxcrossentropy_op(y, y_, use_cudnn=True)
        loss = ht.reduce_mean_op(loss, [0])

    return loss, y, y_

def get_resnet_partition(num_layers):
    if num_layers == 18:
        layers = [2, 2, 2, 2]
    elif num_layers == 34:
        layers = [4, 4, 4, 4]
    elif num_layers == 50:
        layers = [4, 4, 4, 4]
    elif num_layers == 101:
        layers = [8, 9, 8, 8]
    elif num_layers == 152:
        layers = [12, 13, 13, 12]
    else:
        assert False
    return layers
