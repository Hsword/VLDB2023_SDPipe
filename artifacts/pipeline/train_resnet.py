import hetu as ht

import os
import sys
import time
import argparse
import numpy as np

from device_list import my_device, add_cpu_ctx
from resnet import resnet, get_resnet_partition
from utils import get_partition, get_tensorboard_writer, get_lr_scheduler
from reduce_result import reduce_result

def validate(executor, val_batch_num):
    res = []
    for i in range(val_batch_num):
        res.append(executor.run("validate", convert_to_numpy_ret_vals=True))
    if res[0]:
        loss_value = []
        accuracy = []
        for iter_result in res:
            loss_value.append(iter_result[0][0])
            correct_prediction = np.equal(np.argmax(iter_result[1], 1), np.argmax(iter_result[2], 1)).mean()
            accuracy.append(correct_prediction)
        return np.mean(loss_value), np.mean(accuracy)
    return None, None

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--dataset', default='cifar100')
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--pipeline', type=str, default="pipedream")
    parser.add_argument('--preduce', action='store_true')
    parser.add_argument('--adpsgd', action='store_true')
    parser.add_argument('--hetero', default=0.0, type=float)
    parser.add_argument('--name', type=str, default="")
    args = parser.parse_args()
    if args.hetero > 0:
        os.environ["DEBUG_HETERO"] = str(args.hetero)

    assert args.pipeline in ["pipedream", "hetpipe", "dp"]
    if args.pipeline == "dp":
        args.pipeline = None # use common data parallel
    assert args.dataset in ["cifar100", "imagenet"]
    assert args.model in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    ht.worker_init()
    np.random.seed(0)

    if args.model.startswith("resnet"):
        num_layers = int(args.model[6:])
        model_partition = get_resnet_partition(num_layers)
    else:
        raise NotImplementedError
    num_data_parallel = len(my_device[0])
    if args.pipeline == "hetpipe":
        assert not args.preduce
        my_device = add_cpu_ctx(my_device)
    # Note : when we use model average, we don't need to modify lr
    # In allreduce case, we use reduce mean, otherwise we have to modify weight decay param
    device_list = get_partition(my_device, model_partition)

    loss, y, y_ = resnet(args.dataset, args.batch_size, num_layers, device_list)

    if args.dataset == "cifar100":
        num_train_image, num_eval_image = 50000, 10000
    elif args.dataset == "imagenet":
        num_train_image, num_eval_image = 1281167, 50000
    train_batch_num = num_train_image // (args.batch_size * num_data_parallel)
    val_batch_num = num_eval_image // (args.batch_size * num_data_parallel)

    if args.dataset == "cifar100":
        lr_scheduler = get_lr_scheduler(args.learning_rate, 0.2, 60 * train_batch_num, train_batch_num)
    elif args.dataset == "imagenet":
        lr_scheduler = get_lr_scheduler(args.learning_rate, 0.1, 30 * train_batch_num, train_batch_num)

    opt = ht.optim.MomentumOptimizer(learning_rate=lr_scheduler, momentum=0.9, l2reg=args.weight_decay)
    with ht.context(device_list[-1]):
        train_op = opt.minimize(loss)
        executor = ht.Executor({"train" : [loss, y, y_, train_op], "validate" : [loss, y, y_]},
            seed=0, pipeline=args.pipeline, use_preduce=args.preduce, use_adpsgd=args.adpsgd, dynamic_memory=True)

    if executor.config.pipeline_dp_rank == 0:
        writer = get_tensorboard_writer(args.name)
    else:
        writer = None

    if executor.rank == 0:
        print("Training {} epoch, each epoch runs {} iteration".format(args.epochs, train_batch_num))

    for iteration in range(args.epochs):
        start = time.time()
        if args.pipeline:
            res = executor.run("train", batch_num = train_batch_num)
        elif args.preduce:
            res = []
            while not executor.subexecutor["train"].preduce_stop_flag:
                res.append(executor.run("train", convert_to_numpy_ret_vals=True))
            executor.subexecutor["train"].preduce_stop_flag = False
        else:
            res = []
            for i in range(train_batch_num):
                res.append(executor.run("train", convert_to_numpy_ret_vals=True))
        if res[0]:
            time_used = time.time() - start
            loss_value = []
            accuracy = []
            for i, iter_result in enumerate(res):
                loss_value.append(iter_result[0][0])
                correct_prediction = np.equal(np.argmax(iter_result[1], 1), np.argmax(iter_result[2], 1)).mean()
                accuracy.append(correct_prediction)
            loss_value, accuracy = reduce_result([np.mean(loss_value), np.mean(accuracy)])
            if writer:
                writer.add_scalar('Train/loss', loss_value, iteration)
                writer.add_scalar('Train/acc', accuracy, iteration)
            if args.preduce:
                preduce_mean = executor.subexecutor["train"].preduce.mean
                print(preduce_mean)
                if writer:
                    writer.add_scalar('Train/Partner', preduce_mean, iteration)
                executor.subexecutor["train"].preduce.reset_mean()
            if executor.config.pipeline_dp_rank == 0:
                print(iteration, "TRAIN loss {:.4f} acc {:.4f} lr {:.2e}, time {:.4f}".format(
                    loss_value, accuracy, opt.learning_rate, time_used))
        if args.dataset == "cifar100":
            opt.set_learning_rate(args.learning_rate * (0.1 ** (iteration // 60)))
        else:
            opt.set_learning_rate(args.learning_rate * (0.2 ** (iteration // 30)))

        val_loss, val_acc = validate(executor, val_batch_num)
        if val_loss:
            val_loss, val_acc = reduce_result([val_loss, val_acc])
            if executor.config.pipeline_dp_rank == 0:
                print(iteration, "EVAL  loss {:.4f} acc {:.4f}".format(val_loss, val_acc))
            if writer:
                writer.add_scalar('Validation/loss', val_loss, iteration)
                writer.add_scalar('Validation/acc', val_acc, iteration)
    ht.worker_finish()
