import os
import time

# tensorboard --logdir='runs' --port=6006 --host='0.0.0.0'

def get_partition(device_list, partition):
    n = len(partition)
    result = []
    for i in range(n):
        for j in range(partition[i]):
            result.append(device_list[i])
    return result

def get_tensorboard_writer(name=""):
    from torch.utils.tensorboard import SummaryWriter
    if not name:
        name = time.ctime().replace(' ', '_')
    writer = SummaryWriter(log_dir=os.path.join("runs", name))
    return writer

def get_lr_scheduler(initial_lr : float, decay : float, decay_step : int, warmup_step=0):
    def train_scheduler():
        lr = initial_lr
        while True:
            for i in range(decay_step):
                yield lr
            lr = lr * decay

    def warmup_scheduler():
        lr = 0
        for i in range(warmup_step):
            lr += initial_lr / warmup_step
            yield lr

    class Scheduler():
        def __init__(self, gen):
            self.gen = gen
            self.val = next(gen)
        def get(self):
            return self.val
        def step(self):
            self.val = next(gen)
            return self.val

    from itertools import chain
    gen = chain(warmup_scheduler(), train_scheduler())
    return Scheduler(gen)
