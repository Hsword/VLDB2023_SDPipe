import hetu
from hetu.ndarray import gpu
from .communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t
from hetu import get_worker_communicate, wrapped_mpi_nccl_init, new_group_comm

import numpy as np
import ctypes
import random

class PartialReduceBase:
    def __init__(self, reduce_key) -> None:
        self._reduce_key = reduce_key
        self.ps_comm = get_worker_communicate()
        self.comm = wrapped_mpi_nccl_init()
        self._comm_map = {}
        self.rank = self.comm.rank
        self.nrank = self.comm.nrank
        self.same_group_worker = self._get_same_key_worker(reduce_key)
        self.comm_all = self.get_communicator(tuple(self.same_group_worker))

    def _get_same_key_worker(self, reduce_key):
        val = np.zeros(self.nrank)
        val[self.rank] = reduce_key
        val = hetu.ndarray.array(val, ctx=gpu(self.comm.device_id))
        self.comm.dlarrayNcclAllReduce(val, val, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum, None)
        self.comm.stream.sync()
        val = val.asnumpy()
        return list(np.where(np.abs(val - reduce_key) < 1e-3)[0])

    def preduce(self, array, partner, stream=None):
        # array : the array to reduce on
        # partner : the partial reduce group returned by get_partner
        # stream : the stream to run allreduce on
        comm = self.get_communicator(partner, stream)
        comm.dlarrayNcclAllReduce(array, array, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclAvg, stream)

    def _create_partial_comm(self, partner, stream):
        self._comm_map[partner] = new_group_comm(partner, stream)

    def get_communicator(self, partner, stream=None):
        if partner not in self._comm_map.keys():
            self._create_partial_comm(partner, stream)
        return self._comm_map[partner]


class PartialReduce(PartialReduceBase):
    def __init__(self, reduce_key, max_worker, ssp_bound, sync_every, vertical_key=-1):
        super(PartialReduce, self).__init__(reduce_key)
        # reduce_key : in pipeline case, worker on each stage use a unique key
        self._buffer = np.ascontiguousarray(np.repeat(-1, self.nrank + 2).astype(np.int32))
        self._buffer_ptr = self._buffer.ctypes.data_as(ctypes.c_void_p)
        self._wait_time = 1
        self._step = 1
        self._mean_partner = 0
        self._control_flag = False
        self._batch_id = 0
        self._vertical_key = vertical_key if vertical_key >= 0 else self.rank
        self.ps_comm.preduce_init(reduce_key, self.rank, max_worker, ssp_bound, sync_every)

    def get_partner(self, min_worker=2, sync=False):
        # wait_time : the max time to wait, in millisecond
        # max_worker : if max_worker reachs, get_partner will return immediately
        #               in pipeline case, max_worker should be set properly, otherwise -1 is ok
        self._batch_id += 1
        self._min_worker = min_worker
        timestamp = self.ps_comm.preduce_get_partner(
            self._reduce_key, self.rank, ctypes.c_int(self._vertical_key),
            ctypes.c_uint64(self._batch_id), ctypes.c_float(self._wait_time),
            self._buffer_ptr)
        if not sync:
            return timestamp
        else:
            return self.async_wait(timestamp)

    def async_wait(self, timestamp):
        self.ps_comm.wait_timestamp(timestamp)
        result = None
        self._control_flag = (self._buffer[0] == 1)
        for i in range(len(self._buffer)):
            if self._buffer[i] < 0:
                result = tuple(self._buffer[1 : i])
                break
        assert result is not None
        if len(result) < self._min_worker:
            self._wait_time = min(self._wait_time * 2, 10)
        else:
            self._wait_time = max(self._wait_time * 0.9, 0.01)
        self._mean_partner = (self._mean_partner * self._step + len(result)) / (self._step + 1)
        self._step += 1
        # print(self._wait_time, result)
        return result

    def remove_partial_comm(self, partner):
        if not partner in self._comm_map.keys():
            return
        elif len(partner) <= 2:
            return # cached
        del self._comm_map[partner]

    @property
    def mean(self):
        return self._mean_partner

    @property
    def control_flag(self):
        return self._control_flag

    def reset_mean(self):
        self._step = 0

class ADPSGD(PartialReduceBase):
    def __init__(self, reduce_key, max_worker, ssp_bound, sync_every, vertical_key=-1):
        # reduce_key : in pipeline case, worker on each stage use a unique key
        super(ADPSGD, self).__init__(reduce_key)
        self._control_flag = False
        self._batch_id = 0
        self.rng = random.Random()
        self.rng.seed(0)
        self.sync_every = sync_every

        assert len(self.same_group_worker) % 2 == 0

    def get_partner(self, min_worker=2, sync=False):
        # wait_time : the max time to wait, in millisecond
        # max_worker : if max_worker reachs, get_partner will return immediately
        #               in pipeline case, max_worker should be set properly, otherwise -1 is ok
        self._batch_id += 1
        self._control_flag = (self._batch_id % self.sync_every) == 0
        partner = self.same_group_worker.copy()
        self.rng.shuffle(partner)
        idx = partner.index(self.rank)
        if idx % 2 == 0:
            return (partner[idx], partner[idx+1])
        else:
            return (partner[idx-1], partner[idx])


    def async_wait(self, timestamp):
        return timestamp

    @property
    def mean(self):
        return 2

    @property
    def control_flag(self):
        return self._control_flag

    def reset_mean(self):
        self._step = 0
