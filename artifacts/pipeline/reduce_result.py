from device_list import my_device
import numpy as np

from hetu import new_group_comm, ndarray
from hetu.context import DeviceGroup
from hetu.communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t

_comm = None

def reduce_result(array):
    global _comm
    if _comm is None:
        # last layer devices are output devices
        _comm = new_group_comm(DeviceGroup(my_device[-1]))
    if not isinstance(array, ndarray.NDArray):
        array = ndarray.array(array, ctx=ndarray.gpu(_comm.device_id))
    _comm.dlarrayNcclAllReduce(array, array, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclAvg)
    _comm.stream.sync()
    return array.asnumpy()
