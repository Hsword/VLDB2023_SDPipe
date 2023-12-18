import numpy as np
import ctypes
from copy import copy, deepcopy
import hetu as ht
from . import ndarray
from . import gpu_links as gpu_op
from .lr_scheduler import FixedScheduler
from .gpu_ops.Node import Op
from .gpu_ops.EmbeddingLookUp import EmbeddingLookUp_Gradient
from .gpu_ops.ParameterServerCommunicate import ParameterServerCommunicateOp
from .gpu_ops.Variable import PlaceholderOp
from . import cpu_links
from .communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t
from ._base import DNNL_LIB

class Optimizer(object):
    """Optimizers."""

    def __init__(self, learning_rate, l2reg=0):
        if isinstance(learning_rate, (int, float)):
            self.lr_sched = FixedScheduler(abs(learning_rate))
        else:
            assert hasattr(learning_rate, "get") and hasattr(learning_rate, "step"), \
                "A learning rate scheduler should define method 'get' and 'step'."
            self.lr_sched = learning_rate
        # now we don't support l2 regularizer for sparse updates
        # TODO: support l2 regularizer for sparse updates
        # now we don't support l2 regularizer for PS mode parameters
        # TODO: support l2 regularizer for PS mode parameters (after PS mode has optimizer on Servers)
        assert l2reg >= 0, 'L2 regularizer should be positive or 0.'
        self.l2reg = l2reg
        self.params = None
        self.tensors = None
        self.initiated = False

    @property
    def learning_rate(self):
        return self.lr_sched.get()

    def set_learning_rate(self, lr):
        self.lr_sched = FixedScheduler(abs(lr))

    def step(self):
        # modify learning rate for the next iteration
        self.lr_sched.step()

    @staticmethod
    def get_var_list(loss):
        def topo_sort_dfs(node, visited, var_list):
            if node in visited:
                return
            visited.add(node)
            if isinstance(node, PlaceholderOp) and node.trainable:
                var_list.append(node)
                return
            for n in node.inputs:
                topo_sort_dfs(n, visited, var_list)

        visited = set()
        trainable_vars = []
        if isinstance(loss, list):
            for l in loss:
                topo_sort_dfs(l, visited, trainable_vars)
        else:
            topo_sort_dfs(loss, visited, trainable_vars)
        return trainable_vars

    def initiate_states(self, config):
        assert not self.initiated, "Optimizer already initiated."
        self.tensors = [config.placeholder_to_arr_map[node]
                        for node in self.params]
        self.initiated = True

    def update_tensors_version(self, tensor_map):
        for i, node in enumerate(self.params):
            if node in tensor_map:
                old_ctx = self.tensors[i].ctx
                self.tensors[i] = tensor_map[node]
                if tensor_map[node].ctx != old_ctx:
                    self.renew_state(i)

    def renew_state(self, i):
        return

    def minimize(self, loss, var_list=None):
        """Return an optimizer op to update parameters.

        Parameters
        ----------
        loss: loss node that we are minimizing.
        var_list: list of nodes that we are taking derivative wrt.

        Returns
        -------
        An optimizer node.

        """
        self.loss = loss
        if not var_list:
            var_list = self.get_var_list(loss)
        self.params = var_list
        grads, self.backward2forward, self.forward2backward = ht.gradients(
            loss, self.params, return_all=True)
        optimizer_node = OptimizerOp(grads, self)
        return optimizer_node

    def apply_l2_reg(self, node, grad, stream_handle=None):
        if self.l2reg == 0:
            return
        i = self.params.index(node)
        weight = self.tensors[i]
        if ndarray.is_gpu_ctx(weight.ctx):
            gpu_op.add_l2_regularization(weight, grad, self.l2reg, stream_handle)
        elif DNNL_LIB['cpu_SGDOptimizerUpdate']:
            cpu_links.add_l2_regularization(weight, grad, self.l2reg)
        else:
            grad[:] = grad.asnumpy() + self.l2reg * weight.asnumpy()

    def apply_gradient(self, node, grad, stream_handle=None):
        i = self.params.index(node)
        weight = self.tensors[i]
        if ndarray.is_gpu_ctx(weight.ctx):
            if isinstance(grad, ndarray.IndexedSlices):
                gpu_op.sgd_update( weight, grad, -1, stream_handle)
            else:
                gpu_op.matrix_elementwise_add_simple(grad, weight, weight, stream_handle)
        elif isinstance(grad, ndarray.IndexedSlices):
            if DNNL_LIB['cpu_SGDOptimizerSparseUpdate']:
                cpu_links.sgd_update_sparse(weight, grad.indices, grad.values, -1)
            else:
                grad.cpu_deduplicate()
                np_tensor = weight.asnumpy()
                np_tensor[grad.indices.asnumpy().astype(np.int)] += grad.values.asnumpy()
                weight[:] = np_tensor
                grad.free_deduplicate()
        elif DNNL_LIB['DnnlMatrixElementwiseAdd']:
            cpu_links.matrix_elementwise_add(grad, weight, weight)
        else:
            weight[:] = weight.asnumpy() + grad.asnumpy()

    def process_gradient(self, node, grad, stream_handle=None):
        # This call will modify grad so that it can be directly applied to weight tensor
        # optimizer should adjust gradient by momentum and l2reg if desired, and multiply it by learning rate

        # abstract method, each optimizer should implement its own
        raise NotImplementedError

    def __deepcopy__(self, memo):
        assert not self.initiated, 'Should not deep copy optimizer if already initiated!'
        new_opt = copy(self)
        new_opt.loss = deepcopy(self.loss, memo)
        new_opt.params = [deepcopy(node, memo) for node in self.params]
        new_opt.backward2forward = dict([(deepcopy(k, memo), (deepcopy(n, memo) for n in v))
                                         for k, v in self.backward2forward.items()])
        new_opt.forward2backward = dict([(deepcopy(k, memo), (deepcopy(n, memo) for n in v))
                                         for k, v in self.forward2backward.items()])
        return new_opt


class OptimizerOp(Op):
    def __init__(self, grads, optimizer):
        super().__init__(OptimizerOp, grads, None)
        self.name = "Optimizer_%s" % (optimizer.name)
        self.optimizer = optimizer

    def compute(self, input_vals, output_val, stream_handle=None):
        assert output_val is None
        # For PS op, this input_vals is None
        # PS mode doesn't need local update
        if self.comm_mode != 'PS':
            self.optimizer.update(input_vals, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None

    def forward_hook(self, config):
        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        for node in self.inputs:
            node.inplace = False

        self.optimizer.initiate_states(config)
        self.on_cpu = self.on_gpu = None
        self.comm_mode = config.comm_mode
        # some things todo.
        if self.comm_mode != 'PS':
            for i in range(len(self.inputs)):
                # Though the gradients for transfer ops are well defined,
                # we called gradients in optimizer op before transfer ops are added.
                # So here we also add tranfer ops for gradients update.
                # Could be optimized later.
                if not isinstance(self.inputs[i], ParameterServerCommunicateOp):
                    paramctx = self.optimizer.params[i].ctx
                    self.inputs[i] = super().add_transfer_op(
                        self.inputs[i], paramctx, config.h2d_ops, config.d2h_ops)

    def backward_hook(self, config):
        self.comm_mode = config.comm_mode
        new_inputs = []
        for i, node in enumerate(self.inputs):
            current_strategy = config.node_strategy.get(
                self.optimizer.params[i], self.comm_mode)
            cur_node = node
            if current_strategy == 'AllReduce' or (current_strategy == 'Hybrid' and not isinstance(node, EmbeddingLookUp_Gradient)):
                cur_node = ht.allreduceCommunicate_op(
                    node, config.param_allreduce_group.get(self.optimizer.params[i], config.nccl_comm))
                if config.layer_indices is not None and node in config.layer_indices:
                    config.layer_indices[cur_node] = config.layer_indices[node] + 1
            elif current_strategy == 'PS' or (current_strategy == 'Hybrid' and isinstance(node, EmbeddingLookUp_Gradient)):
                cur_node = ht.parameterServerCommunicate_op(
                    node, self.optimizer.params[i], self.optimizer)
                if config.layer_indices is not None and node in config.layer_indices:
                    config.layer_indices[cur_node] = config.layer_indices[node] + 1
            new_inputs.append(cur_node)
        self.inputs = new_inputs

    def re_minimize(self):
        new_grads = ht.gradients(self.optimizer.loss, self.optimizer.params)
        self.inputs = new_grads


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, l2reg=0):
        super(SGDOptimizer, self).__init__(learning_rate, l2reg)
        self.name = 'SGD'

    def get_config(self):
        return (ctypes.c_int(0), (ctypes.c_float * 1)(self.learning_rate), ctypes.c_int(1))

    def initiate_states(self, config):
        super().initiate_states(config)

    def process_gradient(self, node, grad, stream_handle=None):
        i = self.params.index(node)
        weight = self.tensors[i]
        self.apply_l2_reg(node, grad, stream_handle)
        if ndarray.is_gpu_ctx(weight.ctx):
            gpu_op.matrix_elementwise_multiply_by_const(
                grad, -self.learning_rate, grad, stream_handle)
        elif isinstance(grad, ndarray.IndexedSlices):
            grad.values[:] = grad.values.asnumpy() * -self.learning_rate
        else:
            grad[:] = grad.asnumpy() * -self.learning_rate

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if grads[i] == None:
                continue
            self.apply_l2_reg(self.params[i], grads[i], stream_handle)
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                gpu_op.sgd_update(
                    self.tensors[i], grads[i], self.learning_rate, stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    if DNNL_LIB['cpu_SGDOptimizerSparseUpdate']:
                        from .cpu_links import sgd_update_sparse as cpu_sgd_update_sparse
                        cpu_sgd_update_sparse(
                            self.tensors[i], grads[i].indices, grads[i].values, self.learning_rate)
                    else:
                        grads[i].cpu_deduplicate()
                        np_tensor = self.tensors[i].asnumpy()
                        np_tensor[grads[i].indices.asnumpy().astype(
                            np.int)] -= self.learning_rate * grads[i].values.asnumpy()
                        self.tensors[i][:] = np_tensor
                        grads[i].free_deduplicate()
                else:
                    if DNNL_LIB['cpu_SGDOptimizerUpdate']:
                        from .cpu_links import sgd_update as cpu_sgd_update
                        cpu_sgd_update(
                            self.tensors[i], grads[i], self.learning_rate)
                    else:
                        self.tensors[i][:] = self.tensors[i].asnumpy() - \
                            self.learning_rate * grads[i].asnumpy()


class MomentumOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, nesterov=False, l2reg=0):
        super(MomentumOptimizer, self).__init__(learning_rate, l2reg)
        self.momentum = momentum
        self.nesterov = nesterov
        self.name = "Momentum"

    def get_config(self):
        return (ctypes.c_int(self.nesterov + 1), (ctypes.c_float * 2)(self.learning_rate, self.momentum), ctypes.c_int(2))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.velocity = []
        for t in self.tensors:
            self.velocity.append(None if t is None else ndarray.array(
                np.zeros(t.shape, dtype=np.float32), t.ctx))

    def renew_state(self, i):
        t = self.tensors[i]
        self.velocity[i] = ndarray.array(np.zeros(t.shape, dtype=np.float32), t.ctx)

    def process_gradient(self, node, grad, stream_handle=None):
        i = self.params.index(node)
        weight = self.tensors[i]
        self.apply_l2_reg(node, grad, stream_handle)
        if ndarray.is_gpu_ctx(weight.ctx) and isinstance(grad, ndarray.NDArray):
            gpu_op.momentum_update(
                weight, grad, self.velocity[i], self.learning_rate, self.momentum,
                self.nesterov, True, stream_handle)
        else:
            raise NotImplementedError

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if grads[i] == None:
                continue
            self.apply_l2_reg(self.params[i], grads[i], stream_handle)
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                assert isinstance(self.velocity[i], ndarray.NDArray)
                gpu_op.momentum_update(self.tensors[i], grads[i], self.velocity[i], self.learning_rate, self.momentum,
                                       self.nesterov, False, stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    from ._base import DNNL_LIB
                    if DNNL_LIB['cpu_MomentumOptimizerUpdate']:
                        from .cpu_links import momentum_update as cpu_momentum_update
                        cpu_momentum_update(self.tensors[i], grads[i], self.velocity[i], self.learning_rate, self.momentum,
                                            self.nesterov)
                    else:
                        prev_param = self.tensors[i].asnumpy()
                        grad = grads[i].asnumpy()
                        velo = self.velocity[i].asnumpy()
                        if self.nesterov:
                            lr_grads = -self.learning_rate * grad
                            self.velocity[i][:] = self.momentum * \
                                (velo + lr_grads)
                            self.tensors[i][:] = prev_param + velo + lr_grads
                        else:
                            self.velocity[i][:] = self.momentum * \
                                velo - self.learning_rate * grad
                            self.tensors[i][:] = prev_param + velo


class AdaGradOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, initial_accumulator_value=0.0, eps=1e-7, l2reg=0):
        assert initial_accumulator_value >= 0.0, \
            "initial accumulator value must be non-negative"
        assert eps > 0.0, \
            "epsilon must be positive"
        super(AdaGradOptimizer, self).__init__(learning_rate, l2reg)
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.name = "AdaGrad"

    def get_config(self):
        return (ctypes.c_int(3), (ctypes.c_float * 3)(self.learning_rate, self.initial_accumulator_value, self.eps), ctypes.c_int(3))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.accumulator_value = []
        for t in self.tensors:
            self.accumulator_value.append(None if t is None else ndarray.array(
                np.full(t.shape, self.initial_accumulator_value), t.ctx))

    def renew_state(self, i):
        t = self.tensors[i]
        self.accumulator_value[i] = ndarray.array(np.full(t.shape, self.initial_accumulator_value), t.ctx)

    def process_gradient(self, node, grad, stream_handle=None):
        i = self.params.index(node)
        weight = self.tensors[i]
        self.apply_l2_reg(node, grad, stream_handle)
        if ndarray.is_gpu_ctx(weight.ctx) and isinstance(grad, ndarray.NDArray):
            gpu_op.adagrad_update(
                weight, grad, self.accumulator_value[i],
                self.learning_rate, self.eps, True ,stream_handle)
        else:
            raise NotImplementedError

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if grads[i] == None:
                continue
            self.apply_l2_reg(self.params[i], grads[i], stream_handle)
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                gpu_op.adagrad_update(self.tensors[i], grads[i], self.accumulator_value[i],
                    self.learning_rate, self.eps, False, stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    from ._base import DNNL_LIB
                    if DNNL_LIB['cpu_AdaGradOptimizerUpdate']:
                        from .cpu_links import adagrad_update as cpu_adagrad_update
                        cpu_adagrad_update(
                            self.tensors[i], grads[i], self.accumulator_value[i], self.learning_rate, self.eps)
                    else:
                        prev_param = self.tensors[i].asnumpy()
                        grad = grads[i].asnumpy()
                        self.accumulator_value[i][:] = self.accumulator_value[i].asnumpy(
                        ) + np.power(grad, 2)
                        self.tensors[i][:] = \
                            prev_param - self.learning_rate * grad / \
                            (np.sqrt(
                                self.accumulator_value[i].asnumpy()) + self.eps)


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, l2reg=0):
        super(AdamOptimizer, self).__init__(learning_rate, l2reg)
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.name = "Adam"
        self.count = 0

    def get_config(self):
        return (ctypes.c_int(4), (ctypes.c_float * 4)(self.learning_rate, self.beta1, self.beta2, self.epsilon), ctypes.c_int(4))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.m = []
        self.v = []
        for t in self.tensors:
            self.m.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))
            self.v.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))

    def renew_state(self, i):
        t = self.tensors[i]
        self.m[i] = ndarray.array(np.zeros(t.shape), t.ctx)
        self.v[i] = ndarray.array(np.zeros(t.shape), t.ctx)

    def process_gradient(self, node, grad, stream_handle=None):
        i = self.params.index(node)
        if self.count % len(self.params) == 0:
            self.beta1_t *= self.beta1
            self.beta2_t *= self.beta2
        self.count += 1
        weight = self.tensors[i]
        self.apply_l2_reg(node, grad, stream_handle)
        if ndarray.is_gpu_ctx(weight.ctx) and isinstance(grad, (ndarray.NDArray, ndarray.IndexedSlices)):
            gpu_op.adam_update(
                weight, grad, self.m[i], self.v[i],
                self.learning_rate, self.beta1, self.beta2,
                self.beta1_t, self.beta2_t, self.epsilon,
                True, stream_handle)
        else:
            raise NotImplementedError

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.tensors)
        assert params_size == len(grads)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        for i in range(params_size):
            if grads[i] == None:
                continue
            self.apply_l2_reg(self.params[i], grads[i], stream_handle)
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                assert isinstance(self.m[i], ndarray.NDArray)
                assert isinstance(self.v[i], ndarray.NDArray)
                gpu_op.adam_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.learning_rate, self.beta1,
                                   self.beta2, self.beta1_t, self.beta2_t, self.epsilon, False, stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    from ._base import DNNL_LIB
                    if DNNL_LIB['cpu_AdamOptimizerUpdate']:
                        from .cpu_links import adam_update as cpu_adam_update
                        cpu_adam_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.learning_rate, self.beta1,
                                        self.beta2, self.beta1_t, self.beta2_t, self.epsilon)
                    else:
                        prev_param = self.tensors[i].asnumpy()
                        grad = grads[i].asnumpy()
                        self.m[i][:] = self.beta1 * \
                            self.m[i].asnumpy() + (1 - self.beta1) * grad
                        self.v[i][:] = self.beta2 * self.v[i].asnumpy() + \
                            (1 - self.beta2) * grad * grad
                        mc = self.m[i].asnumpy() / (1 - self.beta1_t)
                        vc = self.v[i].asnumpy() / (1 - self.beta2_t)
                        self.tensors[i][:] = prev_param - \
                            self.learning_rate * mc / \
                            (np.sqrt(vc) + self.epsilon)

class AdamWScaledOptimizer(Optimizer):
    def __init__(self, num_data_parallel, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0):
        super(AdamWScaledOptimizer, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.name = "AdamWScaled"
        self.count = 0
        self.scale_rate = 1.0 / float(num_data_parallel)

    def get_config(self):
        return (ctypes.c_int(5), (ctypes.c_float * 5)(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay), ctypes.c_int(5))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.m = []
        self.v = []
        self.m2 = []
        for t in self.tensors:
            self.m.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))
            self.v.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))
            self.m2.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))

    def process_gradient(self, node, grad, stream_handle=None):
        i = self.params.index(node)
        if self.count % len(self.params) == 0:
            self.beta1_t *= self.beta1
            self.beta2_t *= self.beta2
        self.count += 1
        weight = self.tensors[i]
        if ndarray.is_gpu_ctx(weight.ctx) and isinstance(grad, (ndarray.NDArray, ndarray.IndexedSlices)):
            gpu_op.adamw_sacled_update(weight, grad, self.m[i], self.v[i], self.m2[i], self.learning_rate, self.beta1,
                self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.weight_decay, self.scale_rate, True, stream_handle)
        else:
            raise NotImplementedError

    def sync_state(self, comm, stream_handle):
        for arr in self.m2:
            if arr is None:
                continue
            comm.dlarrayNcclAllReduce(arr, arr, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclAvg, stream_handle)
        for arr in self.v:
            if arr is None:
                continue
            comm.dlarrayNcclAllReduce(arr, arr, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclAvg, stream_handle)

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.tensors)
        assert params_size == len(grads)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                assert isinstance(self.m[i], ndarray.NDArray)
                assert isinstance(self.v[i], ndarray.NDArray)
                gpu_op.adamw_sacled_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.m2[i], self.learning_rate, self.beta1,
                                   self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.weight_decay, self.scale_rate, False, stream_handle)
            else:
                raise NotImplementedError

class AdamWOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0):
        super(AdamWOptimizer, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.name = "AdamW"
        self.count = 0

    def get_config(self):
        return (ctypes.c_int(5), (ctypes.c_float * 5)(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay), ctypes.c_int(5))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.m = []
        self.v = []
        for t in self.tensors:
            self.m.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))
            self.v.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))

    def process_gradient(self, node, grad, stream_handle=None):
        i = self.params.index(node)
        if self.count % len(self.params) == 0:
            self.beta1_t *= self.beta1
            self.beta2_t *= self.beta2
        self.count += 1
        weight = self.tensors[i]
        if ndarray.is_gpu_ctx(weight.ctx) and isinstance(grad, (ndarray.NDArray, ndarray.IndexedSlices)):
            gpu_op.adamw_update(weight, grad, self.m[i], self.v[i], self.learning_rate, self.beta1,
                self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.weight_decay, True, stream_handle)
        else:
            raise NotImplementedError

    def sync_state(self, comm, stream_handle):
        for arr in self.m:
            if arr is None:
                continue
            comm.dlarrayNcclAllReduce(arr, arr, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclAvg, stream_handle)
        for arr in self.v:
            if arr is None:
                continue
            comm.dlarrayNcclAllReduce(arr, arr, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclAvg, stream_handle)

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.tensors)
        assert params_size == len(grads)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                assert isinstance(self.m[i], ndarray.NDArray)
                assert isinstance(self.v[i], ndarray.NDArray)
                gpu_op.adamw_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.learning_rate, self.beta1,
                                   self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.weight_decay, False, stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    prev_param = self.tensors[i].asnumpy()
                    grad = grads[i].asnumpy()
                    self.m[i][:] = self.beta1 * \
                        self.m[i].asnumpy() + (1 - self.beta1) * grad
                    self.v[i][:] = self.beta2 * self.v[i].asnumpy() + \
                        (1 - self.beta2) * grad * grad
                    mc = self.m[i].asnumpy() / (1 - self.beta1_t)
                    vc = self.v[i].asnumpy() / (1 - self.beta2_t)
                    update = mc / (np.sqrt(vc) + self.epsilon)
                    self.tensors[i][:] = prev_param - \
                        self.learning_rate * (update + self.weight_decay * self.tensors[i])

class LambOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0):
        super(AdamWOptimizer, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.name = "Lamb"

    def get_config(self):
        return (ctypes.c_int(5), (ctypes.c_float * 5)(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay), ctypes.c_int(5))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.m = []
        self.v = []
        for t in self.tensors:
            self.m.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))
            self.v.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.tensors)
        assert params_size == len(grads)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                assert isinstance(self.m[i], ndarray.NDArray)
                assert isinstance(self.v[i], ndarray.NDArray)
                gpu_op.lamb_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.learning_rate, self.beta1,
                                   self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.weight_decay, stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    prev_param = self.tensors[i].asnumpy()
                    grad = grads[i].asnumpy()
                    self.m[i][:] = self.beta1 * \
                        self.m[i].asnumpy() + (1 - self.beta1) * grad
                    self.v[i][:] = self.beta2 * self.v[i].asnumpy() + \
                        (1 - self.beta2) * grad * grad
                    mc = self.m[i].asnumpy() / (1 - self.beta1_t)
                    vc = self.v[i].asnumpy() / (1 - self.beta2_t)
                    update = mc / (np.sqrt(vc) + self.epsilon)
                    norm2_param = np.sqrt(np.sum(np.power(self.tensors[i], 2)))
                    norm2_update = np.sqrt(np.sum(np.power(update, 2)))
                    self.tensors[i][:] = prev_param - \
                        self.learning_rate * norm2_param / norm2_update * (update + self.weight_decay * self.tensors[i])
