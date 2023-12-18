from hetu import rgpu, gpu, cpu
import socket

device_local = [
    [gpu(0)],
    [gpu(1)],
    [gpu(2)],
    [gpu(3)],
]

device_8dp = [
    [gpu(0), gpu(1), gpu(2), gpu(3), gpu(4), gpu(5), gpu(6), gpu(7)],
] * 4

device_4dp = [
    [gpu(0), gpu(1), gpu(2), gpu(3)],
] * 4

def get_machine_name():
    name = socket.gethostname()
    i = int(name[1])
    if i % 2 == 1:
        i -= 1
    m0 = "w" + str(i)
    m1 = "w" + str(i+1)
    return m0, m1

m0, m1 = get_machine_name()

pipe_4_4 = [
    [rgpu(m0, 0), rgpu(m1, 0), rgpu(m0, 4), rgpu(m1, 4)],
    [rgpu(m0, 1), rgpu(m1, 1), rgpu(m0, 5), rgpu(m1, 5)],
    [rgpu(m0, 2), rgpu(m1, 2), rgpu(m0, 6), rgpu(m1, 6)],
    [rgpu(m0, 3), rgpu(m1, 3), rgpu(m0, 7), rgpu(m1, 7)],
]

pipe_2_4 = [
    [rgpu(m0, 0), rgpu(m1, 0)],
    [rgpu(m0, 1), rgpu(m1, 1)],
    [rgpu(m0, 2), rgpu(m1, 2)],
    [rgpu(m0, 3), rgpu(m1, 3)],
]

pipe_8_2 = [
    [rgpu(m0, 0), rgpu(m0, 2), rgpu(m0, 4), rgpu(m0, 6), rgpu(m1, 0), rgpu(m1, 2), rgpu(m1, 4), rgpu(m1, 6)],
    [rgpu(m0, 0), rgpu(m0, 2), rgpu(m0, 4), rgpu(m0, 6), rgpu(m1, 0), rgpu(m1, 2), rgpu(m1, 4), rgpu(m1, 6)],
    [rgpu(m0, 1), rgpu(m0, 3), rgpu(m0, 5), rgpu(m0, 7), rgpu(m1, 1), rgpu(m1, 3), rgpu(m1, 5), rgpu(m1, 7)],
    [rgpu(m0, 1), rgpu(m0, 3), rgpu(m0, 5), rgpu(m0, 7), rgpu(m1, 1), rgpu(m1, 3), rgpu(m1, 5), rgpu(m1, 7)],
]

pipe_8_4 = [
    [rgpu("w0", 0), rgpu("w0", 4), rgpu("w1", 0), rgpu("w1", 4), rgpu("w2", 0), rgpu("w2", 4), rgpu("w3", 0), rgpu("w3", 4)],
    [rgpu("w0", 1), rgpu("w0", 5), rgpu("w1", 1), rgpu("w1", 5), rgpu("w2", 1), rgpu("w2", 5), rgpu("w3", 1), rgpu("w3", 5)],
    [rgpu("w0", 2), rgpu("w0", 6), rgpu("w1", 2), rgpu("w1", 6), rgpu("w2", 2), rgpu("w2", 6), rgpu("w3", 2), rgpu("w3", 6)],
    [rgpu("w0", 3), rgpu("w0", 7), rgpu("w1", 3), rgpu("w1", 7), rgpu("w2", 3), rgpu("w2", 7), rgpu("w3", 3), rgpu("w3", 7)],
]

pipe_16_4 = [
    [rgpu("w0", 0), rgpu("w0", 4), rgpu("w1", 0), rgpu("w1", 4), rgpu("w2", 0), rgpu("w2", 4), rgpu("w3", 0), rgpu("w3", 4), rgpu("w4", 0), rgpu("w4", 4), rgpu("w5", 0), rgpu("w5", 4), rgpu("w6", 0), rgpu("w6", 4), rgpu("w7", 0), rgpu("w7", 4)],
    [rgpu("w0", 1), rgpu("w0", 5), rgpu("w1", 1), rgpu("w1", 5), rgpu("w2", 1), rgpu("w2", 5), rgpu("w3", 1), rgpu("w3", 5), rgpu("w4", 1), rgpu("w4", 5), rgpu("w5", 1), rgpu("w5", 5), rgpu("w6", 1), rgpu("w6", 5), rgpu("w7", 1), rgpu("w7", 5)],
    [rgpu("w0", 2), rgpu("w0", 6), rgpu("w1", 2), rgpu("w1", 6), rgpu("w2", 2), rgpu("w2", 6), rgpu("w3", 2), rgpu("w3", 6), rgpu("w4", 2), rgpu("w4", 6), rgpu("w5", 2), rgpu("w5", 6), rgpu("w6", 2), rgpu("w6", 6), rgpu("w7", 2), rgpu("w7", 6)],
    [rgpu("w0", 3), rgpu("w0", 7), rgpu("w1", 3), rgpu("w1", 7), rgpu("w2", 3), rgpu("w2", 7), rgpu("w3", 3), rgpu("w3", 7), rgpu("w4", 3), rgpu("w4", 7), rgpu("w5", 3), rgpu("w5", 7), rgpu("w6", 3), rgpu("w6", 7), rgpu("w7", 3), rgpu("w7", 7)],
]

pipe_16_1 = [
    [rgpu(m0, 0), rgpu(m0, 1), rgpu(m0, 2), rgpu(m0, 3), rgpu(m0, 4), rgpu(m0, 5), rgpu(m0, 6), rgpu(m0, 7),
     rgpu(m1, 0), rgpu(m1, 1), rgpu(m1, 2), rgpu(m1, 3), rgpu(m1, 4), rgpu(m1, 5), rgpu(m1, 6), rgpu(m1, 7)]
] * 4

single = [
    [gpu(0)]
] * 4

my_device = pipe_4_4

def add_cpu_ctx(device_list):
    result = []
    for i in range(len(device_list)):
        result.append(device_list[i] + [cpu(0)])
    return result
