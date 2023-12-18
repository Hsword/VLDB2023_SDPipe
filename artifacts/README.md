# SD-PIPE

## Installation
1. Clone the repository.

2. Prepare the environment. We use Anaconda to manage packages. The following command create the conda environment to be used:`conda env create -f environment.yml`. The environment requires Cuda toolkit version 10.2.

3. We use CMake to compile the system. Please copy the example configuration for compilation by `cp cmake/config.example.cmake cmake/config.cmake`. If your system is using another version of NCCL/cudnn/MPI, you should manually change the path defined in this config file.

```bash
# compile
# make all
mkdir build && cd build && cmake ..
make -j
```

4. Run `conda activate pipe` and prepare PYTHONPATH by executing the command `source hetu.exp` at the root folder.

5. Run python and try `import hetu` to check whether the compilation is successful.

## Prepare datasets
The imagenet dataset is used for resnet50 model training. It can be downloaded at [https://www.image-net.org/download.php](https://www.image-net.org/download.php). The downloaded dataset is in pytorch image folder format. It has a TRAIN folder and a EVAL folder, each folder has 1000 sub folders (the 1000 classes) in them. Please set the `IMAGENET_TRAIN_ROOT` and `IMAGENET_VAL_ROOT` variable in the pipeline/datasets/imagenet.py to your imagenet TRAIN and EVAL folder.

We use the wikipedia dataset for bert pretraining. We use the dataset processing method used in [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/ddbcd54056e8d1bc1c4d5a8ab34cb570ebea1947/PyTorch/LanguageModeling/BERT) (commit ddbcd5). In our experiments, we used wiki (not wiki+bookcorpus) and use 128 as the sequence length. Please follow their steps to generate the dataset. You will finally get some hdf5 training files (we don't need the eval files). Put the hdf5 files in a folder and set the directory variable in pipeline/models/wiki.py to point to the folder where hdf5 files are stored.

## Config distributed training
Our system can start multiple pipelines on different machines. If you have two machine named `w0` and `w1`, each machine has 8 GPUs, then you should write a configuration file like the one in pipeline/config/w0_w1.yml. It tells how many worker process and server process will be launched during the training process.

You will also need to specify how the pipeline is placed on these GPUs. We provide how you can do this in pipeline/device_list.py.
```python
pipe_4_4 = [
    [rgpu("w0", 0), rgpu("w0", 4), rgpu("w1", 0), rgpu("w1", 4)],
    [rgpu("w0", 1), rgpu("w0", 5), rgpu("w1", 1), rgpu("w1", 5)],
    [rgpu("w0", 2), rgpu("w0", 6), rgpu("w1", 2), rgpu("w1", 6)],
    [rgpu("w0", 3), rgpu("w0", 7), rgpu("w1", 3), rgpu("w1", 7)],
]
my_device = pipe_4_4 # exported
```
In this case there are 4 pipelines in the 2 machine with 16 GPUs, 0-1-2-3 GPU in machine `w0` belongs to one pipeline.

## Run pipelined data parallel
You will need to make sure that both MPI and NCCL is working in your cluster.
You can enter the pipeline folder and then runs the command to get the full results.
```bash
# resnet50 training
heturun -c config/w0_w1.yml \
python3 train_resnet.py \
--batch-size 64 --model resnet50 --dataset imagenet --learning-rate 0.01 --weight-decay 1e-4 --epochs 100 \
--pipeline pipedream \
--preduce \
```

```bash
# bert training
heturun -c config/w0_w1.yml \
python3 train_bert.py \
--batch-size 64 --learning-rate 1e-5 --epochs 5000 \
--pipeline pipedream \
--preduce \
```

Arguments explanation: You can remove the `--preduce` flag to get the Allreduce results. You can add `--adpsgd` to get the time by appling ADPSGD algorithm. You can use the `--pipeline hetpipe` to get the baseline for HetPipe. You can add  `--hetero 0.1` flag to mimic the heterogeneous environments.
