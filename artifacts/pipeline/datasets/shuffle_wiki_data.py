import h5py
import os.path as osp
import numpy as np
prefix = "~/.cache/hetu/datasets/wikicorpus_en/"
prefix = osp.expanduser(prefix)

fields = ["input_ids", "segment_ids", "masked_lm_positions", "masked_lm_ids", "next_sentence_labels", "input_mask"]
all_data = {}
for key in fields:
    all_data[key] = []
for i in range(256):
    print("loading data ", i)
    fname = osp.join(prefix, "wikicorpus_en_train_{}.hdf5".format(i))
    f = h5py.File(fname)
    for key in fields:
        data = f[key][:]
        all_data[key].append(data)
    f.close()

for key in fields:
    all_data[key] = np.concatenate(all_data[key])
    num_data = len(all_data[key])
idx = np.arange(num_data)
np.random.shuffle(idx)
for key in fields:
    all_data[key] = all_data[key][idx]
split = np.linspace(0, num_data, 256 + 1).astype(int)
print(split.shape)
for i in range(256):
  print("saving data ", i)
  output_file = "wikicorpus_en_train_{}.hdf5".format(i)
  f= h5py.File(output_file, 'w')
  f.create_dataset("input_ids", data=all_data["input_ids"][split[i]:split[i+1]], dtype='i4', compression='gzip')
  f.create_dataset("input_mask", data=all_data["input_mask"][split[i]:split[i+1]], dtype='i1', compression='gzip')
  f.create_dataset("segment_ids", data=all_data["segment_ids"][split[i]:split[i+1]], dtype='i1', compression='gzip')
  f.create_dataset("masked_lm_positions", data=all_data["masked_lm_positions"][split[i]:split[i+1]], dtype='i4', compression='gzip')
  f.create_dataset("masked_lm_ids", data=all_data["masked_lm_ids"][split[i]:split[i+1]], dtype='i4', compression='gzip')
  f.create_dataset("next_sentence_labels", data=all_data["next_sentence_labels"][split[i]:split[i+1]], dtype='i1', compression='gzip')
  f.flush()
  f.close()
