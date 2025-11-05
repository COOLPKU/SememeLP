# Copyright (c) Facebook, Inc. and its affiliates.
"""
List serialization code adopted from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py
"""

from typing import List, Any, Optional
import multiprocessing as mp
import pickle
import numpy as np
import torch
import serialize_comm as comm

def serialize_np(data):
    buffer = pickle.dumps(data, protocol=-1)
    return np.frombuffer(buffer, dtype=np.uint8)
def unserialize_np(data):
    data = memoryview(data)
    return pickle.loads(data)
def serialize_tensor(data):
    buffer = pickle.dumps(data, protocol=-1)
    return torch.tensor(np.frombuffer(buffer, dtype=np.uint8))
def unserialize_tensor(data):
    data = memoryview(data.numpy())
    return pickle.loads(data)


class NumpySerializedList():
    def __init__(self, lst: list, need_pickle=True):
        def _serialize(data):
            if need_pickle:
                buffer = pickle.dumps(data, protocol=-1)
                data =  np.frombuffer(buffer, dtype=np.uint8)
            elif type(data) is np.ndarray:
                pass
            elif type(data) is list:
                data = np.array(data)
            else:
                raise ValueError(f"{type(data)}")
            return data

        print(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(lst)
            )
        )

        self.need_pickle = need_pickle
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._lst = np.concatenate(self._lst)
        self._addr = np.cumsum(self._addr)


        print("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        if self.need_pickle:
            bytes = memoryview(self._lst[start_addr:end_addr])
            data =  pickle.loads(bytes)
        else:
            data = self._lst[start_addr:end_addr]
        return data



class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list, need_pickle=True):
        super().__init__(lst, need_pickle)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        if self.need_pickle:
            bytes = memoryview(self._lst[start_addr:end_addr].numpy())
            data =  pickle.loads(bytes)
        else:
            data = self._lst[start_addr:end_addr]
        return data



def local_scatter(array: Optional[List[Any]]):
    """
    Scatter an array from local leader to all local workers.
    The i-th local worker gets array[i].

    Args:
        array: Array with same size of #local workers.
    """
    if comm.get_local_size() == 1:
        # Just one worker. Do nothing.
        return array[0]
    if comm.get_local_rank() == 0:
        assert len(array) == comm.get_local_size()
        comm.all_gather(array)
    else:
        all_data = comm.all_gather(None)
        array = all_data[comm.get_rank() - comm.get_local_rank()]
    return array[comm.get_local_rank()]


# NOTE: https://github.com/facebookresearch/mobile-vision/pull/120
# has another implementation that does not use tensors.
class TorchShmSerializedList(TorchSerializedList):
    def __init__(self, lst: list, need_pickle = True):
        if comm.get_local_rank() == 0:
            super().__init__(lst, need_pickle)
        if comm.get_local_rank() == 0:
            # Move data to shared memory, obtain a handle to send to each local worker.
            # This is cheap because a tensor will only be moved to shared memory once.
            handles = [None] + [
              bytes(mp.reduction.ForkingPickler.dumps((self._addr, self._lst)))
              for _ in range(comm.get_local_size() - 1)]
        else:
            handles = None
        # Each worker receives the handle from local leader.
        handle = local_scatter(handles)

        if comm.get_local_rank() > 0:
            # Materialize the tensor from shared memory.
            self._addr, self._lst = mp.reduction.ForkingPickler.loads(handle)
            print(f"Worker {comm.get_rank()} obtains a dataset of length="
                  f"{len(self)} from its local leader.")
    
    def moved_to_shared_memory(self):
        if comm.get_local_rank() == 0:
            # Move data to shared memory, obtain a handle to send to each local worker.
            # This is cheap because a tensor will only be moved to shared memory once.
            handles = [None] + [
              bytes(mp.reduction.ForkingPickler.dumps((self._addr, self._lst)))
              for _ in range(comm.get_local_size() - 1)]
        else:
            handles = None
        # Each worker receives the handle from local leader.
        handle = local_scatter(handles)

        if comm.get_local_rank() > 0:
            # Materialize the tensor from shared memory.
            self._addr, self._lst = mp.reduction.ForkingPickler.loads(handle)
            print(f"Worker {comm.get_rank()} obtains a dataset of length="
                  f"{len(self)} from its local leader.")