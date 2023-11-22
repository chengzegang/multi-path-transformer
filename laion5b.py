from typing import Callable, Tuple
from torch.utils.data import IterDataPipe
import webdataset as wds
import glob
from PIL import Image
from torch import Tensor
import torchvision.transforms.v2.functional as TF
import torch
import os


class Laion5B(IterDataPipe):
    def __init__(self, root: str, **kwargs):
        self.root = root
        self.data = wds.WebDataset(
            glob.glob(os.path.join(root, "*.tar")),
            wds.ignore_and_continue,
            shardshuffle=True,
        ).shuffle(10000)

    def __len__(self) -> int:
        return 200000000

    def __iter__(self):
        for data in self.data:
            yield data
