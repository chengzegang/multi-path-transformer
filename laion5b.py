from typing import Callable, Tuple
from torch.utils.data import IterDataPipe
import webdataset as wds
import glob
from PIL import Image
from torch import Tensor
import torchvision.transforms.v2.functional as TF
import torch
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Laion5B(IterDataPipe):
    def __init__(self, root: str, **kwargs):
        self.root = root
        self.data = wds.WebDataset(
            glob.glob(os.path.join(root, "*.tar")),
            wds.ignore_and_continue,
            shardshuffle=True,
        ).shuffle(10000).decode("pil")

    def __len__(self) -> int:
        return 200000000

    def __iter__(self):
        for data in self.data:
            yield data['jpg']

if __name__ == "__main__":
    data = Laion5B('/scratch/work/public/ml-datasets/laion2B-en-data/')
    print(next(iter(data)))