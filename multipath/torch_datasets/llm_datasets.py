import glob
import os
import random

import jsonlines as jl
from datasets import load_dataset
from torch.utils.data import IterableDataset, IterDataPipe
from transformers import AutoTokenizer
import torch.utils.data.datapipes as dp
import datasets as ds


class WebData(IterableDataset):
    @staticmethod
    def _raw_content_to_text(data: dict):
        data["text"] = data["raw_content"]
        return data

    def __init__(self, **kwargs):
        super().__init__()

    def load_dataset(self):
        # info = get_worker_info()

        cc = (
            load_dataset(
                "togethercomputer/RedPajama-Data-V2",
                name="default",
                partition="head_middle",
                snapshots=["2023-06"],
                languages=["en"],
                split="train",
                streaming=True,
            )
            .map(self._raw_content_to_text)
            .shuffle(buffer_size=10000)
        )

        return cc

    def __len__(self):
        return 500000000

    def __iter__(self):
        for d in self.load_dataset():
            yield {"text": d["text"]}


class Jsonlines(IterDataPipe):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __iter__(self):
        with jl.open(self.path) as file:
            for line in file:
                yield line


class Pile_(IterDataPipe):
    def __init__(self, root: str):
        super().__init__()
        self.root = root
        files = glob.glob(os.path.join(self.root, "*.jsonl"))
        self.data = dp.iter.Multiplexer(
            *[
                dp.iter.IterableWrapper(Jsonlines(f)).shuffle().sharding_filter()
                for f in files
            ]
        ).shuffle()

    def __iter__(self):
        for d in self.data:
            yield d


class Pile(IterDataPipe):
    def __init__(self, root: str):
        super().__init__()
        self.data = Pile_(root)

    def __iter__(self):
        for d in self.data:
            yield {"text": d["text"]}


class Sentence(IterDataPipe):
    def __init__(self, dataset: IterDataPipe, max_size: int, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        # self.data = load_dataset("c4", "en", split="train", streaming=True).shuffle()

        self.dataset = dataset
        self.max_size = max_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # root = "/mnt/d/datasets/pixiv"
        text = ""
        for data in self.dataset:
            t = data["text"]
            text += " " + t.strip()
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) <= self.max_size:
                continue
            yield {"text": text}
            text = ""
