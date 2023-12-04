import glob
import os
import random

import jsonlines as jl
from datasets import load_dataset
from torch.utils.data import IterableDataset, IterDataPipe
from transformers import AutoTokenizer
import torch.utils.data.datapipes as dp
import datasets as ds
import re


class _WebData(IterableDataset):
    @staticmethod
    def _raw_content_to_text(data: dict):
        data["text"] = data["raw_content"]
        return data

    def __init__(self, **kwargs):
        super().__init__()

    def load_dataset(self):
        # info = get_worker_info()

        cc = load_dataset(
            "togethercomputer/RedPajama-Data-V2",
            snapshots=["2023-14"],
            languages=["en"],
            name="default",
            streaming=True,
            split="train",
        ).map(self._raw_content_to_text)

        return cc

    def __len__(self):
        return 500000000

    def __iter__(self):
        for d in self.load_dataset():
            yield {"text": d["text"]}


class WebData(IterDataPipe):
    def __init__(self, **kwargs):
        super().__init__()
        self.data = (
            dp.iter.IterableWrapper(_WebData(**kwargs)).shuffle().sharding_filter()
        )

    def __iter__(self):
        for d in self.data:
            yield d


class Jsonlines(IterDataPipe):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __iter__(self):
        with jl.open(self.path) as file:
            for line in file:
                yield line


class _Pile(IterDataPipe):
    def __init__(self, root: str):
        super().__init__()
        self.root = root
        files = glob.glob(os.path.join(self.root, "*.jsonl"))
        self.data = dp.iter.Multiplexer(
            *[
                dp.iter.IterableWrapper(Jsonlines(f).shuffle().sharding_filter())
                for f in files
            ]
        ).shuffle()

    def __iter__(self):
        for d in self.data:
            yield d


class Pile(IterDataPipe):
    def __init__(self, root: str):
        super().__init__()
        self.data = _Pile(root)

    def __iter__(self):
        for d in self.data:
            yield {"text": d["text"]}


class Sentence(IterDataPipe):
    def __init__(self, dataset: IterDataPipe, max_size: int, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

        self.dataset = dataset
        self.max_size = max_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        text = ""
        p = re.compile(r"\n+")
        for data in self.dataset:
            t = data["text"]
            text += " " + t.strip()
            # remove repeat newlines
            text = p.sub("\n", text)
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) <= self.max_size:
                continue
            yield {"text": text}
            text = ""
