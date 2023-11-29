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

        cc = load_dataset(
            "togethercomputer/RedPajama-Data-V2",
            name="default",
            partition="head_middle",
            snapshots=["2023-06", "2022-49"],
            languages=["en", "de", "it", "fr", "es"],
            split="train",
            streaming=True,
        ).map(self._raw_content_to_text)
        en_wiki = load_dataset(
            "graelo/wikipedia", "20230601.en", split="train", streaming=True
        ).shuffle()

        zh_wiki = load_dataset(
            "graelo/wikipedia", "20230601.zh", split="train", streaming=True
        ).shuffle()

        zh_cc = load_dataset(
            "uonlp/CulturaX", "zh", split="train", streaming=True
        ).shuffle()

        ja_wiki = load_dataset(
            "graelo/wikipedia", "20230601.ja", split="train", streaming=True
        ).shuffle()

        ja_cc = load_dataset(
            "uonlp/CulturaX", "ja", split="train", streaming=True
        ).shuffle()

        dataset = ds.interleave_datasets(
            [
                cc,
                en_wiki,
                zh_cc,
                zh_wiki,
                ja_cc,
                ja_wiki,
            ],
            [0.5, 0.2, 0.1, 0.1, 0.05, 0.05],
            stopping_strategy="all_exhausted",
        )
        return dataset

    def __iter__(self):
        for d in self.load_dataset():
            yield {"text": d["text"]}


class Jsonlines(IterableDataset):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __iter__(self):
        with jl.open(self.path) as file:
            for line in file:
                yield line


class Pile_(IterableDataset):
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
        self.data = dp.iter.IterableWrapper(Pile_(root)).shuffle().sharding_filter()

    def __iter__(self):
        for d in self.data:
            yield {"text": d["text"]}


class Sentence(IterableDataset):
    def __init__(
        self, dataset: IterableDataset, max_size: int, tokenizer: AutoTokenizer
    ):
        self.tokenizer = tokenizer
        # self.data = load_dataset("c4", "en", split="train", streaming=True).shuffle()

        self.dataset = dataset
        self.max_size = max_size

    def __iter__(self):
        # root = "/mnt/d/datasets/pixiv"
        text = ""
        for data in self.dataset:
            t = data["text"]
            text += t
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) <= self.max_size:
                continue
            yield {"text": self.tokenizer.convert_tokens_to_string(tokens)}
            text = ""
