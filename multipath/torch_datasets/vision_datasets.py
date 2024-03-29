import asyncio
import glob
import json
import os
import random
from typing import Iterable

import aiofiles
import httpx
import jsonlines as jl
import torchvision
from datasets import load_dataset
from sqlmodel import Field, Session, SQLModel, create_engine, select
from torch.utils.data import IterableDataset, IterDataPipe
from transformers import AutoTokenizer
import torch.utils.data.datapipes as dp
import datasets as ds


class Video(SQLModel, table=True):
    id: int = Field(primary_key=True, default=None)
    url: str = Field(default=None, index=True)
    meta: bytes = Field(default=None)


class VideoStream(Iterable):
    def __init__(self, path: str, step_size: float = 1.0):
        self.path = path
        self.step_size = step_size
        self.metadata = torchvision.io.VideoReader(path, "video").get_metadata()

    def __iter__(self):
        reader = torchvision.io.VideoReader(self.path, "video")
        engine = create_engine("sqlite:///video.db")
        Video.metadata.create_all(engine)

        duration = self.metadata["video"]["duration"][0]
        for i in range(int(duration / self.step_size)):
            t = i * self.step_size
            reader.seek(t)
            reader.set_current_stream("video")
            vf = next(reader)
            reader.seek(t)
            reader.set_current_stream("audio")
            af = next(reader)
            yield {"video": vf, "audio": af, "metadata": self.metadata, "time": t}


class BilibiliVideoStreams(Iterable):
    def __init__(self, db_path: str, cache_dir: str, step_size: float = 1.0):
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.step_size = step_size

    def get_video_formats(self, info: dict):
        formats = info["formats"]
        vfs = []
        for f in formats:
            if f["resolution"] != "audio only":
                vfs.append(f)
        return vfs

    def get_audio_formats(self, info: dict):
        formats = info["formats"]
        vfs = []
        for f in formats:
            if f["resolution"] == "audio only":
                vfs.append(f)
        return vfs

    async def download(client: httpx.AsyncClient, url: str, dst: str):
        async with client.stream("GET", url) as stream:
            stream.raise_for_status()
            async with aiofiles.open(dst, "wb") as f:
                async for chunk in stream.iter_bytes():
                    await f.write(chunk)
        return dst

    async def __aiter__(self):
        engine = create_engine(self.db_path, echo=False)
        stmt = select(Video.url, Video.meta)
        async with httpx.AsyncClient() as client:
            with Session(engine) as session:
                while (video := session.exec(stmt).fetchone()) is not None:
                    meta = json.loads(video.meta)
                    vfs = self.get_video_formats(meta)
                    afs = self.get_audio_formats(meta)
                    video_path = None
                    audio_path = None
                    for vf in vfs:
                        try:
                            title = meta["title"]
                            video_path = await self.download(
                                client,
                                vf["url"],
                                os.path.join(self.cache_dir, f"{title}.mp4"),
                            )

                            break
                        except Exception as e:
                            print(e)
                    if video_path is None:
                        continue
                    for af in afs:
                        try:
                            title = meta["title"]
                            audio_path = await self.download(
                                client,
                                af["url"],
                                os.path.join(self.cache_dir, f"{title}.mp3"),
                            )
                            break
                        except Exception as e:
                            print(e)
                    if audio_path is None:
                        continue

                    video_reader = torchvision.io.VideoReader(video_path, "video")
                    audio_reader = torchvision.io.VideoReader(audio_path, "audio")
                    duration = video_reader.get_metadata()["video"]["duration"][0]
                    for i in range(int(duration / self.step_size)):
                        t = i * self.step_size
                        video_reader.seek(t)
                        audio_reader.seek(t)
                        video_frame = next(video_reader)
                        audio_frame = next(audio_reader)
                        yield {
                            "video": video_frame,
                            "audio": audio_frame,
                            "time": t,
                            meta: meta,
                        }

    def __iter__(self):
        loop = asyncio.get_event_loop()
        aiter = self.__aiter__()

        while True:
            try:
                data = loop.run_until_complete(asyncio.wait_for(aiter.__anext__()))
                yield data.result()
            except StopAsyncIteration:
                return
            except Exception:
                pass
