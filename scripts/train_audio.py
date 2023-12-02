import os
from torch.utils.data import DataLoader, IterableDataset
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import glob
from datasets import load_dataset
from multipath.nn.audio import Autoencoder1d, Discriminator
import wandb
import scipy.io.wavfile as wav


class AudioCorpus(IterableDataset):
    def __init__(self, length: int = 65536):
        self.length = length
        self.datasets = load_dataset("librispeech_asr", "all", split="train.other.500")

    def __iter__(self):
        for data in self.datasets:
            array = torch.from_numpy(data["audio"]["array"])[: self.length]
            if array.shape[0] < self.length:
                array = F.pad(array, (0, self.length - array.shape[0]), mode="constant")
            array = array.view(1, -1)
            sample_rate = data["audio"]["sampling_rate"]
            text = data["text"]
            yield {
                "data": array,
                "sample_rate": sample_rate,
                "text": text,
            }


def collate_fn(batch):
    data = [b["data"] for b in batch]
    text = [b["text"] for b in batch]
    sample_rate = [b["sample_rate"] for b in batch]
    return {
        "data": torch.stack(data),
        "text": text,
        "sample_rate": sample_rate,
    }


def train(
    batch_size: int = 32,
    lr: float = 1e-4,
    save_every: int = 1000,
    model_config: dict = {},
):
    wandb.init(
        name="Multi-Path-Transformer",
        project="Multi-Path-Transformer",
        id="vision-encoder",
    )
    dataset = AudioCorpus()
    dl = DataLoader(
        dataset, batch_size=batch_size, drop_last=True, collate_fn=collate_fn
    )
    model = Autoencoder1d(**model_config).to("cuda").to(torch.bfloat16)
    # D = Discriminator(**model_config).to("cuda").to(torch.bfloat16)
    try:
        ckpts = glob.glob("models/audio*.pt")
        ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[-1].split(".")[0]))
        model.load_state_dict(torch.load(ckpts[-1]))
    except Exception as e:
        print(e)
        print("Starting from scratch")

    opt = torch.optim.AdamW(
        [{"params": model.parameters()}],
        lr=lr,
        fused=True,
        weight_decay=0,
    )

    idx = 0
    for i, batch in enumerate(pbar := tqdm(dl)):
        inputs = batch["data"].to("cuda").to(torch.bfloat16)
        outputs = model(inputs)
        loss = F.mse_loss(outputs, inputs, reduction="sum")

        pbar.set_description(f"{idx} loss {loss:.4f}")
        loss.backward()
        opt.step()
        opt.zero_grad()
        wandb.log({"loss": loss}, i)
        os.makedirs("models", exist_ok=True)
        if i % save_every == 0:
            gt_audio = inputs.detach()[0].float().cpu().numpy()
            pred_audio = outputs.detach()[0].float().cpu().numpy()
            gt_audio = gt_audio.flatten()
            wav.write("gt.wav", int(batch["sample_rate"][0]), gt_audio)
            pred_audio = pred_audio.flatten()
            wav.write("pred.wav", int(batch["sample_rate"][0]), pred_audio)
            wandb.log(
                {
                    "wav": [
                        wandb.Audio(
                            "gt.wav",
                            sample_rate=int(batch["sample_rate"][0]),
                            caption=batch["text"][0],
                        ),
                        wandb.Audio(
                            "pred.wav",
                            sample_rate=int(batch["sample_rate"][0]),
                            caption=batch["text"][0],
                        ),
                    ]
                },
                i,
            )

            ckpts = glob.glob("models/audio*.pt")
            if len(ckpts) > 3:
                for ckpt in sorted(ckpts)[: len(ckpts) - 3]:
                    os.remove(ckpt)
            torch.save(model.state_dict(), f"models/audio-{i}.pt")


if __name__ == "__main__":
    import yaml

    model_configs = yaml.full_load(open("configs/model_configs.yml"))
    train(model_config=model_configs["audio_model"])
