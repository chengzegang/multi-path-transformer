from functools import partial
import os
from torch.utils.data import DataLoader, IterDataPipe
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import glob
from PIL import Image
import torchvision.transforms.v2.functional as TF
import matplotlib.pyplot as plt
from vision import Autoencoder2d
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import datapipes as dp
from torch import Tensor
from laion5b import Laion5B
import warnings
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


def load_image(img: Image.Image, size: int = 256, target_size: int = 512) -> Tensor:
    target = TF.resize(img, target_size, antialias=True)
    target = TF.center_crop(target, target_size)
    target = TF.pil_to_tensor(target).clone()
    target = TF.to_dtype(target, dtype=torch.float32, scale=True)

    inputs = TF.resize(img, size, antialias=True)
    inputs = TF.center_crop(inputs, size)
    inputs = TF.pil_to_tensor(inputs)
    inputs = TF.to_dtype(inputs, dtype=torch.float32, scale=True).clone()
    inputs = TF.normalize(inputs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return inputs, target


def exponential_decay(decay_rate: float, min_factor: float, step: int) -> float:
    return max(min_factor, decay_rate**step)


def is_ext(path: str, ext: str) -> bool:
    return path.lower().endswith(ext)


def get_dataset(name: str, root: str, ext: str | None = "jpeg") -> IterDataPipe:
    if name == "laion5b":
        return dp.iter.IterableWrapper(Laion5B(root)).sharding_filter()
    elif name == "folder":
        return (
            dp.iter.FileLister(root, recursive=True)
            .filter(partial(is_ext, ext=ext))
            .shuffle()
            .sharding_filter()
            .map(lambda x: Image.open(x).convert("RGB"))
        )


def collate_fn(batch: list[Tensor]) -> Tensor:
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets


def train(
    dataset_name: str,
    root: str,
    batch_size: int = 32,
    lr: float = 1e-4,
    save_every: int = 200,
    model_config: dict = {},
):
    torch.set_num_interop_threads(16)
    torch.set_num_threads(16)
    os.makedirs("models", exist_ok=True)
    wandb.init(
        name="Multi-Path-Transformer-Vision-Encoder",
        project="Multi-Path-Transformer",
        id="vision-encoder",
    )
    dataset = get_dataset(dataset_name, root).map(
        partial(load_image, size=256, target_size=512)
    )
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=16,
        collate_fn=collate_fn,
    )
    model = Autoencoder2d(**model_config)
    try:
        ckpts = glob.glob("models/vison*.pt")
        ckpt = sorted(ckpts, key=lambda x: int(x.split("-")[-1].split(".")[0]))[-1]
        model.load_state_dict(torch.load(ckpt))
    except Exception as e:
        print(e)
        print("Starting from scratch")
    model = model.to("cuda").to(torch.bfloat16).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True, weight_decay=0)
    sched = LambdaLR(opt, lr_lambda=lambda x: exponential_decay(0.9996, 0.01, x))
    idx = 0
    grad_accum = 1
    # proxy_model = torch.compile(
    #    model, fullgraph=True, dynamic=False, mode="max-autotune"
    # )
    for i, (inputs, targets) in enumerate(pbar := tqdm(dl)):
        grad_accum = max(64, i // 1000 + 1)
        targets = (
            targets.to("cuda")
            .to(torch.bfloat16)
            .contiguous(memory_format=torch.channels_last)
        )
        inputs = (
            inputs.to("cuda")
            .to(torch.bfloat16)
            .contiguous(memory_format=torch.channels_last)
        )
        outputs = model(inputs)
        up_scaled_outputs = F.interpolate(
            outputs, size=512, mode="nearest", align_corners=False, antialias=True
        )
        loss = F.l1_loss(up_scaled_outputs, targets)
        pbar.set_description(f"{idx} loss {loss:.4f} lr {sched.get_last_lr()[0]:.4e}")
        loss.backward()
        if i % grad_accum == 0:
            opt.step()
            sched.step()
            opt.zero_grad()
        wandb.log({"loss": loss, "lr": sched.get_last_lr()[0]}, i)
        if i % save_every == 0:
            fig = plt.figure()
            ax = plt.subplot(1, 2, 1)
            ax.imshow(TF.to_pil_image(outputs[0].clamp(0, 1)))
            ax = plt.subplot(1, 2, 2)
            ax.imshow(TF.to_pil_image(targets[0]))
            fig.savefig("output.png")
            wandb.log({"example": fig}, i)
            plt.close(fig)

            ckpts = glob.glob("models/vison*.pt")
            if len(ckpts) > 3:
                os.remove(sorted(ckpts)[0])
            torch.save(model.state_dict(), f"models/vison-{i}.pt")


if __name__ == "__main__":
    model_config = {
        "base_channels": 64,
        "num_layers": 4,
        "latent_size": 8,
    }

    local_config = {
        "dataset_name": "folder",
        "root": "/mnt/f/datasets/imagenet/",
        "batch_size": 32,
    }

    greene_config = {
        "dataset_name": "laion5b",
        "root": "/scratch/work/public/ml-datasets/laion2B-en-data/",
        "batch_size": 256,
    }

    train(
        **local_config,
        model_config=model_config,
    )
