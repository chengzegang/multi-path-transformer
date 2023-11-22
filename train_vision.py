import os
from torch.utils.data import DataLoader, IterableDataset
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


class ImageCorpus(IterableDataset):
    def __init__(self, root: str, image_size: int = 256):
        self.image_size = image_size
        self.root = root

    def __iter__(self):
        for f in glob.iglob(self.root + "**/*.JPEG", recursive=True):
            img = Image.open(f).convert("RGB")
            img = TF.resize(img, self.image_size, antialias=True)
            img = TF.center_crop(img, self.image_size)
            img = TF.pil_to_tensor(img)
            img = TF.to_dtype(img, dtype=torch.float32, scale=True)
            inputs = TF.normalize(
                img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            yield inputs, img


def exponential_decay(decay_rate: float, min_factor: float, step: int) -> float:
    return max(min_factor, decay_rate**step)


def train(
    root: str,
    batch_size: int = 32,
    lr: float = 1e-4,
    save_every: int = 200,
    model_config: dict = {},
):
    torch.set_num_interop_threads(16)
    torch.set_num_threads(16)
    wandb.init(
        name="Multi-Path-Transformer",
        project="Multi-Path-Transformer",
        id="vision-encoder",
    )
    dataset = ImageCorpus(root)
    dl = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=16)
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
    sched = LambdaLR(opt, lr_lambda=lambda x: exponential_decay(0.99996, 0.01, x))
    idx = 0
    grad_accum = 1
    proxy_model = torch.compile(
        model, fullgraph=True, dynamic=False, mode="max-autotune"
    )
    for i, (inputs, targets) in enumerate(pbar := tqdm(dl)):
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
        outputs = proxy_model(inputs)
        loss = F.l1_loss(outputs, targets)
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
    train("/mnt/f/datasets/imagenet/", model_config=model_config)
