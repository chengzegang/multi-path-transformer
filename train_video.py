from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import glob
import torchvision.transforms.v2.functional as TF
from mhvit import MHVisionTransformer
import matplotlib.pyplot as plt
from vision_datasets import BilibiliVideoStreams


def train(db_path: str, cache_dir: str):
    dataset = BilibiliVideoStreams(db_path, cache_dir)
    dl = DataLoader(dataset, batch_size=8, drop_last=True)
    model = MHVisionTransformer(128, 1024, 16, 12, 128).to("cuda").to(torch.bfloat16)
    try:
        ckpts = glob.glob("models/vision*.pt")
        ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[-1].split(".")[0]))
        model.load_state_dict(torch.load(ckpts[-1]))
    except Exception as e:
        print(e)
        print("Starting from scratch")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    idx = 0
    for i, (inputs, targets) in enumerate(pbar := tqdm(dl)):
        inputs = inputs.to("cuda").to(torch.bfloat16)
        targets = targets.to("cuda").to(torch.bfloat16)
        outputs = model(inputs)
        loss = F.l1_loss(outputs, targets)
        pbar.set_description(f"{idx} loss {loss:.4f} ")
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i % 100 == 0:
            fig = plt.figure()
            ax = plt.subplot(1, 2, 1)
            ax.imshow(TF.to_pil_image(outputs[0]))
            ax = plt.subplot(1, 2, 2)
            ax.imshow(TF.to_pil_image(targets[0]))
            fig.savefig("output.png")
            plt.close(fig)


if __name__ == "__main__":
    train("/mnt/f/datasets/imagenet/")
