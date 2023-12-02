conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --solver=libmamba -y
conda install transformers datasets pyarrow pandas evaluate accelerate sentencepiece wandb pynvml wandb typer jsonlines matplotlib -c conda-forge --solver=libmamba -y
git config --add user.email "20091803+chengzegang@users.noreply.github.com"
git config --add user.name "Zegang Cheng"