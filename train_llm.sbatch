#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=168:00:00
#SBATCH --mem=128GB 
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=llm
#SBATCH --output=llm.out

# To use torch.compile
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib:/usr/lib64:/usr/lib32:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.2

singularity exec --nv \
	    --overlay /scratch/$USER/containers/overlay.ext3:ro  \
	    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/zc2309/multi-path-transformer/; pip install evaluate; python /scratch/zc2309/multi-path-transformer/train_llm.py"

