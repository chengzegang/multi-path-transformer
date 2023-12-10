#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=168:00:00
#SBATCH --mem=490GB 
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=llm4
#SBATCH --output=llm4.out

# To use torch.compile

export PATH=/usr/local/cuda-12/bin:/usr/local/cuda-12/lib64:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:/usr/lib:/usr/lib64:/usr/lib32:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12
export TORCH_HOME=/scratch/zc2309/multi-path-transformer/torch
export MPLCONFIGDIR=/scratch/zc2309/.cache/matplotlib


name=$(echo $1 | cut -d '-' -f1 -)
node1=$(echo $1 | cut -d '-' -f2 - | tr -d '[')

export NUM_WORKERS=$SLURM_JOB_CPUS_PER_NODE
#export MASTER_ADDR=ga019
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export OMP_NUM_THREADS=1
echo "NNODES: $SLURM_JOB_NUM_NODES"
echo "NPROC_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "MASTER_ADDR: $MASTER_ADDR"

singularity exec --nv \
	    --overlay /scratch/$USER/containers/overlay.ext3:ro  \
	    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/zc2309/multi-path-transformer/; \\
		torchrun \\
    		--rdzv-id=$SLURM_JOB_ID \\
    		--nnodes=$SLURM_JOB_NUM_NODES \\
    		--nproc-per-node=4 \\
    		--rdzv-backend=c10d \\
    		--rdzv-endpoint=$MASTER_ADDR \\
    		cli.py configs/500m-pile-1a100-80gb.yml"

