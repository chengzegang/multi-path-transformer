CONFIG=$1
export OMP_NUM_THREADS=1

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
export NNODES=$SLURM_JOB_NUM_NODES
export NPROC_PER_NODE=$SLURM_TASKS_PER_NODE
export NUM_WORKERS=$SLURM_JOB_CPUS_PER_NODE
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)


echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "NNODES: $NNODES"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "MASTER_ADDR: $MASTER_ADDR"

torchrun \
    --rdzv-id=$SLURM_JOB_ID \
    --nnodes=$NNODES \
    --nproc-per-node=$NPROC_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR \
    --max-restarts=3 \
    cli.py $CONFIG