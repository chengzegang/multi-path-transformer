CONFIG=$1
export OMP_NUM_THREADS=1
NNODES=1
NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
export NNODES=$NNODES
export NPROC_PER_NODE=$NPROC_PER_NODE

torchrun \
    --nnodes=$NNODES \
    --nproc-per-node=$NPROC_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    cli.py CONFIG