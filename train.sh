export OMP_NUM_THREADS=1
NNODES=1
NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
export NNODES=$NNODES
export NPROC_PER_NODE=$NPROC_PER_NODE

torchrun \
    --nnodes=$NNODES \
    --nproc-per-node=2 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    cli.py configs/500m.yml