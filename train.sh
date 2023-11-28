export OMP_NUM_THREADS=1
NNODES=1
NPRC_PER_NODE=$(nvidia-smi -L | wc -l)
export NNODES=$NNODES
export NPRC_PER_NODE=$NPRC_PER_NODE
torchrun \
    --nnodes=$NNODES \
    --nproc-per-node=$NPRC_PER_NODE \
    --max-restarts=3 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    cli.py configs/ai4ce.yml