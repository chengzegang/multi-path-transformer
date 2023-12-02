CONFIG=$1
export OMP_NUM_THREADS=1

torchrun \
    --rdzv-id=0 \
    --nnodes=1 \
    --nproc-per-node=8 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    cli.py $CONFIG