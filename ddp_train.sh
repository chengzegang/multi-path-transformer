export OMP_NUM_THREADS=1
torchrun \
    --nnodes=1 \
    --nproc-per-node=3 \
    --max-restarts=3 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    cli.py configs/ai4ce.yml