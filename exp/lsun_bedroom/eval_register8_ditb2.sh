export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

for register in 8; do
    torchrun --rdzv-endpoint=localhost:29404 --nnodes=1 --nproc_per_node=1 sample_ddp.py --model DiT-B/2 --num-fid-samples 50000 --ckpt results/DiT-B-2_register8_lsun/checkpoints/0050000.pt --register $register --lsun
done