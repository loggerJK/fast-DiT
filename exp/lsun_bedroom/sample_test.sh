#!/bin/bash

export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python sample.py --model DiT-L/2 --image-size 256 --ckpt /media/dataset1/project/jiwon/fast-DiT/results/DiT-L-2_register8/checkpoints/0050000.pt --register 8 --save_attn
