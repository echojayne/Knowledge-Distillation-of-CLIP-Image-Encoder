#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python3 main.py --batch-size 3000,1024 --epochs 18,200 --lr 0.01,0.001 --device cuda --multi-gpu 1