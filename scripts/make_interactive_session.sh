#!/bin/bash
srun --mem=70G -c 16 --gres=gpu:1 --qos=nopreemption -p interactive -n 1 --pty bash