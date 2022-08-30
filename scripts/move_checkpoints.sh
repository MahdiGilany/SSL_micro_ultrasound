#!/bin/bash

n=0

for line in $(cat ckpts.txt)
do
    echo $line
    echo resnet_10_supervised_noaugs_ckpts/version_${n}.ckpt
    cp $line resnet_10_supervised_noaugs_ckpts version_${num}.ckpt
    n=$(($n + 1))
    echo $n
done