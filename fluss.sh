#!/bin/sh
#SBATCH --cpus-per-task=128
#SBATCH --output=/home/rathjjgf/logs/slurm-%j.out
#SBATCH --error=/home/rathjjgf/logs/slurm-%j.err
#SBATCH --time=10-00:00:00           # estimated runtime (dd-hh:mm:ss)


python3 ~/repos/mask_image_reconstruct_image/refactored/train_transformer.py -c 2 -ms selective -bs 128 -d -m  &
python3 ~/repos/mask_image_reconstruct_image/refactored/train_transformer.py -c 2 -ms selective -bs 128 -d -m  &
python3 ~/repos/mask_image_reconstruct_image/refactored/train_transformer.py -c 2 -ms selective -bs 128 -d -m  &
python3 ~/repos/mask_image_reconstruct_image/refactored/train_transformer.py -c 2 -ms selective -bs 128 -d -m  &
python3 ~/repos/mask_image_reconstruct_image/refactored/train_transformer.py -c 2 -ms selective -bs 128 -d -m  &

python3 ~/repos/mask_image_reconstruct_image/refactored/train_transformer.py -c 1 -ms selective -bs 128 -d -m  &
python3 ~/repos/mask_image_reconstruct_image/refactored/train_transformer.py -c 1 -ms selective -bs 128 -d -m  &
python3 ~/repos/mask_image_reconstruct_image/refactored/train_transformer.py -c 1 -ms selective -bs 128 -d -m  &
python3 ~/repos/mask_image_reconstruct_image/refactored/train_transformer.py -c 1 -ms selective -bs 128 -d -m  &
python3 ~/repos/mask_image_reconstruct_image/refactored/train_transformer.py -c 1 -ms selective -bs 128 -d -m  &

wait
exit 0