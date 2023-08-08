work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
nohup python train.py --metrics mse --exp slic_ablation_channel_0035_7777_192v2 -c /data2/jiangwei/work_space/CCSLICfinal/playground/experiments/slic_ablation_channel_0035_7777_192v2/checkpoints/checkpoint_025.pth.tar --gpu_id 1 --lambda 0.0035 -lr 1e-4 --clip_max_norm 1.0 --seed 1984 --batch-size 8 & > 0035v2.txt
