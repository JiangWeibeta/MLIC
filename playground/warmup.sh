work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
nohup python warmup.py --metrics mse --exp slic_ablation_channel_0018_linear_192v2 --gpu_id 3 --lambda 0.0018 -lr 1e-4 --clip_max_norm 1.0 --seed 1984 --batch-size 8 & > 0018v2.txt
