work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
nohup python train.py --metrics ms-ssim --exp slic_uneven_charm_ms_0873_192 --gpu_id 1 --lambda 8.73 -lr 1e-4 --clip_max_norm 1.0 --seed 909 --batch-size 8 -c /data2/jiangwei/work_space/CCSLICfinal/playground/experiments/slic_uneven_charn_ms_6050_192/checkpoint_0067.pth.tar & > 0130v2.txt
