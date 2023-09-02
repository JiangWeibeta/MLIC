work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
nohup python train.py --metrics ms-ssim --exp mlicpp_msssim_q1 --gpu_id 1 --lambda 2.4 -lr 1e-4 --clip_max_norm 1.0 --seed 909 --batch-size 8 -c /data2/jiangwei/work_space/MLICPP/playground/experiments/mlicpp_msssim_q1/checkpoint_0067.pth.tar & > 0130v2.txt
