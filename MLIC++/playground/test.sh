work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
CUDA_VISIBLE_DEVICES='0' python test.py -exp toy_exp_0067 --gpu_id 0 -c /data00/jiangwei/work_space/mlic++_mse_q2.pth.tar -d /data00/jiangwei/dataset/image
