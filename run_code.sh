export CUDA_VISIBLE_DEVICES=-1
python example.py --job_name "ps" --task_index 0 &
export CUDA_VISIBLE_DEVICES=0
python example.py --job_name "worker" --task_index 0 &
export CUDA_VISIBLE_DEVICES=1
python example.py --job_name "worker" --task_index 1 &
export CUDA_VISIBLE_DEVICES=2
python example.py --job_name "worker" --task_index 2 
