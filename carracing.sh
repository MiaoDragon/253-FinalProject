# TODO: using cuda is much faster but runs out of memory, may want to move
# trajectory memory on/off gpu throughout training

python3 train.py --env CarRacing-v0 --max_epi 1000 --max_iter 1000 \
--save_epi 100 --memory_capacity 100 --learning_rate 0.001 \
--obs_num 4 --model_path model/carracing.pkl --seed 1 --use_cnn 1 \
--importance_all 1 --clipping 1 --clip_upper 10 --clip_lower 1 --gamma 1. \
--distribution gaussian --frame_interval 8 --use_cuda 0 --init_std 1. \
--final_std .1
