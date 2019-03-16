python3 train.py --env Pendulum-v0 --max_epi 4000 --max_iter 200 \
--save_epi 100 --memory_capacity 1000 --learning_rate 0.001 \
--obs_num 1 --model_path ../model/pendulum_beta_clip_allIS.pkl --seed 1 --use_cnn 0 \
--importance_all 1 --clipping 1 --clip_upper 10 --clip_lower 1 --gamma 1.
