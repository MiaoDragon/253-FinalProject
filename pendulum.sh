python3 train.py --env Pendulum-v0 --max_epi 300 --max_iter 100 \
--save_epi 100 --memory_capacity 1000 --learning_rate 0.001 \
--obs_num 4 --model_path ../model/baseline.pkl --seed 1 --start_std 2. \
--final_std 0.15 --std_decay_epi_ratio 0.7 --use_cnn 0 --clip_factor 0.5
