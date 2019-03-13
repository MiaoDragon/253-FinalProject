python3 train.py --env Pendulum-v0 --max_epi 500 --max_iter 100 \
--save_epi 100 --memory_capacity 1000 --learning_rate 0.01 \
--obs_num 4 --model_path ../model/baseline.pkl --seed 1 --start_std 2. \
--final_std 0.1 --std_decay_epi_ratio 0.7
