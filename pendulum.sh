python3 train_gaussian.py --env Pendulum-v0 --max_epi 4000 --max_iter 1000 \
--save_epi 100 --memory_capacity 100 --learning_rate 0.01 \
--obs_num 1 --model_path ../model/baseline.pkl --seed 1 --use_cnn 0 \
--importance_all 1 --clipping 0
