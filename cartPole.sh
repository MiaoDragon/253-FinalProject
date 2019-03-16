python3 discrete_train.py --env CartPole-v0 --max_epi 500 --max_iter 500 \
--save_epi 100 --memory_capacity 100 --learning_rate 0.01 \
--obs_num 1 --model_path ../model/cartPole_allIS_no_clip.pkl --seed 1 --use_cnn 0 \
--importance_all 1 --clipping 1 --clip_upper 10 --clip_lower 1 --gamma 1.
