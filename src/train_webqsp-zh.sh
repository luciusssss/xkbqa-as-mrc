# 1. train on squad 1.1
python run_squad.py \
--model_type roberta \
--model_name_or_path xlm-roberta-large \
--do_train \
--do_lower_case \
--data_dir ../data/external/squad \
--train_file train-v1.1.json \
--predict_file dev-v1.1.json \
--cache_dir ../models/xlm-roberta-large_squad \
--per_gpu_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--save_steps 100000 \
--logging_steps 50 \
--output_dir ../models/xlm-roberta-large_squad \
--overwrite_output_dir

# 2. train on xmrc data
python run_squad.py \
--model_type roberta \
--model_name_or_path ../models/xlm-roberta-large_squad \
--do_train \
--do_lower_case \
--data_dir ../data/external/xmrc \
--train_file combined_xmrc_zh.json \
--cache_dir ../models/xlm-roberta-large_squad-xmrc \
--per_gpu_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--save_steps 100000 \
--logging_steps 50 \
--output_dir ../models/xlm-roberta-large_squad-xmrc \
--overwrite_output_dir

# 3. train on webqsp-zh
python run_squad_choice.py \
--model_type xlm-roberta \
--model_name_or_path ../models/xlm-roberta-large_squad-xmrc \
--do_train \
--do_eval \
--do_lower_case \
--data_dir ../data/processed/webqsp-zh_xkbqa-as-mrc \
--train_file train_squad.json \
--predict_file test_squad.json \
--cache_dir ../models/xlm-roberta-large_squad-xmrc_webqsp-zh \
--per_gpu_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--save_steps 5000 \
--logging_steps 50 \
--output_dir ../models/xlm-roberta-large_squad-xmrc_webqsp-zh \
--overwrite_output_dir