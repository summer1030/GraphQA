#! /bin/sh

for time in {1..1}
do
	python3 eval_graph.py --model_type bert \
		--model_name_or_path bert-base-cased \
	       	--data_dir ./graphdata/ \
			--output_dir ./test/ \
			--tensorboard_save_path ./runs/test\
			--train_file train-v1.1-filtered.json.back \
			--predict_file dev-v1.1-filtered.json.back \
			--save_steps 1\
			--logging_steps 1\
			--do_train \
			--evaluate_during_training \
			--num_train_epochs 1\
			--begin_evaluation_steps 0\
			--learning_rate 2e-5 \
			--per_gpu_train_batch_size 32\
			--per_gpu_eval_batch_size 32\
			--overwrite_output_dir \
			--version_2_with_negative \
			--max_seq_length 384 \
			--threads 10\
			--gpus 6 \

done
