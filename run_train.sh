#! /bin/sh

for time in {1..1}
do
	python3 train_graph.py --model_type bert \
		--model_name_or_path bert-base-cased \
	       	--data_dir ./graphdata_reduced/ \
			--output_dir ~/saved/graph_qa/graph_embed_intialize_binary_reduced/\
			--tensorboard_save_path ./runs/graph_embed_intialize_binary_reduced/\
			--train_file train-v1.1-filtered.json \
			--predict_file dev-v1.1-filtered.json \
			--save_steps 200 \
			--logging_steps 200\
			--do_train\
			--do_eval \
			--num_train_epochs 7 \
			--evaluate_during_training\
			--begin_evaluation_steps 600\
			--learning_rate 2e-5 \
			--per_gpu_train_batch_size 25\
			--per_gpu_eval_batch_size 25\
			--overwrite_output_dir \
			--max_seq_length 384 \
			--threads 10 \

	done
