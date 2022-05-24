# Converting the standard QA task to graph prediction task

The development environment can be checked in `requirements.txt`.

## Process data

Use the [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/) dataset.

Use the banepar parser via spaCy. More information of the parser can be found here,
```
https://github.com/nikitakit/self-attentive-parser,
https://spacy.io/universe/project/self-attentive-parser.
```
Process the data to perform a graphical task.

Choose samples whose answer is one internal node in the constituency graph excluding the part-of-speech nodes.

The scripts for processing the data are in `./utils/`.

### Step1

```
python ProcessAnswer.py --data_split dev --squad_version 1.1 --reduce_nodes_operation

Generate three intermediate files,
 - have_multiple_answer_nodes_{data.split}.pkl
 - cannot_find_answer_nodes_{data.split}.pkl
 - parsed_info_original_{data.split}.pkl
```

### Step2

Input the three files from the previous step,

```
python FilterDataset.py \
        --cannot_find_answer_nodes ./cannot_find_answer_nodes_train.pkl \
        --have_multiple_answer_nodes ./have_multiple_answer_nodes_train.pkl \
        --parsed_info ./parsed_info_originaltrain.pkl \
        --squad_version 1.1 \
        --original_data ../data/train-v1.1-modified.json \
        --save_file ../data/train-v1.1-filtered.json \
```
```
Totally, there are 42584 samples after filtering.
We split 34893 question-answer pairs from 360 documents for training, 
and 7691 question-answer pairs from 82 documents for developing.
```

The intermediate files can be downloaded [here](https://drive.google.com/file/d/1kD6XOQz8uzMkH_leBnasihX5V04tb4jm/view?usp=sharing). Save all to the `./utils/` directory.

The processed data can be downloaded [here](https://drive.google.com/file/d/1_0TfSoGiRDpxgyVZNb0izMTicFAVDAuZ/view?usp=sharing). Save all to the `./data/` directory.


## Build graphs

`GraphBuildingByClassInitialize.py:` Build the constituency graphs. The virtual nodes are initialized by specific categories. For instance, all NP nodes use the same initialized, and all VP nodes use the same initialization.

`GraphBuildingByEmbedInitialize.py:` Build the constituency graphs. Each virtual node is initialized by content embeddings of its connected nodes at lower level.
 
`GraphBuildingByEmbedInitializeReduced.py:` Build the constituency graphs with reduced nodes. Remove all internal nodes that have a single child, or remove all part-of-sppech nodes in the constituency graph. Each virtual node is initialized by content embeddings of its connected nodes at lower level.

Run the scripte by,
```
python GraphBuildingByClassInitialize.py --data_split dev\
                                         --model_name_or_path bert-case-cased\
                                         --save_path ./data/squad_files/constituency_graphs
```

## Models

`./models/bert_hgt_ori.py:` The original [syntax-informed QA models](https://github.com/summer1030/Syntax-informed-QA).

`./models/bert_hgt_graph.py:` The model for the current work.


## Fine-tune models

The main files for training are,
```
- train_base.py: Train baseline models.
- train_ori.py: Train the original syntax-informed QA models.
- train_graph.py: Train the current model that converts QA to a graphical task.
```

Begin the training by run the script with
```
bash run_train.sh
```

The hypter-parameters are described in `run_train.sh`,

```
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

```


## Evaluate models

Evaluate a saved checkpoing by run the script with
```
bash run_eval.sh
```

The hypter-parameters are described in `run_eval.sh`,

```
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

```

## Process the predictions

The predicted nodes need to be recovered to the texts.

The utils are in `ProcessPrediction.py`. The parameteres can be set are,

```
python ProcessPredictions.py --parsed ./utils/parsed_info_filtered.pkl\
                             --predictions predictions_dev_binary\
                             --cached_file ./data/cached_dev_bert-base-cased_384\
                             --save_file ./results_dev_binary.json\
                             --use_binary\
                             --see_topk\                    
```

