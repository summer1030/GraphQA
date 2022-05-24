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

```
Graph
```
