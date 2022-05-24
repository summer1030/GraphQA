import re
import collections
import json
import os
import logging
import torch
from functools import partial
# from torch.multiprocessing import Pool
from multiprocessing import Pool, cpu_count, Queue # import Pool
from torch_geometric.data import Data
# import malaya
import numpy as np
import nltk
import tokenizations
import time
import string
import pickle
from tqdm import tqdm
from transformers.data.metrics.squad_metrics import _get_best_indexes, _compute_softmax, normalize_answer
from transformers.data.processors.squad import _improve_answer_span, _new_check_is_max_context, SquadFeatures

from transformers.file_utils import is_tf_available, is_torch_available
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy
#from transformers.utils import logging

from SquadData import SquadData

MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}

logger = logging.getLogger(__name__)

def squad_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning(f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'")
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implementation also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        edge_info = pickle.load(open('/home/fangyi/graph-qa-main/data/squad_files/constituency_graphs_reduced/{}'.format(example.qas_id),'rb'))

        features.append(
            ExtendedSquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
                num_nodes = len(tokens),
                t2t_edge_index = edge_info['token_to_token_edges'],
                t2v_edge_index = edge_info['token_to_virtual_edges'],
                v2v_edge_index = edge_info['virtual_to_virtual_edges'],
                added_node_mapping = edge_info['added_node_embeds_mapping'],
                virtual_node_mapping = edge_info['virtual_node_embeds_mapping'],
            )
        )
    return features

def squad_convert_example_to_features_init(tokenizer_for_convert: PreTrainedTokenizerBase):
    global tokenizer
    tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model. It is
    model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset, if 'tf': returns a tf.data.Dataset
        threads: multiple processing threads.


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """
    # Defining helper methods
    features = []

    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )

    new_features = []
    unique_id = 0
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # # modify feature index here
        if not is_training:
            # cls_idx and feature_idx are named like this because
            # "Any attribute that is named *index will automatically increase its value based on the cumsum of nodes"
            # https://github.com/rusty1s/pytorch_geometric/issues/2052
            dataset = [SquadData(input_ids=torch.tensor(f.input_ids, dtype=torch.long),
                            attention_mask=torch.tensor(f.attention_mask, dtype=torch.long),
                            token_type_ids=torch.tensor(f.token_type_ids, dtype=torch.long),
                            feature_idx=torch.tensor(f.unique_id, dtype=torch.long),
                            cls_idx=torch.tensor(f.cls_index, dtype=torch.long),
                            p_mask=torch.tensor(f.p_mask, dtype=torch.float),
                            t2t_edge_index=torch.transpose(torch.tensor(f.t2t_edge_index,dtype=torch.long),0,1),
                            t2v_edge_index=torch.transpose(torch.tensor(f.t2v_edge_index,dtype=torch.long),0,1),
                            v2v_edge_index=torch.transpose(torch.tensor(f.v2v_edge_index,dtype=torch.long),0,1),
                            added_node_mapping=torch.transpose(torch.tensor(f.added_node_mapping, dtype=torch.long),0,1),
                            virtual_node_mapping=torch.transpose(torch.tensor(f.virtual_node_mapping, dtype=torch.long),0,1),
                            qas_id=f.qas_id,
                            num_token_to_orig_map=torch.tensor(len(f.token_to_orig_map), dtype=torch.long),
                            num_nodes=f.num_nodes) for i, f in enumerate(features)]
        else:
            dataset = []
            for i, f in enumerate(features):
                # if f.t2t_edge_index != []:
                #     f.t2t_edge_index = torch.transpose(torch.tensor(f.t2t_edge_index,dtype=torch.long),0,1)
                # else:
                #     f.t2t_edge_index = torch.tensor(f.t2t_edge_index,dtype=torch.long)
                dataset.append(SquadData(input_ids=torch.tensor(f.input_ids, dtype=torch.long),
                            attention_mask=torch.tensor(f.attention_mask, dtype=torch.long),
                            token_type_ids=torch.tensor(f.token_type_ids, dtype=torch.long),
                            cls_idx=torch.tensor(f.cls_index, dtype=torch.long),
                            start_position=torch.tensor(f.start_position, dtype=torch.long),
                            end_position=torch.tensor(f.end_position, dtype=torch.long),
                            feature_idx=torch.tensor(f.unique_id, dtype=torch.long),
                            p_mask=torch.tensor(f.p_mask, dtype=torch.float),
                            t2t_edge_index=torch.transpose(torch.tensor(f.t2t_edge_index,dtype=torch.long),0,1),
                            t2v_edge_index=torch.transpose(torch.tensor(f.t2v_edge_index,dtype=torch.long),0,1),
                            v2v_edge_index=torch.transpose(torch.tensor(f.v2v_edge_index,dtype=torch.long),0,1),
                            added_node_mapping=torch.transpose(torch.tensor(f.added_node_mapping, dtype=torch.long),0,1),
                            virtual_node_mapping=torch.transpose(torch.tensor(f.virtual_node_mapping, dtype=torch.long),0,1),
                            qas_id=f.qas_id,
                            num_token_to_orig_map=torch.tensor(len(f.token_to_orig_map), dtype=torch.long),
                            num_nodes=f.num_nodes
                            ))

        return features, dataset
    else:

        return features

class ExtendedSquadFeatures(SquadFeatures):
    def __init__(self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        t2t_edge_index,
        t2v_edge_index,
        v2v_edge_index,
        added_node_mapping,
        virtual_node_mapping,
        num_nodes,
        qas_id: str = None,
        encoding: BatchEncoding = None,
    ):
        super().__init__(input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        qas_id,
        encoding)
        self.t2t_edge_index = t2t_edge_index
        self.t2v_edge_index = t2v_edge_index
        self.v2v_edge_index = v2v_edge_index
        self.added_node_mapping = added_node_mapping
        self.virtual_node_mapping = virtual_node_mapping
        self.num_nodes = num_nodes

def process_sequence(sequences, batch_first=False, padding_value=0.0):
    # we need to pad to equal length, if not dataparallel will fail..
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = 384 # pad to 384
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor
