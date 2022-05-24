
import re, csv
import argparse
import pickle
import json
import math
import string
import torch
import collections
from tqdm import tqdm
from os.path import exists

from CustomSquadMethods import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers import AutoTokenizer

def LoadPickle(file):
    return pickle.load(open(file,'rb'))

def LoadJson(file):
    return json.load(open(file,'r'))

def LoadGT(cached_file):
    features_and_dataset = torch.load(cached_file)
    features, dataset, examples = (
    features_and_dataset["features"],
    features_and_dataset["dataset"],
    features_and_dataset["examples"],
)
    return examples

def processConstituency(pStr):
    nodes = []
    cur = "";
    stack = [];
    nid = 0;
    wordIndex = 0
    for i in range(len(pStr)):
        if(pStr[i] == ' ' or pStr[i] == '\n'):
            if (len(cur) > 0):
                newNode = {
                    "nodeID": nid,
                    "nodeType": "Internal",
                    "name": cur,
                    "children": []
                }
                cur = "";
                nid += 1;
                if (len(stack) > 0):
                    stack[len(stack) - 1]["children"].append(newNode);
                stack.append(newNode);
                nodes.append(newNode)
        elif pStr[i] == ')':
            if (len(cur) > 0):
                newNode = {
                    "nodeID": nid,
                    "nodeType": "Leaf",
                    "name": cur,
                    "wordIndex": wordIndex,
                    "children": []
                }
                cur = "";
                nid += 1;
                wordIndex += 1;
                stack[len(stack) - 1]["children"].append(newNode);
                nodes.append(newNode)
                stack.pop();
            else:
                if (len(stack) == 1):
                    root = stack[0]
                stack.pop();
        elif pStr[i] == '(':
            continue
        else:
            cur = cur + pStr[i];
    return nodes

def get_leaves(node,words=[]):

    if node['children'] == []:
        words += [node['name']]
    else:
        for each in node['children']:
            get_leaves(each,words)
    return

def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:

        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]
        #gold_answers = normalize_answer(example.answer_text)

        # if not gold_answers:
        #     # For unanswerable questions, only correct answer is empty string
        #     gold_answers = [""]

        if qas_id not in preds:
            print(f"Missing prediction for {qas_id}")
            continue

        prediction = preds[qas_id] if type(preds[qas_id]) is not list else preds[qas_id][0]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

        # if max(compute_f1(a, prediction) for a in gold_answers) == 1:
        #     import pdb;pdb.set_trace()
        # exact_scores[qas_id] = compute_exact(gold_answers, prediction)
        # f1_scores[qas_id] = compute_f1(gold_answers, prediction)

    return exact_scores, f1_scores

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def main(args):
    parsed = LoadPickle(args.parsed)
    predictions = LoadPickle(args.predictions)

    use_binary = args.use_binary
    examples = LoadGT(args.cached_file)
    results = {}
    see_topk = args.see_topk


    if not exists(args.save_file):

        have_problems=[]

        for qid in tqdm(predictions):

            results[qid] = []

            q_num = 0
            for sent in parsed[qid]['conparsed_question']:
                nodes = processConstituency(sent)
                q_num += len(nodes)

            if not see_topk:
                ans_node_idxes =  [torch.argmax(predictions[qid][q_num+2:-1]).item()] if use_binary else [torch.argmax(predictions[qid][q_num+2:-1],dim=0)[1].item()]
            else:
                ans_node_idxes = [x[1].item() for x in torch.topk(predictions[qid][q_num+2:-1],k=5)][1] if use_binary else [x[1].item() for x in torch.topk(predictions[qid][q_num+2:-1],k=5,dim=0)[1]]

            for ans_node_idx in ans_node_idxes:
                cur_num = 2
                for sent in parsed[qid]['conparsed_context']:
                    nodes = processConstituency(sent)
                    if cur_num + len(nodes)>ans_node_idx:
                        break
                    else:
                        cur_num += len(nodes)

                words = []
                get_leaves(nodes[ans_node_idx-cur_num],words)
                span = ' '.join(words)

                results[qid] += [span]

        with open('cannot_predict.text','w') as f:
            for each in have_problems:
                f.write(each+'\n')

        with open(args.save_file,"w") as f:
            json.dump(results,f)

    else:
        results = LoadJson(args.save_file)

    exact_scores, f1_scores = get_raw_scores(examples,results)


    # save to csv

    """
    correct = []
    incorrect = []

    for example in examples:

        qid=example.qas_id


        if qid in results and qid in exact_scores:

            prediction=results[qid]
            question=example.question_text
            passage=example.context_text
            #gt=example.answers[0]['text'] if example.answers!=[] else ""
            gt=example.answer_text
            ex=exact_scores[qid]
            f1=exact_scores[qid]
            if f1>0:
                correct += [(qid,passage,question,prediction[0],prediction[1],prediction[2],prediction[3],prediction[4],gt)]
            else:
                incorrect += [(qid,passage,question,prediction[0],prediction[1],prediction[2],prediction[3],prediction[4],gt)]


    header = ['ID', 'Passage', 'Question', 'Prediction-Top1', 'Prediction-Top2','Prediction-Top3','Prediction-Top4','Prediction-Top5','Ground-Truth']

    with open('correct_examples_training.csv','w') as fc:
        writer = csv.writer(fc)
        writer.writerow(header)
        writer.writerows(correct)

    with open('incorrect_examples_training.csv','w') as fi:
        writer = csv.writer(fi)
        writer.writerow(header)
        writer.writerows(incorrect)
    """


    overall_ex = sum(exact_scores.values())/len(exact_scores.values())
    overall_f1 = sum(f1_scores.values())/len(f1_scores.values())


    print(exact_scores,f1_scores)

    print('EM/F1: {}/{}'.format(overall_ex, overall_f1))



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--parsed', defulat='./utils/parsed_info_filtered.pkl', type=str, help='Path to the parsed results')
    parser.add_argument('--predictions', default='./predictions_dev_binary', type=str, help='Path to the predicted results')
    parser.add_argument('--cached_file', default='./graphdata/cached_dev_bert-base-cased_384', type=str, help='Path to the cached data')
    parser.add_argument('--save_file', default='./results_dev_binary.json', type=str, help='Path to save the output file')
    parser.add_argument('--use_binary', action='store_true', help='max length of the input sequence')
    parser.add_argument('--see_topk', action='store_true', help='max length of the input sequence')


    main(args)
