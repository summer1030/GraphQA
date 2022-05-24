
"""
*******************************************************************

Check whether the answer can be a node in the constituency graph.

*******************************************************************
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '7'

import json
import argparse
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize, WhitespaceTokenizer
import spacy,benepar
import pickle

import torch
import torch.nn as nn
import torch_scatter
from transformers import BatchEncoding


print('Initialize the spaCy model...')
#initialize the parsing model
spacy.prefer_gpu()
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('benepar', config={'model': 'benepar_en3_large'})

if spacy.prefer_gpu():
    print("SpaCy is using GPU.")
else:
    print("SpaCy does not use GPU.")

def load_json(file):
    return json.load(open(file,'r'))

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

def process_text(sent):
    return sent.strip(' ').strip('\n').replace('″','"').replace('…','...').replace('½','*').replace('\n',' ').replace('  ',' ').replace('´','\'').replace('ﬂ','f').replace('№','No')

# def get_sent_id(sents,answer_start_pos):
#     count=0
#     for i,sent in enumerate(sents):
#         if sent==" " or "":
#             continue
#         #print(sent+"********")
#         try:
#             count+=list(WhitespaceTokenizer().span_tokenize(sent))[-1][1]+1
#         except:
#             import pdb;pdb.set_trace()
#         if count > answer_start_pos:
#             break
#     return i

# #meanshile, generate constituency parsing results
def get_sent_id(sents,answer_start_pos):
    count=0
    for i,sent in enumerate(sents):
        count+=list(WhitespaceTokenizer().span_tokenize(sent))[-1][1]+1
        if count > answer_start_pos:
            break
    return i

def parse_context(spacy_doc):
    sents=[]
    parsed_sents=[]
    for i,sent in enumerate(spacy_doc.sents):
        if str(sent) == " " or "":
            continue
        sents += [str(sent)]
        parsed_sents += [sent._.parse_string]
    return sents,parsed_sents

def get_leaves(node,words=[]):

    if node['children'] == []:
        words += [node['name']]
    else:
        for each in node['children']:
            get_leaves(each,words)
    return

def get_constituents(nodes):
    constituents=[]
    for node_id,node in enumerate(nodes):
        words=[]
        get_leaves(node,words)
        constituent = ' '.join(words)
        constituents += [(node_id,constituent)]
    return constituents

def reduce_nodes(nodes):

    reduced_nodes = []
    reduced_nodeid_mapping = {}

    for i,node in enumerate(nodes):
        if node['nodeType'] == 'Internal' and len(node['children'])==1 and node['children'][0]['nodeType'] == 'Leaf':
            continue
        else:
            #print(i,node['nodeID'])
            reduced_nodeid_mapping[node['nodeID']] = len(reduced_nodes)
            node['nodeID'] = len(reduced_nodes)
            reduced_nodes += [node]

    return reduced_nodes,reduced_nodeid_mapping

def update_nodeid(node,reduced_nodeid_mapping):
    if not node['children']:
        return
    else:
        for child in node['children']:
            if child['nodeID'] in reduced_nodeid_mapping:
                child['nodeID']=reduced_nodeid_mapping[child['nodeID']]
            update_nodeid(child,reduced_nodeid_mapping)

def main(args):

    data = load_json('../data/{}-v{}-modified.json'.format(args.data_split,args.squad_version))
    #parsed_by_qid = load_json('./all_con_parsed_by_qid_{}.json'.format(args.data_split))

    have_multiple_answer_nodes=[]
    cannot_find_answer_nodes=[]

    parsed_info={}

    for doc_id,doc in enumerate(tqdm(data['data'])):
        for para_id,para in enumerate(doc['paragraphs']):
            context = para['context']
            try:
                spacy_doc=nlp(context)
            except:
                try:
                    spacy_doc=nlp(process_text(context))
                except:
                    import pdb;pdb.set_trace
            sents,parsed_sents = parse_context(spacy_doc)
            for qa_id,qa in enumerate(para['qas']):
                qid = qa['id']
                answer = qa['answers'][0]['text']
                answer_start_pos = qa['answers'][0]['answer_start']

                # # already have the parsed results
                # sent_id = get_sent_id(doc,context,answer_start_pos)
                # parsed_sent = parsed_by_qid[qid]['parsed_context'][sent_id]

                #generate the parsed results

                sent_id = get_sent_id(sents,answer_start_pos)
                parsed_sent = parsed_sents[sent_id]
                nodes = processConstituency(parsed_sent)

                if args.reduce_nodes_operation:
                    nodes,reduced_nodeid_mapping=reduce_nodes(nodes)
                    for node in nodes:
                        update_nodeid(node, reduced_nodeid_mapping)

                constituents = get_constituents(nodes)

                answer_nodes=[]
                for constituent in constituents:
                    # predict among all virtual nodes
                    if constituent[1] == answer and nodes[constituent[0]]['nodeType']=='Internal':
                        # add this constraint -> only predict among the virtual nodes that do not represent the pos tags
                        if len(nodes[constituent[0]]['children']) == 1 and nodes[constituent[0]]['children'][0]['nodeType']=='Leaf':
                            continue
                        answer_nodes += [constituent]


                if len(answer_nodes)>1:
                    have_multiple_answer_nodes += [(doc_id,para_id,qa_id,qid)]
                    continue

                if answer_nodes == []:
                    cannot_find_answer_nodes += [(doc_id,para_id,qa_id,qid)]
                    continue

                question = qa['question']
                try:
                    spacy_q = nlp(question)
                except:
                    spacy_q=nlp(process_text(context))
                q,parsed_q = parse_context(spacy_q)

                parsed_info[qid]={}
                parsed_info[qid]['doc_id'] = doc_id
                parsed_info[qid]['para_id'] = para_id
                parsed_info[qid]['qa_id'] = qa_id
                parsed_info[qid]['tokenized_context'] = sents
                parsed_info[qid]['conparsed_context'] = parsed_sents
                parsed_info[qid]['virtual_noodes'] = nodes
                parsed_info[qid]['answer_sent_id'] = sent_id
                parsed_info[qid]['answer_node_id'] = answer_nodes
                parsed_info[qid]['tokenized_question'] = q
                parsed_info[qid]['conparsed_question'] = parsed_q

    #             with open('original_parsed_sents_part1'.format("train"),'a') as fout:
    #                 for sent_idx,sent in enumerate(sents):
    #                     fout.write(str(doc_id)+' '+str(para_id)+' '+str(qa_id)+' '+qid+str(sent_idx)+' '+sent+'\n')

    #             with open('constituency_parsed_sents_part1'.format("train"),'a') as fout:
    #                 for sent_idx,parsed_sent in enumerate(parsed_sents):
    #                     fout.write(str(doc_id)+' '+str(para_id)+' '+str(qa_id)+' '+qid+str(sent_idx)+' '+parsed_sent+'\n')
                # if qid == '57339a5bd058e614000b5e91':
                #     import pdb;pdb.set_trace()
                # except:
                #     import pdb;pdb.set_trace()

    # with open('have_multiple_answer_nodes_{}_part2'.format(args.data_split),'w') as fh:
    #     for line in have_multiple_answer_nodes:
    #         for item in line[:-1]:
    #             fh.write(str(item)+' ')
    #         fh.write(line[-1]+'\n')
    #     #fout.write(str(doc_id)+' '+str(para_id)+' '+str(qa_id)+' '+qid+'\n')

    # with open('cannot_find_answer_nodes_{}_part2'.format(args.data_split),'w') as fc:
    #     #fout.write(str(doc_id)+' '+str(para_id)+' '+str(qa_id)+' '+qid+'\n')
    #     for line in cannot_find_answer_nodes:
    #         for item in line[:-1]:
    #             fc.write(str(item)+' ')
    #         fc.write(line[-1]+'\n')


    with open('have_multiple_answer_nodes_{}_reduced.pkl'.format(args.data_split),'wb') as fh:
        pickle.dump(have_multiple_answer_nodes,fh)

    with open('cannot_find_answer_nodes_{}_reduced.pkl'.format(args.data_split),'wb') as fc:
        pickle.dump(cannot_find_answer_nodes,fc)

    with open('parsed_info_original_{}_reduced.pkl'.format(args.data_split),'wb') as fout:
        pickle.dump(parsed_info,fout)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split','-d', default=None, required=True, type=str, help='dev or train')
    parser.add_argument('--squad_version','-v', default='1.1', type=str, help='1.1 or 2.0')
    parser.add_argument('--reduce_nodes_operation', '-reduce_nodes', action='store_true', help='whether reduce nodes')

    args = parser.parse_args()

    main(args)
