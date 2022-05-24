
"""
*******************************************************************

Build the constituency graphs. The virtual nodes are initialized by specific categories.
For instance, all NP nodes use the same initialized, and all VP nodes use the same initialization.

*******************************************************************
"""

import argparse

import torch
import torch.nn as nn
import torch_scatter
from transformers import BatchEncoding
from transformers import AutoTokenizer

import torch
import torch.nn as nn
import torch_scatter
from transformers import BatchEncoding

import spacy, benepar
from tqdm import tqdm
import pickle
import json

print('Initialize the parsing model...')
spacy.prefer_gpu()

#initialize the consistency parsing model
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('benepar', config={'model': 'benepar_en3_large'})



import torch
import torch.nn as nn
import torch_scatter
from transformers import BatchEncoding


def load_parsed_results(data_split):
    results = pickle.load(open('./utils/parsed_info_originaltrain.pkl.back','rb'))
    return results

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

# emb_dim -> dimension of node embeddings
def initialize_tag_embeddings(emb_dim=768):

    #get all tags
    from spacy import glossary
    pos_tags = glossary.GLOSSARY

    #tags do not contain 'S', so add another len(pos_tags)+1 for generaing embedding of 'S'
    tag_embeds = nn.Embedding(len(pos_tags)+1,emb_dim)
    tag_embeds.weight.requires_grad=False

    # embed = virtual_node_embeds[list(pos_tags).index(lookup_tag)]
    # virtual_node_embeds = torch.randn((len(pos_tags),emd_dim),requires_grad=False)

    tag_embeds_dict = {}
    for tag in pos_tags:
        lookup_tensor = torch.tensor(list(pos_tags).index(tag),dtype=torch.long)
        tag_embeds_dict[tag] = tag_embeds(lookup_tensor)
    tag_embeds_dict['S'] = tag_embeds(torch.tensor(len(pos_tags),dtype=torch.long))
    return tag_embeds_dict


def initialize_virtual_node_embeddings(nodes, tag_embeds_dict, emb_dim=768):
    virtual_node_embeds = []
    virtual_node_mapping = {} #map nodes id to virtual node id

    virtual_node_id = 0
    for i, node in enumerate(nodes):
        if node['nodeType'] == 'Leaf':
            continue
        if node['name'] not in tag_embeds_dict:
            print('Add the embedding for {}'.format(node['name']))
            tag_embeds_dict.update({node['name']:torch.randn(emb_dim,requires_grad=False)})
        else:
            virtual_node_embeds += [tag_embeds_dict[node['name']]]
        virtual_node_mapping[node['nodeID']] = virtual_node_id
        virtual_node_id += 1
    return torch.stack(virtual_node_embeds),virtual_node_mapping,tag_embeds_dict

def create_token_to_virtual_edges(nodes, virtual_node_mapping, added_nodes,
    previous_token_num=0, preivous_virtual_node_num=0):
    edge_index = []
    for i, node in enumerate(nodes):
        if len(node['children'])==1 and node['children'][0]['nodeType']=='Leaf':
            #print(node['children'][0]['name'],node['name'])
            #update_word_node_id_in_edges
            if node['children'][0]['wordIndex'] not in added_nodes:
                src_node_id = node['children'][0]['wordIndex'] + previous_token_num+1
            else:
                src_node_id = added_nodes[node['children'][0]['wordIndex']]
            det_node_id = virtual_node_mapping[i] + preivous_virtual_node_num
            edge_index += [[src_node_id, det_node_id]]
    return edge_index

def create_virtual_to_virtual_edges(nodes, virtual_node_mapping,
    previous_token_num=0, preivous_virtual_node_num=0):
    edge_index = []
    for i, node in enumerate(nodes):
        if node['children'] != []:
            det_node_id = virtual_node_mapping[node['nodeID']] + preivous_virtual_node_num
            for direct_neighbor in node['children']:
                if direct_neighbor['nodeType'] == 'Internal':
                    src_node_id = virtual_node_mapping[direct_neighbor['nodeID']] + preivous_virtual_node_num
                    edge_index += [[src_node_id, det_node_id]]
    return edge_index

#bert embeddings for the current sentence
#tokenized_text without special tokens
def add_word_nodes_and_create_subword_to_word_edges(tokenized_text,added_node_embeds=[],max_len=384,previous_token_num=0,previous_added_word_node_num=0):

    edge_index = []
    added_nodes = {}

    for i in range(len(tokenized_text.input_ids)):
        if BatchEncoding.word_to_tokens(tokenized_text,i) != None:
            #print(BatchEncoding.word_to_tokens(tokenized_text,i))
            start_token_pos = BatchEncoding.word_to_tokens(tokenized_text,i).start
            end_token_pos = BatchEncoding.word_to_tokens(tokenized_text,i).end

            if previous_token_num+end_token_pos<=max_len-1 and end_token_pos-start_token_pos>1:

                #added_nodes[added_node_idx] = added_node_idx+previous_added_word_node_num+max_len-1
                added_nodes[i] = len(added_nodes)+previous_added_word_node_num+max_len
                #added_node_embeds += [torch.mean(bert_embeds[start_token_pos:end_token_pos],dim=0)]
                #print(tokenizer.convert_ids_to_tokens(tokenized_text.input_ids))
                added_node_embeds += [(len(added_nodes)+previous_added_word_node_num+max_len-1, previous_token_num+start_token_pos+1, previous_token_num+end_token_pos+1)]
                #print(added_node_embeds)
                for node_idx in range(start_token_pos,end_token_pos):
                    edge_index += [[node_idx+1+previous_token_num,len(added_nodes)+previous_added_word_node_num+max_len-1]]


    return added_nodes,added_node_embeds,edge_index

def recover_sent(nodes):
    return ' '.join([x['name'] for x in nodes if x['nodeType']=='Leaf']).replace(' .','.')

def build_by_sentence(parsed_result,tag_embeds_dict,added_word_token_nodes_stored=[],
                      virtual_to_virtual_edges_stored=[],
                      token_to_token_edges_stored=[],
                      token_to_virtual_edges_stored=[],
                        added_node_embeds_stored=[],
                      virtual_node_embeds_stored=[],
                        preivous_virtual_node_num=0,
                        previous_token_num=1,
                        previous_added_word_node_num=0,max_len=384):

    #parsed_result='(S (NP (NNP Bill)) (ADVP (RB frequently)) (VP (VBD got) (NP (PRP$ his) (NNS buckets)) (PP (IN from) (NP (DT the) (NN store))) (PP (IN for) (NP (DT a) (NN dollar)))) (. .))'

    nodes=processConstituency(parsed_result)

    original_text = recover_sent(nodes)
    tokenized_text = tokenizer(original_text,add_special_tokens=False)
    #print('Porcess sentence,length is {}'.format(len(tokenized_text.input_ids)))

    #except the [CLS] and [SEP]
    if previous_token_num+len(tokenized_text.input_ids) > max_len-3:
        keep_token_num = max_len-3-previous_token_num
        nodes= update_nodes_by_max_len(nodes,tokenized_text,keep_token_num,max_len)

    #initialize the embedding of a virtual node by lookup_tag_embedding by :
    #lookup_tag_embedding[nodes[0]['name']]
    virtual_node_embeds,virtual_node_mapping,tag_embeds_dict=initialize_virtual_node_embeddings(nodes, tag_embeds_dict)
    virtual_node_embeds_stored += [virtual_node_embeds]


    virtual_to_virtual_edges_stored += create_virtual_to_virtual_edges(nodes, virtual_node_mapping, previous_token_num, preivous_virtual_node_num)

    added_nodes,added_node_embeds_stored,token_to_token_edges = add_word_nodes_and_create_subword_to_word_edges(tokenized_text,added_node_embeds_stored,max_len,previous_token_num,previous_added_word_node_num)
    added_word_token_nodes_stored += list(added_nodes.values())
    token_to_token_edges_stored += token_to_token_edges

    token_to_virtual_edges_stored += create_token_to_virtual_edges(nodes,virtual_node_mapping, added_nodes, previous_token_num, preivous_virtual_node_num)

    preivous_virtual_node_num += virtual_node_embeds.size()[0]
    if previous_token_num+len(tokenized_text.input_ids) > max_len-3:
        previous_token_num += keep_token_num
    else:
        previous_token_num += len(tokenized_text.input_ids)
    previous_added_word_node_num += len(added_nodes)

    return (virtual_to_virtual_edges_stored,
        token_to_token_edges_stored,
        token_to_virtual_edges_stored,
            added_word_token_nodes_stored,
            tag_embeds_dict,
            preivous_virtual_node_num,
            previous_token_num,
            previous_added_word_node_num,
           added_node_embeds_stored,
            virtual_node_embeds_stored,
           )

def update_nodes_by_max_len(nodes,tokenized_text,keep_token_num,max_len=384):
    global nlp
    keep_tokens = tokenized_text.input_ids[:keep_token_num+1]
    try:
        last_word_index = BatchEncoding.token_to_word(tokenized_text,keep_token_num)
        updated_sent = ' '.join(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokenized_text.input_ids)).split(' ')[:last_word_index])
    except:
        import pdb;pdb.set_trace()

    if updated_sent == '':
        updated_sent = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokenized_text.input_ids[0]))
    #re-do constituency parsing for the updated shorter sentence
    doc = nlp(updated_sent)
    sent = list(doc.sents)[0]
    updated_parsing_results = sent._.parse_string
    #tree=Tree.fromstring(updated_parsing_results)
    #tree.pretty_print()
    updated_nodes=processConstituency(updated_parsing_results)
    return updated_nodes

def main(args):

    global tokenizer

    print('Load the constituency parsing results...')
    parsed_results = load_parsed_results(args.data_split)

    print('Begin to process...')

    print('Initialize or load the embedding for all all tags...')
    #initialize the embeddings of all tags
    #virtual nodes of same tag have the same initialization
    #tag_embeds_dict = initialize_tag_embeddings(args.emb_dim)
    with open('./tag_embeds_dict','rb') as f:
        tag_embeds_dict = pickle.load(f)

    all_info = {}

    max_len = args.max_len

    qapList = sorted(list(parsed_results.keys()))

    for qid in tqdm(qapList):

        #for q_idx in range(len(parsed_results[para_id]['qas'])):

        qap_info = {}

        preivous_virtual_node_num=0
        previous_token_num=1
        previous_added_word_node_num=0

        virtual_to_virtual_edges = []
        virtual_node_embeds = []
        token_to_token_edges = []
        token_to_virtual_edges= []
        added_word_token_nodes = []
        added_node_embeds = []

        for sen_idx in range(len(parsed_results[qid]['conparsed_question'])):
            parsed_result = parsed_results[qid]['conparsed_question'][sen_idx]

            outputs = build_by_sentence(parsed_result,tag_embeds_dict,
                            added_word_token_nodes,
                              virtual_to_virtual_edges,
                              token_to_token_edges,
                              token_to_virtual_edges,added_node_embeds,virtual_node_embeds,preivous_virtual_node_num,previous_token_num,previous_added_word_node_num)

            virtual_to_virtual_edges = outputs[0]
            token_to_token_edges = outputs[1]
            token_to_virtual_edges =  outputs[2]
            added_word_token_nodes =  outputs[3]

            tag_embeds_dict = outputs[4]
            preivous_virtual_node_num = outputs[5]
            previous_token_num = outputs[6]
            previous_added_word_node_num = outputs[7]
            added_node_embeds = outputs[8]
            virtual_node_embeds = outputs[9]


        previous_token_num += 1

        answer_info_sent_id = parsed_results[qid]['answer_sent_id']
        answer_info_node_id = parsed_results[qid]['answer_node_id'][0][0]
        answer_info = (answer_info_sent_id,answer_info_node_id)

        #first process the questions, then process the context
        for sen_idx in range(len(parsed_results[qid]['conparsed_context'])):

            if previous_token_num >=max_len-3:
                break

            if sen_idx == answer_info[0]:
                answer_node = preivous_virtual_node_num + answer_info[1]

            #parsed_result='(S (NP (NNP Bill)) (ADVP (RB frequently)) (VP (VBD got) (NP (PRP$ his) (NNS buckets)) (PP (IN from) (NP (DT the) (NN store))) (PP (IN for) (NP (DT a) (NN dollar)))) (. .))'

            parsed_result = parsed_results[qid]['conparsed_context'][sen_idx]

            outputs = build_by_sentence(parsed_result,tag_embeds_dict,
                            added_word_token_nodes,
                              virtual_to_virtual_edges,
                              token_to_token_edges,
                              token_to_virtual_edges,added_node_embeds,virtual_node_embeds,preivous_virtual_node_num,previous_token_num,previous_added_word_node_num,max_len)

            #print('added_word_token_nodes',added_word_token_nodes)

            virtual_to_virtual_edges = outputs[0]
            token_to_token_edges = outputs[1]
            token_to_virtual_edges =  outputs[2]
            added_word_token_nodes =  outputs[3]

            tag_embeds_dict = outputs[4]
            preivous_virtual_node_num = outputs[5]
            previous_token_num = outputs[6]
            previous_added_word_node_num = outputs[7]
            added_node_embeds = outputs[8]
            virtual_node_embeds = outputs[9]

        #(word_idx,start_pos,end_pos)
        qap_info['virtual_node_embeds'] = torch.cat(virtual_node_embeds,dim=0)
        qap_info['virtual_to_virtual_edges'] = virtual_to_virtual_edges
        qap_info['token_to_token_edges'] = token_to_token_edges if token_to_token_edges != [] else [(0,0)]
        qap_info['token_to_virtual_edges'] = token_to_virtual_edges
        qap_info['added_node_embeds_mapping'] = added_node_embeds if added_node_embeds != [] else [(0,0,0)]
        qap_info['answer_node'] = answer_node

        with open('{}/{}'.format(args.save_path,qid), "wb") as f:
            pickle.dump(qap_info, f)


    with open('tag_embeds_dict','wb') as f:
        pickle.dump(tag_embeds_dict, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split','-d', default="train", type=str, help='dev or train')
    parser.add_argument('--emb_dim', default=768, type=int, help='dimension of node embedding')
    parser.add_argument('--max_len', default=384, type=int, help='max length of the input sequence')
    parser.add_argument("--model_name_or_path", default='bert-base-cased', type=str, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument('--save_path','-s', default="./data/squad_files/constituency_graphs", type=str, help='Path to save the generated graph data')

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print('Initialize the pre-trained tokenizer of {}...'.format(args.model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    main(args)
