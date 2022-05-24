
"""
*******************************************************************

Filter the original dataset to generate a subset for our graph task.

*******************************************************************
"""

import json, os
import pickle
from tqdm import tqdm

def loadText(file):
    return [x.strip('\n').split(' ') for x in open(file,'r').readlines()]

def loadJson(file):
    return json.load(open(file,'r'))

def loadPickle(file):
    return pickle.load(open(file,'rb'))

def main(args):

    print('Load data...')
    CannotFindAnswerNodes = loadPickle(args.cannot_find_answer_nodes)
    HaveMultipleAnswerNodes = loadPickle(args.have_multiple_answer_nodes)
    ParsedInfo = loadPickle(args.parsed_info)

    #LongPassages = loadPickle('./long_passages.pkl')
    filtered = os.listdir('../data/squad_files/constituency_graphs/')

    OriginalData = loadJson(args.original_data)

    FilteredData = {}
    FilteredData['version'] = args.squad_version
    FilteredData['data'] = []

    print('Begin to process ...')
    remainedNum = 0
    ## the first 360 documents as training data
    ## the remaining documents as development data
    for docIDX, doc in enumerate(tqdm(OriginalData['data'][:360])):
        title = doc['title']
        filteredParas = []
        for paraIDX, para in enumerate(doc['paragraphs']):
            context = para['context']
            filteredQA = []
            for qaIDX, qa in enumerate(para['qas']):
                qid = qa['id']
                if qid in filtered and qid in ParsedInfo:
                    context = ' '.join(ParsedInfo[qid]['tokenized_context'])
                    #info = (docIDX+360, paraIDX, qaIDX, qid)
                    info = (docIDX, paraIDX, qaIDX, qid)
                    # if info in HaveMultipleAnswerNodes or info in CannotFindAnswerNodes:
                    #     continue
                    # else:
                    #     filteredQA += [qa]
                    #     remainedNum += 1
                    filteredQA += [qa]
                    remainedNum += 1
            if filteredQA != []:
                filteredPara = {}
                filteredPara['context'] = context
                filteredPara['qas'] = filteredQA
                filteredParas += [filteredPara]
        if filteredParas != []:
            filteredDoc = {}
            filteredDoc['title'] = title
            filteredDoc['paragraphs'] = filteredParas
            FilteredData['data'] += [filteredDoc]

    with open(args.save_file,'w') as f:
        json.dump(FilteredData,f)

    print('Remain {} samples'.format(remainedNum))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cannot_find_answer_nodes',type=str, default='./cannot_find_answer_nodes_train_reduced.pkl')
    parser.add_argument('--have_multiple_answer_nodes', type=str, default='./have_multiple_answer_nodes_train_reduced.pkl')
    parser.add_argument('--parsed_info', type=str, default='./parsed_info_originaltrain.pkl')
    parser.add_argument('--squad_version','-v', default='1.1', type=str, help='1.1 or 2.0')
    parser.add_argument('--original_data', type=str, default='../data/train-v1.1-modified.json')
    parser.add_argument('--save_file', type=str, default='../data/train-v1.1-filtered.json')

    args = parser.parse_args()

    main(args)
