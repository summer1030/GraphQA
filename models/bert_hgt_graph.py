
"""
*******************************************************************

Do the graph prediciction.
Build constituency graphs on top of a transformer-based backbone, and
then predict the results among all the nodes.

*******************************************************************
"""


import numpy as np
from collections import defaultdict

import torch_scatter
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel, BertForQuestionAnswering
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv, GATConv
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, FastRGCNConv, GATConv, HGTConv, Linear
from torch_geometric.utils import accuracy
from torch_geometric.utils import to_undirected

import pickle
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERT_HGT(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, num_dims, num_edge_types, use_crossentropy=True):
        super().__init__(config)
        self.num_labels = config.num_labels
        print("num dims {}".format(num_dims))

        # build the backbone
        self.bert = BertModel(config, add_pooling_layer=False)
        graph_data = HeteroData()
        graph_data['token'].x = None
        graph_data['virtual'].x = None
        graph_data['token','connects','token'].edge_index = None
        graph_data['token','belongs','virtual'].edge_index = None
        graph_data['virtual','consists','virtual'].edge_index = None
        graph_data['virtual'].y = None

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph_data.node_types:
            self.lin_dict[node_type] = Linear(-1, config.hidden_size)

        self.num_gnn_layers = 2
        self.num_heads = 4
        self.gnn_convs = torch.nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = HGTConv(config.hidden_size, config.hidden_size, graph_data.metadata(),
                           self.num_heads, group='sum')
            self.gnn_convs.append(conv)

        self.use_crossentropy = use_crossentropy
        self.threshold = 0.5
        self.class_weights = torch.Tensor([1,10]).to(device)
        if self.use_crossentropy:
            self.qa_outputs = torch.nn.Linear(config.hidden_size, config.num_labels)
            self.loss_fct = CrossEntropyLoss(weight=self.class_weights)
        else:
            self.sigmoid = nn.Sigmoid()
            self.qa_outputs = torch.nn.Linear(config.hidden_size,1)
            self.loss_fct = nn.BCELoss()

        self.init_weights()


    def generate_virtual_node_embeds(virtual_node_mapping, bert_embeds):
        embeds=[]
        lookup=defaultdict(list)

        for item in virtual_node_mapping:
            lookup[item[0]].append(item[1])
        for i in range(max(lookup.keys())+1):
            if lookup[i]:
                tmp=[]
                for idx in lookup[i]:
                    tmp+=[bert_embeds[idx]]
                tmp = torch.mean(torch.stack(tmp),dim=0)
                embeds += [tmp]
            else:
                pad = torch.zeros(768)
                embeds += [pad]
        return embeds

    def get_virtual_node_info(self, qas_ids, bert_embeds):

        virtual_node_embeds = []
        node_labels = []
        passage_labels = []
        lens_info = []
        for i,each in enumerate(qas_ids):
            data = pickle.load(open('../data/squad_files/constituency_graphs_reduced/{}'.format(each),'rb'))
            virtual_node_mapping = data['virtual_node_embeds_mapping']
            virtual_node_embeds += self.generate_virtual_node_embeds(virtual_node_mapping, bert_embeds[i])
            node_labels += [data['node_label']]
            lens_info += [data['node_label'].size()[0]]
            passage_labels += [data['node_label']]
        return torch.stack(virtual_node_embeds,dim=0),torch.cat(node_labels,dim=0),torch.tensor(lens_info),passage_labels

    def add_word_token_embeds(self, bert_embeds, mapping):
        added_node_embeds = []
        for j in range(mapping.size()[1]):
            start_token_pos, end_token_pos = mapping[1,j],mapping[2,j]
            added_node_embeds += [torch.mean(bert_embeds[start_token_pos:end_token_pos],dim=0)]
        added_node_embeds = torch.stack(added_node_embeds)

        return torch.cat((bert_embeds,added_node_embeds),dim=0)

    def compute_passage_acc(self, preds, lens_info, passage_labels):
        updated_preds = []
        start_pos = torch.tensor(0)
        correct_num = 0
        for l in lens_info:
            end_pos = start_pos+l
            updated_preds += [preds[start_pos:end_pos]]
            start_pos = end_pos
        for i,pred in enumerate(updated_preds):
            label = passage_labels[i].to(device)
            # if (pred == label).sum().item() == label.size()[0]:
            #     correct_num += 1
            if torch.argmax(label).item() == torch.argmax(pred).item():
                correct_num += 1
        return correct_num/len(passage_labels),updated_preds

    def forward(self, data,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids = data.input_ids
        attention_mask = data.attention_mask
        token_type_ids = data.token_type_ids
        # print("in forward, self.use_pos = {}".format(self.use_pos))
        num_nodes = data.num_nodes

        if hasattr(data, 'start_position'):
            start_positions = data.start_position
            end_positions = data.end_position
        else:
            start_positions = None
            end_positions = None
        #num_tokens = data.num_tokens
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        graph_data = HeteroData()

        """
        initial_reps_list = []
        for i, sample_output in enumerate(sequence_output):
            initial_reps_list.append(sample_output)
        initial_reps = torch.cat(initial_reps_list)
        graph_data['token'].x = initial_reps
        graph_data['token','connects','token'].edge_index = edge_index
        """

        sent_start_index=(data.added_node_mapping[0,:]==384).nonzero()
        passed_sent_index=(data.added_node_mapping[0,:]==0).nonzero()
        all_sent_start_index = sorted(torch.cat((sent_start_index,passed_sent_index),0))

        initial_reps_list = []
        for i, sample_output in enumerate(sequence_output):
            if data.added_node_mapping[0,all_sent_start_index[i]] != 0:
                if i<= len(all_sent_start_index)-2:
                    tmp=self.add_word_token_embeds(sample_output,data.added_node_mapping[:,all_sent_start_index[i]:all_sent_start_index[i+1]])
                    initial_reps_list.append(tmp)
                elif i == len(all_sent_start_index)-1:
                    tmp=self.add_word_token_embeds(sample_output,data.added_node_mapping[:,all_sent_start_index[i]:])
                    initial_reps_list.append(tmp)
            else:
                initial_reps_list.append(sample_output)

        initial_reps = torch.cat(initial_reps_list)

        #virtual_nodes_info = (virtual_nodes_embeds, virtual_nodes_labels, lens_info, passage_labels)
        virtual_nodes_info = self.get_virtual_node_info(data.qas_id, initial_reps_list)

        graph_data['token'].x = initial_reps
        graph_data['virtual'].x = virtual_nodes_info[0].to(input_ids.device)
        graph_data['token','connects','token'].edge_index = to_undirected(data.t2t_edge_index)
        graph_data['token','belongs','virtual'].edge_index = to_undirected(data.t2v_edge_index)
        graph_data['virtual','consists','virtual'].edge_index = to_undirected(data.v2v_edge_index)
        graph_data['virtual'].y = virtual_nodes_info[1].to(input_ids.device)

        self.lin_dict.to(device)

        for node_type, x in graph_data.x_dict.items():
            graph_data.x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        #graph_data.x_dict = F.relu(self.conv1(graph_data.x_dict, graph_data.edge_index_dict))
        #graph_data.x_dict = F.relu(self.conv2(graph_data.x_dict, graph_data.edge_index_dict ))

        for conv in self.gnn_convs:
            graph_data.x_dict = conv(graph_data.x_dict, graph_data.edge_index_dict)

        #logits = self.qa_outputs(graph_data.x_dict['token']) #classicial qa
        logits = self.qa_outputs(graph_data.x_dict['virtual'])

        if self.use_crossentropy:
            loss = self.loss_fct(logits, graph_data['virtual'].y.long())
            preds = torch.argmax(logits,dim=1)
            #node_acc = accuracy(preds,graph_data['virtual'].y.long())
        else:
            logits = self.sigmoid(logits).squeeze(-1)
            import pdb;pdb.set_trace()
            loss = self.loss_fct(logits, graph_data['virtual'].y)
            preds = logits.ge(self.threshold).float()
            #node_acc = (preds == graph_data['virtual'].y).sum().item()/logits.size()[0]

        passage_acc,preds = self.compute_passage_acc(logits, virtual_nodes_info[2].to(device), virtual_nodes_info[3])
        return (loss,preds,passage_acc)
