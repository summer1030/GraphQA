
"""
*******************************************************************

Model w

*******************************************************************
"""

import torch_scatter
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel, BertForQuestionAnswering
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv, GATConv
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import numpy as np

from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, FastRGCNConv, GATConv, HGTConv, Linear

import pickle
import joblib

class BERT_HGT(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, num_dims, num_edge_types, num_gnn_layers=2, num_heads=4, graph_dir='../data/squad_files/constituency_graphs'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.graph_dir = graph_dir

        print("num dims {}".format(num_dims))
        self.bert = BertModel(config, add_pooling_layer=False)

        #initialize a graph
        graph_data = HeteroData()
        graph_data['token'].x = None
        graph_data['virtual'].x = None
        graph_data['token','connects','token'].edge_index = None
        graph_data['token','belongs','virtual'].edge_index = None
        graph_data['virtual','consists','virtual'].edge_index = None

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph_data.node_types:
            self.lin_dict[node_type] = Linear(-1, config.hidden_size)

        self.num_gnn_layers = num_gnn_layers
        self.num_heads = num_heads
        self.gnn_convs = torch.nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = HGTConv(config.hidden_size, config.hidden_size, graph_data.metadata(),
                           self.num_heads, group='sum')
            self.gnn_convs.append(conv)

        self.qa_outputs = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_virtual_node_embeds(self, qas_ids):
        virtual_node_embeds = []
        for each in qas_ids:
            virtual_node_embeds += [pickle.load(open('{}/{}'.format(self.graph_dir,each),'rb'))['virtual_node_embeds']]
        return torch.cat(virtual_node_embeds,dim=0)

    def add_word_token_embeds(self, bert_embeds, mapping):
        added_node_embeds = []
        for j in range(mapping.size()[1]):
            start_token_pos, end_token_pos = mapping[1,j],mapping[2,j]
            added_node_embeds += [torch.mean(bert_embeds[start_token_pos:end_token_pos],dim=0)]
        added_node_embeds = torch.stack(added_node_embeds)

        return torch.cat((bert_embeds,added_node_embeds),dim=0)

    def recover_logits(self, logits, mapping, sent_start_index):
        updated_logits = []
        start_pos = 0
        # updated_logits += [logits[:384,:]]
        # for i,item in enumerate(sent_start_index[1:]):
        #     if mapping[0,sent_start_index[i]]== 0:
        #         start_pos = end_pos
        #     else:
        #         start_pos = i*384+item
        #     end_pos = start_pos+384
        #     print(start_pos,end_pos)
        #     updated_logits +=[logits[start_pos:end_pos,:].unsqueeze(0)]
        # print(logits.size())
        count=0
        while count<len(sent_start_index):
            end_pos=start_pos+384
            updated_logits += [logits[start_pos:end_pos,:].unsqueeze(0)]
            if mapping[0,sent_start_index[count]] == 0:
                start_pos = end_pos
            else:
                try:
                    start_pos = end_pos+sent_start_index[count+1]-sent_start_index[count]
                except:
                    pass
            count+=1

        try:
            return torch.cat(updated_logits,dim=0)
        except:
            import pdb;pdb.set_trace()

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

        graph_data['token'].x = initial_reps
        graph_data['virtual'].x = self.get_virtual_node_embeds(data.qas_id).to(input_ids.device)
        graph_data['token','connects','token'].edge_index = data.t2t_edge_index
        graph_data['token','belongs','virtual'].edge_index = data.t2v_edge_index
        graph_data['virtual','consists','virtual'].edge_index = data.v2v_edge_index


        self.lin_dict.cuda()

        for node_type, x in graph_data.x_dict.items():
            graph_data.x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        #graph_data.x_dict = F.relu(self.conv1(graph_data.x_dict, graph_data.edge_index_dict))
        #graph_data.x_dict = F.relu(self.conv2(graph_data.x_dict, graph_data.edge_index_dict ))

        for conv in self.gnn_convs:
            graph_data.x_dict = conv(graph_data.x_dict, graph_data.edge_index_dict)
        logits = self.qa_outputs(graph_data.x_dict['token'])

        updated_logits = self.recover_logits(logits, data.added_node_mapping, all_sent_start_index)

        start_logits, end_logits = updated_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
