from torch_geometric.data import Data

class SquadData(Data):
    def __cat_dim__(self, key, item):
        # cls_idx and feature_idx are named like this because
        # "Any attribute that is named *index will automatically increase its value based on the cumsum of nodes"
        # https://github.com/rusty1s/pytorch_geometric/issues/2052
        # the return None is so that we can batch along new dimensions, so they won't return num_examples * num_features
        # i.e. remains 8*384 instead of 3072
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html#batching-along-new-dimensions
        if key == 'input_ids':
            return None
        if key == 'attention_mask':
            return None
        if key == 'token_type_ids':
            return None
        if key == 'feature_idx':
            return None
        if key == 'cls_idx':
            return None
        if key == 'p_mask':
            return None
        if key == 'token_mapping':
            return None
        if key == 'num_tokens':
            return None
        if key == 'start_position': # comment out and change to start_index so that it increments with batching properly
            return None
        if key == 'end_position':
            return None
        if key == 'pos_tags':
            return None
        else:
            return super().__cat_dim__(key, item)
