# coding: utf-8

import torch
import torch.nn as nn
from copy import deepcopy

class TreeNode:
    """Class to represent a node in the expression tree"""
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class TreeEmbedding:
    """Class to hold tree embeddings with terminal state indicator"""
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


class Score(nn.Module):
    """Scoring mechanism for tree-based operations"""
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        
        # Repeat hidden across all positions
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        
        # Calculate scores
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        
        # Reshape
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        
        # Apply mask if provided
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
            
        return score


class Prediction(nn.Module):
    """Prediction module with problem-aware dynamic encoding"""
    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        self.dropout = nn.Dropout(dropout)
        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # Linear transformations for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        # Operation prediction layer
        self.ops = nn.Linear(hidden_size * 2, op_nums)

        # Attention mechanisms
        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        current_embeddings = []

        # Get embeddings from the top of each stack or use padding
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        # Process node embeddings with left child information
        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)
        current_embeddings = self.dropout(current_node)

        # Apply attention to get context
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # Prepare number embeddings
        batch_size = current_embeddings.size(0)
        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        # Prepare leaf input
        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # Score numbers
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # Predict operation
        op = self.ops(leaf_input)

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    """Module to generate nodes in the expression tree"""
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Embedding layer for operator tokens
        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        
        # Linear transformations for node generation
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        # Embed operator label
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        
        # Prepare inputs
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        # Generate left and right children with gating
        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        
        # Apply gating mechanism
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        
        return l_child, r_child, node_label_


class Merge(nn.Module):
    """Module to merge subtrees in the expression tree"""
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        # Apply dropout to inputs
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        # Merge with gating mechanism
        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        
        return sub_tree


class TreeBeam:
    """Class to store beam search states for tree decoding"""
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        
        # Create safe copies of the components
        self.node_stack = self._copy_node_stack(node_stack)
        self.embedding_stack = self._copy_embedding_stack(embedding_stack)
        self.left_childs = self._copy_left_childs(left_childs)
        self.out = out.copy() if out else []  # Simple list copy
    
    def _copy_node_stack(self, node_stack):
        """Safe copy of node_stack that handles Tensors properly"""
        copied_stack = []
        for stack in node_stack:
            new_stack = []
            for node in stack:
                if node is None:
                    new_stack.append(None)
                else:
                    # Create a new node with a clone of the embedding tensor
                    if hasattr(node, 'embedding') and torch.is_tensor(node.embedding):
                        new_node = TreeNode(node.embedding.clone().detach(), node.left_flag)
                    else:
                        new_node = TreeNode(node.embedding, node.left_flag)
                    new_stack.append(new_node)
            copied_stack.append(new_stack)
        return copied_stack
    
    def _copy_embedding_stack(self, embedding_stack):
        """Safe copy of embedding_stack that handles Tensors properly"""
        copied_stack = []
        for stack in embedding_stack:
            new_stack = []
            for embed in stack:
                if embed is None:
                    new_stack.append(None)
                else:
                    # Create a new TreeEmbedding with a clone of the embedding tensor
                    if hasattr(embed, 'embedding') and torch.is_tensor(embed.embedding):
                        new_embed = TreeEmbedding(embed.embedding.clone().detach(), embed.terminal)
                    else:
                        new_embed = TreeEmbedding(embed.embedding, embed.terminal)
                    new_stack.append(new_embed)
            copied_stack.append(new_stack)
        return copied_stack
    
    def _copy_left_childs(self, left_childs):
        """Safe copy of left_childs that handles Tensors properly"""
        new_left_childs = []
        for child in left_childs:
            if child is None:
                new_left_childs.append(None)
            elif torch.is_tensor(child):
                new_left_childs.append(child.clone().detach())
            else:
                new_left_childs.append(child)
        return new_left_childs


# Import TreeAttn here to avoid circular imports
from models.basic_models import TreeAttn