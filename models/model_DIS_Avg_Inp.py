# -*- coding: utf-8 -*-

# Copyright 2020 Sungho Jeon and Heidelberg Institute for Theoretical Studies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permemoryssions and
# limemorytations under the License.

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import logging
import math

import networkx as nx
import collections

from models.encoders.encoder_main import Encoder_Main
from models.transformer.encoder import TransformerInterEncoder

import models.model_base
from models.model_base import masked_softmax

import utils
from utils import FLOAT, LONG, BOOL

import models.stru_trans.attention as tt_attn
import models.stru_trans.models as tt_model
import models.stru_trans.modules as tt_module
import copy

import models.tree_trans.attention as tree_attn
import models.tree_trans.models as tree_model
import models.tree_trans.modules as tree_module

import fairseq.modules as fairseq

# from apex.normalization.fused_layer_norm import FusedLayerNorm

logger = logging.getLogger()


class Coh_Model_DIS_Avg_Inp(models.model_base.BaseModel):
    def __init__(self, config, corpus_target, embReader):
        super().__init__(config)

        ####
        # init parameters
        self.corpus_target = config.corpus_target
        self.max_num_sents = config.max_num_sents  # document length, in terms of the number of sentences
        self.max_len_sent = config.max_len_sent  # sentence length, in terms of words
        self.max_len_doc = config.max_len_doc  # document length, in terms of words
        self.avg_num_sents = config.avg_num_sents
        self.batch_size = config.batch_size

        self.avg_len_doc = config.avg_len_doc

        self.vocab = corpus_target.vocab  # word2id
        self.rev_vocab = corpus_target.rev_vocab  # id2word
        self.pad_id = corpus_target.pad_id
        self.num_special_vocab = corpus_target.num_special_vocab

        self.dropout_rate = config.dropout
        self.output_size = config.output_size  # the number of final output class

        self.use_gpu = config.use_gpu
        self.gen_logs = config.gen_logs

        if not hasattr(config, "freeze_step"):
            config.freeze_step = 5000

        self.output_attentions = config.output_attentions  # flag for huggingface impl

        self.topk_fwr = config.topk_fwr
        self.threshold_sim = config.threshold_sim
        # self.topk_back = config.topk_back
        self.topk_back = 1
        self.use_np_focus = config.use_np_focus

        # np information test
        self.map_np_sbw_total = config.map_np_sbw_total
        self.max_num_np = config.max_num_np

        ########
        #
        self.base_encoder = Encoder_Main(config, embReader)

        #
        self.sim_cosine_d0 = torch.nn.CosineSimilarity(dim=0)
        self.sim_cosine_d1 = torch.nn.CosineSimilarity(dim=1)
        self.sim_cosine_d2 = torch.nn.CosineSimilarity(dim=2)

        ## tree-transformer
        c = copy.deepcopy
        num_heads = 4
        N=4  # num of layers
        d_model=self.base_encoder.encoder_out_size
        d_ff=self.base_encoder.encoder_out_size 
        dropout=self.dropout_rate
        
        attn = tt_attn.MultiHeadedAttention(num_heads, d_model)
        group_attn = tt_attn.GroupAttention(d_model)
        ff = tt_module.PositionwiseFeedForward(d_model, d_ff, dropout)
        position = tt_module.PositionalEncoding(d_model, dropout)
        # word_embed = nn.Sequential(Embeddings(d_model, vocab_size), c(position))
        self.tt_encoder = tt_model.Encoder(tt_model.EncoderLayer(d_model, c(attn), c(ff), group_attn, dropout), 
                N, d_model, dropout)  # we do not need an embedding layer here
        
        for p in self.tt_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
        if self.use_gpu:
            self.tt_encoder.cuda()

        ###
        # attn_tree = tree_attn.MultiHeadedAttention(num_heads, d_model)
        # group_attn = tree_attn.GroupAttention(d_model)
        # ff_tree = tree_module.PositionwiseFeedForward(d_model, d_ff, dropout)
        # position = tree_module.PositionalEncoding(d_model, dropout)
        # # word_embed = nn.Sequential(Embeddings(d_model, vocab_size), c(position))
        # self.tree_encoder = tree_model.Encoder(tree_model.EncoderLayer(d_model, c(attn_tree), c(ff_tree), group_attn, dropout), 
        #         N, d_model, dropout)  # we do not need an embedding layer here
        
        # for p in self.tree_encoder.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
    
        # if self.use_gpu:
        #     self.tree_encoder.cuda()

        ###
        vocab_size_ctrans = 5

        num_heads = 4
        N = 4
        attn_tree2 = tree_attn.MultiHeadedAttention(num_heads, d_model)
        group_attn2 = tree_attn.GroupAttention(d_model)
        ff_tree2 = tree_module.PositionwiseFeedForward(d_model, d_ff, dropout)
        position2 = tree_module.PositionalEncoding(d_model, dropout)
        # self.embed_ctrans = nn.Sequential(tree_module.Embeddings(d_model, vocab_size_ctrans), c(position2))
        self.tree_encoder2 = tree_model.Encoder(tree_model.EncoderLayer(d_model, c(attn_tree2), c(ff_tree2), group_attn2, dropout), 
                N, d_model, dropout)  # we do not need an embedding layer here
        
        for p in self.tree_encoder2.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
        if self.use_gpu:
            self.tree_encoder2.cuda()

        ###

        #        
        # self.context_weight = nn.Parameter(torch.zeros(self.base_encoder.encoder_out_size,1))
        self.context_weight = nn.Parameter(torch.zeros(self.base_encoder.encoder_out_size,1))
        nn.init.xavier_uniform_(self.context_weight)

        self.context_weight_2 = nn.Parameter(torch.zeros(self.base_encoder.encoder_out_size,1))
        nn.init.xavier_uniform_(self.context_weight_2)

        self.context_weight_3 = nn.Parameter(torch.zeros(self.base_encoder.encoder_out_size,1))
        nn.init.xavier_uniform_(self.context_weight_3)

        self.context_weight_4 = nn.Parameter(torch.zeros(self.max_num_sents-1, 1))
        nn.init.xavier_uniform_(self.context_weight_4)

        self.context_weight_5 = nn.Parameter(torch.zeros(self.max_num_sents-1, 1))
        nn.init.xavier_uniform_(self.context_weight_5)

        self.context_weight_unified = nn.Parameter(torch.zeros(self.base_encoder.encoder_out_size * 2, 1))
        nn.init.xavier_uniform_(self.context_weight_unified)

        #### modeling sequence of centring transitions
        ## embedding centering transitions
        
        # self.embed_ctrans = torch.nn.Embedding(vocab_size_ctrans, d_model)  # 0 is pad
        # nn.init.xavier_uniform_(self.embed_ctrans.weight)

        # vocab = set(test_sentence)  # it memoryght not need if we directly assign index to transitions
        # word_to_ix = {word: i for i, word in enumerate(vocab)}

        # ## baseline: naive-transformer
        ff_size = self.base_encoder.encoder_out_size  # or 512?
        # ff_size = self.encoder_coh.encoder_out_size * 4  # or 512?
        num_heads = 4
        ds_do = 0.1
        inter_layers = 4
        # self.encoder_naive = TransformerInterEncoder(d_model=d_model,
        #                                             d_ff=ff_size,
        #                                             heads=num_heads, 
        #                                             dropout=ds_do, 
        #                                             num_inter_layers=inter_layers)

        # self.encoder_naive2 = TransformerInterEncoder(d_model=d_model*3,
        #                                             d_ff=ff_size*3,
        #                                             heads=num_heads, 
        #                                             dropout=ds_do, 
        #                                             num_inter_layers=inter_layers)

        # self.encoder_ctrans = CoAttnTrans(config, 
        #     num_heads=num_heads, num_layer=N, d_model= d_model, d_ff=d_ff, dropout=dropout)

        self.attn_concat = nn.Linear(d_model*2, d_model)

        # self.encoder_mem_trans = MemAttnTrans(config, 
            # num_heads=num_heads, num_layer=N, d_model= d_model, d_ff=d_ff, dropout=dropout)




        # self.encoder_mem_trans = MemUniAttnTrans(config, 
            # num_heads=num_heads, num_layer=N, d_model= d_model, d_ff=d_ff, dropout=dropout)

        

        
        ## skip pair idea
        self.size_max_pool = 4
        # self.size_max_pool_1d = 4
        self.size_max_pool_1d = self.base_encoder.encoder_out_size // 4

        # self.num_conv_nb = 7
        # self.conv_nb_list = nn.ModuleList(
        #     [nn.Conv1d(in_channels=1,out_channels=1,kernel_size=3,stride=2,padding=1,
        #               dilation=1,groups=1,bias=True)
        #      for _ in range(self.num_conv_nb)])

        ## memory transformer with NP repr idea
        # self.mem_intra_sent = GatedMemoryUpdate(self.base_encoder.encoder_out_size)  # Memory between NPs within sentence
        # self.mem_inter_sent = GatedMemoryUpdate(self.base_encoder.encoder_out_size)  # Memory between sentences

        #####################
        fc_in_size = self.base_encoder.encoder_out_size
        # fc_in_size = self.base_encoder.encoder_out_size * 2
        # fc_in_size = self.base_encoder.encoder_out_size * 3
        # fc_in_size = self.base_encoder.encoder_out_size  + (self.size_max_pool**2)
        # fc_in_size = self.base_encoder.encoder_out_size  + self.size_max_pool_1d
        # fc_in_size = self.base_encoder.encoder_out_size  + self.max_num_sents-1
        # fc_in_size = self.base_encoder.encoder_out_size  + (self.size_max_pool_1d*self.num_conv_nb)

        # fc_in_size = self.base_encoder.encoder_out_size  + (self.size_max_pool**2) + self.size_max_pool_1d
        # fc_in_size = self.base_encoder.encoder_out_size * 2 + (self.size_max_pool**2)

        linear_1_out = fc_in_size // 2
        linear_2_out = linear_1_out // 2

        self.linear_1 = nn.Linear(fc_in_size, linear_1_out)
        nn.init.xavier_uniform_(self.linear_1.weight)

        self.linear_2 = nn.Linear(linear_1_out, linear_2_out)
        nn.init.xavier_uniform_(self.linear_2.weight)

        self.linear_out = nn.Linear(linear_2_out, self.output_size)
        if corpus_target.output_bias is not None:  # bias
            init_mean_val = np.expand_dims(corpus_target.output_bias, axis=1)
            bias_val = (np.log(init_mean_val) - np.log(1 - init_mean_val))
            self.linear_out.bias.data = torch.from_numpy(bias_val).type(torch.FloatTensor)
        nn.init.xavier_uniform_(self.linear_out.weight)

        #

        # self.conv_2d_np = nn.Conv2d(in_channels=1,
        #               out_channels=1,
        #               kernel_size=3,
        #               stride=2,
        #               padding=1,
        #               dilation=1,
        #               groups=1,
        #               bias=True)

        # self.conv_2d_np2 = nn.Conv2d(in_channels=1,
        #               out_channels=1,
        #               kernel_size=5,
        #               stride=2,
        #               padding=1,
        #               dilation=1,
        #               groups=1,
        #               bias=True)

        # self.conv_1d_np = nn.Conv1d(in_channels=1,
        #               out_channels=1,
        #               kernel_size=3,
        #               stride=2,
        #               padding=1,
        #               dilation=1,
        #               groups=1,
        #               bias=True)

        # self.conv_1d_np2 = nn.Conv1d(in_channels=1,
        #               out_channels=1,
        #               kernel_size=5,
        #               stride=2,
        #               padding=1,
        #               dilation=1,
        #               groups=1,
        #               bias=True)

        # self.max_adapt_pool1 = nn.AdaptiveMaxPool2d(self.size_max_pool)

        # self.max_adapt_pool1d = nn.AdaptiveMaxPool1d(self.size_max_pool_1d)

        # # depthwise conv from ICLR19
        # self.dconv1 = fairseq.LightweightConv1dTBC(input_size=self.base_encoder.encoder_out_size, 
        #                     kernel_size=5,
        #                     padding_l=2,
        #                     num_heads=4,
        #                     weight_dropout=0.1,
        #                     weight_softmax=False)

        # self.dconv2 = fairseq.LightweightConv1dTBC(input_size=self.base_encoder.encoder_out_size, 
        #                     kernel_size=3,
        #                     padding_l=2,
        #                     num_heads=4,
        #                     weight_dropout=0.1,
        #                     weight_softmax=False)

        # self.dyconv1 = fairseq.DynamicConv1dTBC(input_size=self.base_encoder.encoder_out_size, 
        #                     kernel_size=5,
        #                     padding_l=2,
        #                     num_heads=4,
        #                     weight_dropout=0.1,
        #                     weight_softmax=False)

        # self.dyconv2 = fairseq.DynamicConv1dTBC(input_size=self.base_encoder.encoder_out_size, 
        #                     kernel_size=3,
        #                     padding_l=2,
        #                     num_heads=4,
        #                     weight_dropout=0.1,
        #                     weight_softmax=False)

        # # Dynamic conv1d from CVPR20
        # self.ms_dconv1 = Dynamic_conv1d(
        #                     in_planes=self.max_num_sents-1,
        #                     out_planes=self.max_num_sents-1,
        #                     kernel_size=3, ratio=0.25, padding=1)

        #
        self.selu = nn.SELU()
        self.elu = nn.ELU()
        self.leak_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.dropout_01 = nn.Dropout(0.1)
        self.dropout_02 = nn.Dropout(0.2)

        self.softmax_d0 = nn.Softmax(dim=0)
        self.softmax = nn.Softmax(dim=1)
        self.softmax_last = nn.Softmax(dim=-1)

        # self.layer_norm1 = nn.LayerNorm(linear_1_out, eps=1e-6)
        # self.layer_norm2 = nn.LayerNorm(linear_2_out, eps=1e-6)

        return
    # end __init__

#### Functions
#####################################################################################

    def sent_repr_avg(self, batch_size, encoder_out, len_sents):
        """return sentence representation by averaging of all words."""

        sent_mask = torch.sign(len_sents)  # (batch_size, len_sent)
        num_sents = sent_mask.sum(dim=1)  # (batch_size)

        sent_repr = torch.zeros(batch_size, self.max_num_sents, self.base_encoder.encoder_out_size)
        sent_repr = utils.cast_type(sent_repr, FLOAT, self.use_gpu)
        for cur_ind_doc in range(batch_size):
            list_sent_len = len_sents[cur_ind_doc]
            cur_sent_num = int(num_sents[cur_ind_doc])
            cur_loc_sent = 0
            list_cur_doc_sents = []

            for cur_ind_sent in range(cur_sent_num):
                cur_sent_len = int(list_sent_len[cur_ind_sent])
                cur_sent_lens = cur_sent_lens + 1e-9 # prevent zero division
                
                cur_sent_repr = torch.div(torch.sum(encoder_out[cur_ind_doc, cur_loc_sent:cur_loc_sent+cur_sent_len], dim=0), cur_sent_len)  # avg version
                cur_sent_repr = cur_sent_repr.view(1, 1, -1)  # restore to (1, 1, xrnn_cell_size)
                
                list_cur_doc_sents.append(cur_sent_repr)
                cur_loc_sent = cur_loc_sent + cur_sent_len

            # end for cur_len_sent

            cur_sents_repr = torch.stack(list_cur_doc_sents, dim=1)  # (batch_size, num_sents, rnn_cell_size)
            cur_sents_repr = cur_sents_repr.squeeze(2)  # not needed when the last repr is used

            sent_repr[cur_ind_doc, :cur_sent_num, :] = cur_sents_repr
        # end for cur_doc

        return sent_repr
    # end def sent_repr_avg

    def get_fwrd_centers_np_enc2(self, text_inputs, mask_input, len_sents, num_sents, tid):
        """ Determemoryne fowrard-looking centers using an attention matrix in a PLM """
        """ This version encodes two sentences to take the benefit of correference resolution in a PLM """
        """ It interpolates represenattions and attentions from two encoded sentences"""
        batch_size = text_inputs.size(0)

        # prepare the np information for given batch documents
        num_np = []  ## not used now
        batch_map_np = []  # list of list of map
        for cur_tid in tid:
            try:
                cur_tid = str(int(cur_tid))
            except ValueError:
                cur_tid = str(cur_tid)

            list_doc_map_np = self.map_np_sbw_total[cur_tid]  # list of map_np
            # padding with "None"
            list_doc_map_np += [None] * (self.max_num_sents - len(list_doc_map_np))

            num_np.append(len(list_doc_map_np))  # the number of NPs at each sentence
            batch_map_np.append(list_doc_map_np)  # the index of NP at each sentence
        # end for

        #### encoding sentences
        fwrd_repr = torch.zeros(batch_size, self.max_num_sents, self.topk_fwr, self.base_encoder.encoder_out_size)
        fwrd_repr = utils.cast_type(fwrd_repr, FLOAT, self.use_gpu)

        avg_sents_repr = torch.zeros(batch_size, self.max_num_sents, self.base_encoder.encoder_out_size)  # averaged sents repr in the sent level encoding
        avg_sents_repr = utils.cast_type(avg_sents_repr, FLOAT, self.use_gpu)

        # batch_cp_ind = torch.zeros(batch_size, self.max_num_sents)  # only used for manual analysis later
        # batch_cp_ind = utils.cast_type(batch_cp_ind, LONG, self.use_gpu)

        # when cp analysis for np focus
        batch_cp_nps = [[None for _ in range(self.max_num_sents)] for _ in range(batch_size)]  # 2d string list of NPs

        # when cp analysis for non-np focus
        batch_cp_ind = torch.zeros(batch_size, self.max_num_sents)  # only used for manual analysis later
        batch_cp_ind = utils.cast_type(batch_cp_ind, LONG, self.use_gpu)

        cur_ind = torch.zeros(batch_size, dtype=torch.int64)
        cur_ind = utils.cast_type(cur_ind, LONG, self.use_gpu)
        len_sents = utils.cast_type(len_sents, LONG, self.use_gpu)

        # np ver: need to make a new represenations only includes NPs
        np_avg_repr = torch.zeros(batch_size, self.max_num_sents, self.base_encoder.encoder_out_size)
        np_avg_repr = utils.cast_type(np_avg_repr, FLOAT, self.use_gpu)

        np_all_repr = torch.zeros(batch_size, self.max_num_sents, self.max_num_np, self.base_encoder.encoder_out_size)
        np_all_repr = utils.cast_type(np_all_repr, FLOAT, self.use_gpu)
        num_np_sents = torch.zeros(batch_size, self.max_num_sents)
        num_np_sents = utils.cast_type(num_np_sents, LONG, self.use_gpu)

        #
        prev_repr = [None] * batch_size
        prev_attn = [None] * batch_size
        for sent_i in range(self.max_num_sents):
            cur_sent_lens = len_sents[:, sent_i]
            cur_max_len = int(torch.max(cur_sent_lens))

            ## early stop, if there is no actual sentence in this batch anymore
            if cur_max_len < 1:
                break

            ##
            cur_sent_ids = torch.zeros(batch_size, cur_max_len, dtype=torch.int64)
            cur_sent_ids = utils.cast_type(cur_sent_ids, LONG, self.use_gpu)
            cur_mask = torch.zeros(batch_size, cur_max_len, dtype=torch.int64)
            cur_mask = utils.cast_type(cur_mask, FLOAT, self.use_gpu)

            prev_ind = cur_ind
            cur_ind = cur_ind + cur_sent_lens

            ## init input ids
            for batch_ind, sent_len in enumerate(cur_sent_lens):
                cur_loc = cur_ind[batch_ind]
                prev_loc = prev_ind[batch_ind]
                if cur_loc-prev_loc > 2:
                    cur_sent_ids[batch_ind, :cur_loc-prev_loc-2] = text_inputs[batch_ind, prev_loc:cur_loc-2]
                    cur_mask[batch_ind, :cur_loc-prev_loc-2] = mask_input[batch_ind, prev_loc:cur_loc-2]
            # end for

            ## concat the ids of the next sentence unless it is the last sentence
            
            if sent_i < self.max_num_sents-1:
                next_sent_lens = len_sents[:, sent_i+1]
            else:
                next_sent_lens = torch.zeros(batch_size, dtype=torch.int64)
                next_sent_lens = utils.cast_type(next_sent_lens, LONG, self.use_gpu)
            # end if            

            # concat the ids with the next ids of sent, only if there is at least one next sentence in the batch
            has_next_sent = next_sent_lens > 0
            next_max_len = int(torch.max(next_sent_lens))
            if next_max_len > 0:
                ## extend the ids tensor size to concat the ids of the next sentence
                next_sent_ids = torch.zeros(batch_size, next_max_len, dtype=torch.int64)
                next_sent_ids = utils.cast_type(next_sent_ids, LONG, self.use_gpu)
                next_mask = torch.zeros(batch_size, next_max_len, dtype=torch.int64)
                next_mask = utils.cast_type(next_mask, FLOAT, self.use_gpu)

                cur_sent_ids = torch.cat((cur_sent_ids, next_sent_ids), dim=1)
                cur_mask = torch.cat((cur_mask, next_mask), dim=1)

                ahead_prev_ind = cur_ind
                ahead_cur_ind = cur_ind + next_sent_lens

                for batch_ind, sent_len in enumerate(next_sent_lens):
                    cur_len = cur_sent_lens[batch_ind] - 2
                    cur_loc = ahead_cur_ind[batch_ind]
                    prev_loc = ahead_prev_ind[batch_ind]
                    if cur_loc-prev_loc > 0:  # concat only when there is a next sentence in this batch
                        cur_sent_ids[batch_ind, cur_len:cur_len+cur_loc-prev_loc] = text_inputs[batch_ind, prev_loc:cur_loc]
                        cur_mask[batch_ind, cur_len:cur_len+cur_loc-prev_loc] = mask_input[batch_ind, prev_loc:cur_loc]
                # end for
            # end if cur_max_len

            #### encode sentences
            cur_encoded = self.base_encoder(cur_sent_ids, cur_mask, cur_sent_lens-2+next_sent_lens)
            encoded_sent = cur_encoded[0]  # encoded output for the current sent (b, seq, dim)
            attn_sent = cur_encoded[1]  # averaged attention for the current sent (b, seq, seq)

            ## interpolate the repr and attn if it is not the first or the last
            # iterate batch to check that whether they are concattnated with the next sentence or not
            # if prev is None, then it is the first sentence
            # if has_next_sent is False, then it is the last sentence
            # otherwise, we interpolate the current sentence with the previous encoded output for the same sentence

            encoded_sent_i = torch.zeros(batch_size, cur_max_len, encoded_sent.size(2), dtype=torch.float32)
            encoded_sent_i = utils.cast_type(encoded_sent_i, FLOAT, self.use_gpu)
            attn_sent_i = torch.zeros(batch_size, cur_max_len, cur_max_len, dtype=torch.float32)
            attn_sent_i = utils.cast_type(attn_sent_i, FLOAT, self.use_gpu)

            for cur_batch in range(batch_size):
                # split encoded vectors to two vectors
                sent1_repr = encoded_sent[cur_batch]  # init by a raw output
                sent1_attn = attn_sent[cur_batch]
                cur_sent_len = cur_sent_lens[cur_batch]

                if has_next_sent[cur_batch]:
                    len_sent1 = cur_sent_lens[cur_batch]
                    len_sent2 = next_sent_lens[cur_batch]

                    # split representations to two sents
                    sent1_repr = encoded_sent[cur_batch, :len_sent1-2]  # cuz two special tokens are missed
                    sent2_repr = encoded_sent[cur_batch, len_sent1-2 : len_sent1-2+len_sent2]

                    # split attention weights to two sents
                    sent1_attn = attn_sent[cur_batch, :len_sent1, :len_sent1]
                    sent1_attn[-2:, -2:] = 0
                    sent2_attn = attn_sent[cur_batch, len_sent1-2:len_sent1-2+len_sent2, len_sent1-2:len_sent1-2+len_sent2]

                    #### interpolate for representations
                    if prev_repr[cur_batch] is not None:
                        pad_special_tokens = torch.zeros(2, sent1_repr.size(1), dtype=torch.float32)
                        pad_special_tokens = utils.cast_type(pad_special_tokens, FLOAT, self.use_gpu)

                        sent1_repr = torch.cat((sent1_repr, pad_special_tokens), dim=0)
                        
                        interpolated = (sent1_repr + prev_repr[cur_batch])
                        sum_len = len_sent1-2+len_sent2
                        interpolated[:sum_len-2] = interpolated[:sum_len-2] / 2  # only interpolate excep the last two tokens

                        sent1_repr = interpolated
                    # end if
                    else:
                        # this is the case of the first and the last sentence
                        cur_sent_len = cur_sent_len-2

                    ##### interpolate for attention weights
                    if prev_attn[cur_batch] is not None:
                        # pad_special_tokens = torch.zeros(2, 2, dtype=torch.float32)
                        # pad_special_tokens = utils.cast_type(pad_special_tokens, FLOAT, self.use_gpu)
                        # sent1_attn = torch.cat((sent1_attn, pad_special_tokens), dim=0)
                        
                        interpolated = (sent1_attn + prev_attn[cur_batch])
                        sum_len = len_sent1-2+len_sent2
                        interpolated[:sum_len-2, :sum_len-2] = interpolated[:sum_len-2, :sum_len-2] / 2  # only interpolate excep the last two tokens

                        sent1_attn = interpolated

                    #### prepare the next encoding
                    prev_repr[cur_batch] = sent2_repr
                    prev_attn[cur_batch] = sent2_attn

                else:
                    prev_repr[cur_batch] = None
                    prev_attn[cur_batch] = None
                # end else

                encoded_sent_i[cur_batch, :cur_sent_len, :] = sent1_repr[:cur_sent_len]
                attn_sent_i[cur_batch, :cur_sent_len, :cur_sent_len] = sent1_attn[:cur_sent_len, :cur_sent_len]

            # end for interpolate

            ### replace it for conveinence, otherwise I can change the names in the below parts
            encoded_sent = encoded_sent_i
            attn_sent = attn_sent_i


            ###


            #### assign sent repr first
            sent_lens_eps = cur_sent_lens + 1e-9 # prevent zero division
            cur_avg_repr = torch.div(torch.sum(encoded_sent, dim=1), sent_lens_eps.unsqueeze(1))
            avg_sents_repr[:, sent_i] = cur_avg_repr



            ## filter out: we do not consider special tokens and punctation as a center; <.>, <sep>, and <cls>
            list_diag = []
            for batch_ind, cur_mat in enumerate(attn_sent):
                cur_diag = torch.diag(cur_mat, diagonal=0)

                ## masking as the length of each sentence
                cur_batch_sent_len = int(cur_sent_lens[batch_ind])  # i th sentence with batch
                if cur_batch_sent_len > 3:
                    cur_diag[cur_batch_sent_len-3:] = -10  # also remove puntation
                else:
                    cur_diag[cur_batch_sent_len-2:] = -10  # only remove the special tokens
                list_diag.append(cur_diag)
            # end for
            attn_diag = torch.stack(list_diag)  # because torch.daig does not support batch (batch_size, cur_max_len)

            #################################################
            #################################################
            """ Select centers using encoded represenations and attetion weights """
            """ encoded_sents, attn_diag """

            # it is always NP version for this model
            selected, batch_cp_nps, np_all_repr, num_np_sents = \
                                     self.select_fwrd_centers_np(sent_i, batch_map_np, encoded_sent, \
                                                                 attn_diag, batch_cp_nps, avg_sents_repr, np_all_repr, num_np_sents)

            fwrd_repr[:, sent_i, :self.topk_fwr] = selected

            # #### Np version

            # ## make NP representations and attn for them
            # # extract np_map for current ith sentence for a given batch, due to the outer loop
            # list_j_np_map = [row[sent_i] for row in batch_map_np]

            # # need to be batched, need to record the nps to track to interpret
            # # t_attn_np = torch.zeros(batch_size, self.max_num_np)
            # t_attn_np = torch.empty(batch_size, self.max_num_np).fill_(-10.)
            # t_attn_np = utils.cast_type(t_attn_np, FLOAT, self.use_gpu)

            # ## make NP representations by averaging
            # for batch_ind, cur_j_np_list in enumerate(list_j_np_map):
            #     # print("TID")
            #     # print(tid[batch_ind])
            #     # NPs at the j sentence
            #     np_ind = 0
            #     list_avg_np_repr = []
            #     if cur_j_np_list is not None:  # if there is a NP at this sentence (j th)
            #         for np_ind, cur_np_pair in enumerate(cur_j_np_list):  # iterate for each NP at this sent
            #             cur_np = cur_np_pair[0]
            #             cur_span = cur_np_pair[1]
            #             s_ind, e_ind = cur_span

            #             # extract repr corresponding to spans, and average them to make NP reprs
            #             np_repr_span = encoded_sent[batch_ind, s_ind:e_ind+1, :]            
            #             # cur_repr_np = torch.div(torch.sum(repr_span, dim=0), e_ind-s_ind+1)
            #             # np_repr[batch_ind, sent_i, np_ind, :] = cur_repr_np

            #             # average attn scores to make NP score
            #             # print(attn_diag[batch_ind, :])
            #             attn_score_np = attn_diag[batch_ind, s_ind:e_ind+1]
            #             # print(attn_score_np)
            #             attn_score_np = torch.div(torch.sum(attn_score_np, dim=0), e_ind-s_ind+1)
            #             # print(attn_score_np)
            #             t_attn_np[batch_ind, np_ind] = attn_score_np

            #             list_avg_np_repr.append(np_repr_span)

            #             # assign np all repr directly to use it out of this module
            #             avg_np_span = torch.div(torch.sum(np_repr_span, dim=0), np_repr_span.size(1))
            #             np_all_repr[batch_ind, sent_i, np_ind] = avg_np_span
            #         # end for

            #         # num_np_sent.append(len(cur_j_np_list))  # add the number of NPs at this sent

            #         ## avgerage NPs for this sentence
            #         if len(list_avg_np_repr) > 0:
            #             cur_np_repr = torch.cat(list_avg_np_repr, dim=0)
            #             cur_avg_repr = torch.div(torch.sum(cur_np_repr, dim=0), cur_np_repr.size(1))
            #             np_avg_repr[batch_ind, sent_i] = cur_avg_repr
            #         else:
            #             np_avg_repr[batch_ind, sent_i] = avg_sents_repr[batch_ind, sent_i]

            #         ## store the nubmer of NP for a sentence
            #         num_np_sents[batch_ind, sent_i] = len(cur_j_np_list)

            #     # end if
            # # end for batch


            # # select forward-looking centers using attention scores
            # temp_fwr_centers, fwr_sort_ind = torch.sort(attn_diag, dim=1, descending=True)  # forward centers are selected by attn

            # temp_fwr_centers, fwr_sort_ind = torch.sort(t_attn_np, dim=1, descending=True)  # forward centers are selected by attn
            # fwr_sort_ind = fwr_sort_ind[:, :self.topk_fwr]

            # # we need to go reverse to identify original np phrases
            # # we just can return NPs captured as Cp directly, instead of their indexes
            # for cur_batch in range(batch_size):
            #     cur_j_np_list = list_j_np_map[cur_batch]
            #     cur_cp_ind = fwr_sort_ind[cur_batch, 0]

            #     if cur_j_np_list is not None and cur_cp_ind < len(cur_j_np_list):  # exception handling when sents do not have NPs
            #         cur_np, cur_np_pair = cur_j_np_list[cur_cp_ind]  # cur_np: string, cur_np_pair: pair for indexes
            #         batch_cp_nps[cur_batch][sent_i] = cur_np  # store the NP string

            # # end for

            # # batch_cp_ind[:, sent_i] = fwr_sort_ind[:, 0]  # only consider the top-1 item for Cp (depreciated)
            
            # fwr_centers = torch.zeros(batch_size, self.topk_fwr)
            # fwr_centers = utils.cast_type(fwr_centers, LONG, self.use_gpu)
            # fwr_centers[:, :fwr_sort_ind.size(1)] = fwr_sort_ind  # to handle execeptional case when the sent is shorter than topk

            # # prevent out of index problem, it happens when NP is lower than topk number
            # # fwr_centers[fwr_centers > encoded_sent.size(1)-1] = 0  # first item
            # fwr_centers[fwr_centers > encoded_sent.size(1)-1] = encoded_sent.size(1)-1  # last item

            # # Non-NP version
            # # selected = encoded_sent.gather(1, fwr_centers.unsqueeze(-1).expand(batch_size, self.topk_fwr, self.base_encoder.encoder_out_size))

            # # Np version
            # cur_sent_np_all = np_all_repr[:, sent_i, :, :]
            # selected = cur_sent_np_all.gather(2, fwr_centers.unsqueeze(-1).expand(batch_size, self.topk_fwr, self.base_encoder.encoder_out_size))

            # fwrd_repr[:, sent_i, :self.topk_fwr] = selected

            # ## then we need to handle fwr centers when the sentecne has NPs fewer than topk number
            # # simple idea is to assign avgeraged vector of the corresponding sentence
            # for cur_batch in range(batch_size):
            #     # cur_num_np_sent = num_np_sent[cur_batch]
            #     cur_num_np_sent = num_np_sents[cur_batch, sent_i]
                
            #     if cur_num_np_sent < 1:
            #         fwrd_repr[cur_batch, sent_i, 0, :] = avg_sents_repr[cur_batch, sent_i]
            #     elif cur_num_np_sent < self.topk_fwr:
            #         # fwrd_repr[cur_batch, sent_i, cur_num_np_sent:] = 0.0  # assign zero vector for memoryssing NP
            #         fwrd_repr[cur_batch, sent_i, cur_num_np_sent:] = fwrd_repr[cur_batch, sent_i, 0]  # assign the first item again

            #   # end for batch

            # # end if

        # end for sent_i

        return avg_sents_repr, fwrd_repr, np_avg_repr, batch_cp_nps, np_all_repr, num_np_sents
    # end def get_fwrd_centers

    def select_fwrd_centers_any(self, sent_i, encoded_sent, attn_diag, batch_cp_ind):

        batch_size = encoded_sent.size(0)

        temp_fwr_centers, fwr_sort_ind = torch.sort(attn_diag, dim=1, descending=True)  # forward centers are selected by attn
        fwr_sort_ind = fwr_sort_ind[:, :self.topk_fwr]
        batch_cp_ind[:, sent_i] = fwr_sort_ind[:, 0]  # only consider the top-1 item for Cp

        # # non-batched selecting by indices
        # temp = []
        # for batch_ind, cur_fwr in enumerate(fwr_centers):
        #     cur_fwrd_repr = encoded_sent[batch_ind].index_select(0, cur_fwr)
        #     temp.append(cur_fwrd_repr)
        # cur_fwrd_repr = torch.stack(temp)
        # fwrd_repr[:, sent_i] = cur_fwrd_repr

        # batched version selecting by indices
        fwr_centers = torch.zeros(batch_size, self.topk_fwr)
        fwr_centers = utils.cast_type(fwr_centers, LONG, self.use_gpu)
        fwr_centers[:, :fwr_sort_ind.size(1)] = fwr_sort_ind  # to handle execeptional case when the sent is shorter than topk
        selected = encoded_sent.gather(1, fwr_centers.unsqueeze(-1).expand(batch_size, self.topk_fwr, self.base_encoder.encoder_out_size))

        return selected, batch_cp_ind

    def select_fwrd_centers_np(self, sent_i, batch_map_np, encoded_sent, attn_diag, batch_cp_nps, avg_sents_repr, np_all_repr, num_np_sents):

        batch_size = encoded_sent.size(0)

        ## make NP representations and attn for them
        # extract np_map for current ith sentence for a given batch, due to the outer loop
        list_j_np_map = [row[sent_i] for row in batch_map_np]

        # need to be batched, need to record the nps to track to interpret
        t_attn_np = torch.empty(batch_size, self.max_num_np).fill_(-10.)
        t_attn_np = utils.cast_type(t_attn_np, FLOAT, self.use_gpu)

        ## make NP representations by averaging
        for batch_ind, cur_j_np_list in enumerate(list_j_np_map):
            # NPs at the j sentence
            np_ind = 0
            list_avg_np_repr = []
            if cur_j_np_list is not None:  # if there is a NP at this sentence (j th)
                for np_ind, cur_np_pair in enumerate(cur_j_np_list):  # iterate for each NP at this sent
                    cur_np = cur_np_pair[0]
                    cur_span = cur_np_pair[1]
                    s_ind, e_ind = cur_span

                    # extract repr corresponding to spans, and average them to make NP reprs
                    np_repr_span = encoded_sent[batch_ind, s_ind:e_ind+1, :]            

                    # average attn scores to make NP score
                    attn_score_np = attn_diag[batch_ind, s_ind:e_ind+1]
                    attn_score_np = torch.div(torch.sum(attn_score_np, dim=0), e_ind-s_ind+1)
                    t_attn_np[batch_ind, np_ind] = attn_score_np

                    list_avg_np_repr.append(np_repr_span)

                    # assign np all repr directly to use it out of this module
                    avg_np_span = torch.div(torch.sum(np_repr_span, dim=0), np_repr_span.size(1))
                    np_all_repr[batch_ind, sent_i, np_ind] = avg_np_span
                # end for

                ## store the nubmer of NP for a sentence
                num_np_sents[batch_ind, sent_i] = len(cur_j_np_list)

            # end if
        # end for batch

        ####
        # select forward-looking centers using attention scores
        temp_fwr_centers, fwr_sort_ind = torch.sort(t_attn_np, dim=1, descending=True)  # forward centers are selected by attn
        fwr_sort_ind = fwr_sort_ind[:, :self.topk_fwr]

        # we need to go reverse to identify original np phrases
        # we just can return NPs captured as Cp directly, instead of their indexes
        for cur_batch in range(batch_size):
            cur_j_np_list = list_j_np_map[cur_batch]
            cur_cp_ind = fwr_sort_ind[cur_batch, 0]

            if cur_j_np_list is not None and cur_cp_ind < len(cur_j_np_list):  # exception handling when sents do not have NPs
                cur_np, cur_np_pair = cur_j_np_list[cur_cp_ind]  # cur_np: string, cur_np_pair: pair for indexes
                batch_cp_nps[cur_batch][sent_i] = cur_np  # store the NP string

        # end for

        # batch_cp_ind[:, sent_i] = fwr_sort_ind[:, 0]  # only consider the top-1 item for Cp (depreciated)
        
        fwr_centers = torch.zeros(batch_size, self.topk_fwr)
        fwr_centers = utils.cast_type(fwr_centers, LONG, self.use_gpu)
        fwr_centers[:, :fwr_sort_ind.size(1)] = fwr_sort_ind  # to handle execeptional case when the sent is shorter than topk

        # prevent out of index problem, it happens when NP is lower than topk number
        # fwr_centers[fwr_centers > encoded_sent.size(1)-1] = 0  # first item
        fwr_centers[fwr_centers > encoded_sent.size(1)-1] = encoded_sent.size(1)-1  # last item

        # Non-NP version
        # selected = encoded_sent.gather(1, fwr_centers.unsqueeze(-1).expand(batch_size, self.topk_fwr, self.base_encoder.encoder_out_size))

        # Np version
        cur_sent_np_all = np_all_repr[:, sent_i, :, :]
        selected = cur_sent_np_all.gather(2, fwr_centers.unsqueeze(-1).expand(batch_size, self.topk_fwr, self.base_encoder.encoder_out_size))

        # exception handling
        for cur_batch in range(batch_size):
            # cur_num_np_sent = num_np_sent[cur_batch]
            cur_num_np_sent = num_np_sents[cur_batch, sent_i]
            
            if cur_num_np_sent < 1:
                selected[cur_batch, 0, :] = avg_sents_repr[cur_batch, sent_i]

                ## assign average repr to the first np item when it does not have NP
                np_all_repr[cur_batch, sent_i, 0, :] = avg_sents_repr[cur_batch, sent_i]

            elif cur_num_np_sent < self.topk_fwr:
                # fwrd_repr[cur_batch, sent_i, cur_num_np_sent:] = 0.0  # assign zero vector for memoryssing NP
                selected[cur_batch, cur_num_np_sent:] = selected[cur_batch, 0]  # assign the first item again

        # end for batch

        return selected, batch_cp_nps, np_all_repr, num_np_sents

    def get_back_centers(self, avg_sents_repr, fwrd_repr):
        """ Determemoryne backward-looking centers"""
        batch_size = avg_sents_repr.size(0)
        back_repr = torch.zeros(batch_size, self.max_num_sents, self.topk_back, self.base_encoder.encoder_out_size)
        back_repr = utils.cast_type(back_repr, FLOAT, self.use_gpu)

        for sent_i in range(self.max_num_sents):
            if sent_i == 0 or sent_i == self.max_num_sents-1:
                # there is no backward center in the first sentence
                continue
            # end if
            else:
                prev_fwrd_repr = fwrd_repr[:, sent_i-1, :, :]  # (batch_size, topk_fwrd, dim)
                cur_fwrd_repr = fwrd_repr[:, sent_i, :, :]  # (batch_size, topk_fwrd, dim)
                cur_sent_repr = avg_sents_repr[:, sent_i, :]  # (batch_size, dim)

                sim_rank = self.sim_cosine_d2(prev_fwrd_repr, cur_sent_repr.unsqueeze(1))
                
                max_sim_val, max_sim_ind = torch.max(sim_rank, dim=1)
                
                idx = max_sim_ind.view(-1, self.topk_back, 1).expand(max_sim_ind.size(0), self.topk_back, self.base_encoder.encoder_out_size)
                cur_back_repr = prev_fwrd_repr.gather(1, idx)
                back_repr[:, sent_i] = cur_back_repr

                # end for topk_i
            # end else

        # end for sent_i

        return back_repr
    # end def get_back_centers

    def get_disco_seg(self, cur_sent_num, ind_batch, fwrd_repr, cur_batch_repr):
        """ construct hierarchical discourse segments """

        # record the sequence of centering transitions (list of list)
        center_trans = []  # "conti", "retin", "shift"

        cur_seg_list = []  # current segment
        cur_seg_ind = 0
        stack_focus = []  # stack representing focusing

        seg_map = dict()
        adj_list = []  # adjacency list
        list_root_ds = []  # list up the first level segments

        for sent_i in range(cur_sent_num):
            cur_pref_repr = fwrd_repr[ind_batch, sent_i, 0, :]
            cur_pref_repr = cur_pref_repr.unsqueeze(0)
            cur_seg_list = cur_seg_list + [sent_i]

            status_trans = "0"  # 1: conti, 2: retain, 3: shift-conti, 4: shift-retain

            # for the first two sentences, skip them to make a initial stack
            if sent_i  < 2:
                # for the first and the second sent, just push
                # center_trans.append("conti")
                # center_trans.append(0)
                center_trans.append(1)
                continue
            # handle the last sentence
            elif sent_i == cur_sent_num-1:
                if len(stack_focus) < 1:
                    list_root_ds.append(cur_seg_ind)
                else:
                    top_seg_stack = stack_focus[-1]
                    adj_pair = (top_seg_stack, cur_seg_ind)
                    adj_list.append(adj_pair)
                
                seg_map[cur_seg_ind] = cur_seg_list
                stack_focus.append(cur_seg_ind)

                center_trans.append(1)
            # end if
            else:
                cur_back_repr = cur_batch_repr[sent_i, :, :]

                isCont = False
                # while len(stack_focus) > 0 and stack_focus[-1]!=0:       
                while len(stack_focus) > 0:                       
                    # consider average of sentences included in the top segment on the stack
                    top_seg_stack = stack_focus[-1]
                    cur_sent_stack = seg_map[top_seg_stack]
                    prev_repr = cur_batch_repr[cur_sent_stack, :, :]  
                    prev_back_repr = torch.div(torch.sum(prev_repr, dim=0), len(cur_sent_stack))

                    # calcualte the simemorylarity between backward
                    sim_back_vec = self.sim_cosine_d1(prev_back_repr, cur_back_repr)
                    sim_avg = torch.div(torch.sum(sim_back_vec, dim=0), sim_back_vec.size(0))
                    
                    # simemorylarity between the current backward and the prefered in the precedding sentence
                    sim_back_pref = self.sim_cosine_d1(cur_back_repr, cur_pref_repr)
                    
                    # if we find a place either continue or retain, then stop the loop
                    if sim_avg > self.threshold_sim:  # stack the item, and move to the next sentence
                        # if sim_back_fwrd > self.threshold_sim:  ## continuing
                        if sim_back_pref > self.threshold_sim:  ## continuing
                            isCont = True

                            # center_trans.append("conti")
                            if status_trans == "-1":
                                status_trans = 3
                            else:
                                status_trans = 1

                            # center_trans.append(status_trans)
                        else:  ## retaining
                            # push the current segment
                            isCont = False
                            # update stack and segment map
                            if len(stack_focus) < 1:
                                list_root_ds.append(cur_seg_ind)
                            else:
                                top_seg_stack = stack_focus[-1]
                                adj_pair = (top_seg_stack, cur_seg_ind)
                                adj_list.append(adj_pair)
                            
                            seg_map[cur_seg_ind] = cur_seg_list
                            stack_focus.append(cur_seg_ind)

                            cur_seg_ind += 1
                            cur_seg_list = []

                            # center_trans.append("retin")
                            if status_trans == "-1":
                                status_trans = 4
                            else:
                                status_trans = 2
                            # center_trans.append(status_trans)
                        # end else

                        center_trans.append(status_trans)
                        break                            
                    # shifting: pop the top item in the stack, and iterate to find the location
                    else:  
                        del stack_focus[-1]  # pop the top segment
                        isCont = False

                        # center_trans.append("shift")
                        status_trans = "-1"
                        # center_trans.append(2)
                # end while len(stack_focus)

                if ~isCont and len(stack_focus) < 1:
                    # when loop is over because pop everyting
                    stack_focus.append(cur_seg_ind)
                    seg_map[cur_seg_ind] = cur_seg_list
                    list_root_ds.append(cur_seg_ind)

                    cur_seg_ind += 1
                    cur_seg_list = []

                    status_trans = 3
                    center_trans.append(status_trans)
                    # end if
                # end if
            # end else
        # end for sent_i

        # return seg_map, adj_list, list_root_ds
        return seg_map, adj_list, list_root_ds, center_trans
    # end get_disco_seg

    #
    def make_tree_stru(self, seg_map, adj_list, list_root_ds):
        """ make a tree structure using the structural information """
        cur_tree = nx.DiGraph()  # tree structure for current document

        # consider root first
        for i in list_root_ds:
            cur_root_seg = seg_map[i]
            cur_tree.add_node(cur_root_seg[0])  # add the first item of segments in the root level

        # connect the first item of each segment in the root level
        for i in range(len(list_root_ds)):
            for j in range(i+1, len(list_root_ds)):
                cur_root_pair = (list_root_ds[i], list_root_ds[j])
                # adj_list.append(cur_root_pair)
                
                src_seg = seg_map[cur_root_pair[0]]
                dst_seg = seg_map[cur_root_pair[1]]        
                cur_tree.add_edge(src_seg[0], dst_seg[0])  # connect the first item of segments
        
        # connect sentences each other within intra segment
        for cur_seg, sents_seg in seg_map.items():
            if len(sents_seg) > 1:
                for i in range(len(sents_seg)-1):
                    cur_tree.add_edge(sents_seg[i], sents_seg[i+1])

        # then between segments
        for cur_pair in adj_list:
            src_seg = seg_map[cur_pair[0]]
            dst_seg = seg_map[cur_pair[1]]

            cur_tree.add_edge(src_seg[0], dst_seg[0])  # first sentence version

        # connect between siblings
        for cur_root in list_root_ds:
            childs = nx.descendants(cur_tree, cur_root)
            for cur_child in childs:
                siblings = list(cur_tree.successors(cur_child))
                if len(siblings) > 1:

                    for i in range(len(siblings)):
                        for j in range(i+1, len(siblings)):
                            cur_tree.add_edge(siblings[i], siblings[j])

        return cur_tree
    # end def make_tree_stru

    #
    def centering_attn(self, text_inputs, mask_input, len_sents, num_sents, tid):

        ## Parser stage1: determemoryne foward-looking centers and preferred centers
        # avg_sents_repr, fwrd_repr, batch_cp_ind = self.get_fwrd_centers(text_inputs, mask_input, len_sents)

        avg_sents_repr, fwrd_repr, np_avg_repr, batch_cp_nps, np_all_repr, num_np_sents = self.get_fwrd_centers_np_enc2(text_inputs, mask_input, len_sents, num_sents, tid)

        ## Parser stage2: decide backward center
        back_repr = self.get_back_centers(avg_sents_repr, fwrd_repr)

        ## Parser stage3: construct hierarchical discourse segments
        batch_segMap = []
        batch_adj_mat = []
        batch_adj_list = []
        batch_root_list = []

        batch_center_trans = []  # recording centering transitions
        trans_inputs = torch.zeros(text_inputs.size(0), self.max_num_sents)  # assume that an utterance is a sentence
        trans_inputs = utils.cast_type(trans_inputs, LONG, self.use_gpu)
        trans_mask = torch.zeros(text_inputs.size(0), self.max_num_sents)  # assume that an utterance is a sentence
        trans_mask = utils.cast_type(trans_mask, BOOL, self.use_gpu)

        ##
        for ind_batch, cur_batch_repr in enumerate(back_repr):
            cur_sent_num = int(num_sents[ind_batch])

            ## Parser stage3-1: get structural information
            # seg_map, adj_list, list_root_ds = self.get_disco_seg(cur_sent_num, ind_batch, fwrd_repr, cur_batch_repr)  # emnlp20 version
            seg_map, adj_list, list_root_ds, center_trans = self.get_disco_seg(cur_sent_num, ind_batch, fwrd_repr, cur_batch_repr)

            # update trans_inputs using center_trans
            t_trans = torch.LongTensor(center_trans)
            trans_inputs[ind_batch, 0:t_trans.size(0)] = t_trans
            trans_mask[ind_batch, 0:t_trans.size(0)] = True

            ## Parser stage3-2: make a tree structure using the information
            cur_tree = self.make_tree_stru(seg_map, adj_list, list_root_ds)

            ## Parser stage3-3: make a numpy array from networkx tree
            cur_adj_mat = np.zeros((self.max_num_sents, self.max_num_sents))
            undir_tree = cur_tree.to_undirected()  # we make an undirected tree
            np_adj_mat = nx.to_numpy_matrix(undir_tree)

            cur_adj_mat[:np_adj_mat.shape[0], :np_adj_mat.shape[1]]=np_adj_mat

            ## store structures for statistical analysis
            batch_adj_mat.append(cur_adj_mat)
            batch_adj_list.append(adj_list)
            batch_root_list.append(list_root_ds)
            batch_segMap.append(list(seg_map.items()) )

            batch_center_trans.append(center_trans)

        # end for ind_batch

        # structural information which will be passed to structure-aware transformer
        adj_mat = torch.from_numpy(np.array(batch_adj_mat))
        adj_mat = utils.cast_type(adj_mat, FLOAT, self.use_gpu)
        # batch_cp_ind = batch_cp_ind.tolist()

        # return  adj_mat, avg_sents_repr, batch_adj_list, batch_root_list, batch_segMap, batch_cp_ind
        # return  adj_mat, avg_sents_repr, batch_adj_list, batch_root_list, batch_segMap, batch_cp_ind, batch_center_trans
        return adj_mat, avg_sents_repr, batch_adj_list, batch_root_list, batch_segMap, batch_cp_nps, batch_center_trans, trans_inputs, trans_mask, fwrd_repr, np_avg_repr, np_all_repr, num_np_sents
    #

    #
    def get_np_pair_repr(self, np_all_repr, num_np_sents, len_sents, num_sents, sent_mask, tid):

        batch_size = np_all_repr.size(0)

        pair_np_repr = torch.zeros(batch_size, self.max_num_sents-1, 2, self.base_encoder.encoder_out_size)  # assume that an utterance is a sentence
        pair_np_repr = utils.cast_type(pair_np_repr, FLOAT, self.use_gpu)

        eps = 1e-8
        for sent_i in range(self.max_num_sents-1):
        # for sent_i in range(self.max_num_sents):

            #### step1: get the closest NP repr between adjacent sentences
            # s1 = np_all_repr[:, sent_i, :]  # (batch, num_np, dim)
            # s2 = np_all_repr[:, sent_i+1, :]

            # get pairwise cosine distance for all NP
            # prod = torch.bmm(s1, s2.transpose(2, 1))
            # norm = torch.norm(np_avg_repr, 2, 2)  # l2 norm
            # sim_nps = torch.div(prod, norm.view(norm.shape[0], norm.shape[1], 1))
            # sim_nps = torch.div(sim_nps, torch.unsqueeze(norm, 1))  # (batch_size, num_state, num_state)
            # sim_nps = torch.where(torch.isnan(sim_nps), torch.zeros_like(sim_nps), sim_nps)  # prevent nan by zero
            
            # s1_n = s1 / torch.norm(s1, 2, 2).unsqueeze(2)  # s1 is the current sentence
            # s2_n = s2 / torch.norm(s2, 2, 2).unsqueeze(2)  # s2 is the next sentence
            # sim_adj = torch.bmm(s1_n, s2_n.transpose(2, 1))

            # s1_n = torch.norm(s1, 2, 2).unsqueeze(2)  # s1 is the current sentence
            # s2_n = torch.norm(s2, 2, 2).unsqueeze(2)  # s2 is the next sentence
            # s1_norm = s1 / torch.max(s1_n, eps * torch.ones_like(s1_n))
            # s2_norm = s2 / torch.max(s2_n, eps * torch.ones_like(s2_n))
            # sim_adj = torch.bmm(s1_norm, s2_norm.transpose(2, 1))

            # # masking by the number of NP for sentences  (should be modified for batch, maybe torch.scatter?)
            # for cur_batch in range(batch_size):
            #     cur_batch_np_sents = num_np_sents[cur_batch]
            #     num_np_s1 = cur_batch_np_sents[sent_i]
            #     num_np_s2 = cur_batch_np_sents[sent_i+1]

            #     sim_adj[cur_batch, num_np_s1:, :] = -10  # masking for row
            #     sim_adj[cur_batch, :, num_np_s2:] = -10  # masking for column
            # # end for masking


            # select the maximum value to make a index list
            next_sent_i = sent_i+1
            list_ind_pair = []
            for cur_batch in range(batch_size):
                # cur_sim_adj = sim_adj[cur_batch]
                cur_sent_num = num_sents[cur_batch]

                # # we do not take care of actual sentences, because it will be masked in the tree-transformer if it is not an real sentence
                # if sent_i == cur_sent_num-1 or sent_i == self.max_num_sents-1:  # if this is the last sentence, then connect with the first sent
                    # next_sent_i = 0

                s1 = np_all_repr[cur_batch, sent_i, :]  # (batch, max_num_np, dim)
                s2 = np_all_repr[cur_batch, next_sent_i, :]

                ## cosine sim
                s1_n, s2_n = s1.norm(dim=1)[:, None], s2.norm(dim=1)[:, None]
                s1_norm = s1 / torch.max(s1_n, eps * torch.ones_like(s1_n))
                s2_norm = s2 / torch.max(s2_n, eps * torch.ones_like(s2_n))
                cur_sim_adj = torch.mm(s1_norm, s2_norm.transpose(0, 1))

                ## masking
                cur_batch_np_sents = num_np_sents[cur_batch]
                num_np_s1 = cur_batch_np_sents[sent_i]
                num_np_s2 = cur_batch_np_sents[next_sent_i]

                if num_np_s1 > 0:   cur_sim_adj[num_np_s1:, :] = -10  # masking for row
                else:   cur_sim_adj[1:, :] = -10  # masking for row
                if num_np_s2 > 0:   cur_sim_adj[:, num_np_s2:] = -10  # masking for column
                else:   cur_sim_adj[:, 1:] = -10  # masking for column

                ## get max index at 2d tensor
                # ind_t = (cur_sim_adj==torch.max(cur_sim_adj)).nonzero()  # shape: (1, 2)

                # alternative way using max func twice
                max_i_vals, max_i_indices = torch.max(cur_sim_adj, 0)
                max_j_index = torch.max(max_i_vals, 0)[1]
                ind_t = [max_i_indices[max_j_index], max_j_index]

                # ind_int_pair = (int(ind_t[0,0]), int(ind_t[0,1]))
                ind_int_pair = (int(ind_t[0]), int(ind_t[1]))
                list_ind_pair.append(ind_int_pair)

                # # extract two reprs from two sentences using the index list
                cur_ind_pair = list_ind_pair[cur_batch]
                ind1, ind2 = cur_ind_pair[0], cur_ind_pair[1]

                np_repr1 = s1[ind1]  # NP at ind1 on the sentence1
                np_repr2 = s2[ind2]

                pair_np_repr[cur_batch, sent_i, 0] = np_repr1
                pair_np_repr[cur_batch, sent_i, 1] = np_repr2
        
            # end for

        # end for sent_i

        #### setp2: average two NP reprs to make a pair repr

        np_pair_avg = torch.div(torch.sum(pair_np_repr, dim=2), 2)  # (batch, sent_num, dim)
        mask_np_pair_avg = sent_mask[:, 1:]
        # mask_np_pair_avg = sent_mask

        return np_pair_avg, mask_np_pair_avg

        # pair_np_repr = pair_np_repr.reshape(batch_size, -1, pair_np_repr.size(3))
        # mask_np_pair = sent_mask[:, 1:].repeat(1, 2)

        # return pair_np_repr, mask_np_pair


#### Forward function
#####################################################################################

    #
    def forward(self, text_inputs, mask_input, len_seq, len_sents, tid, len_para=None, list_rels=None, mode=""):
        # mask_input: (batch, max_tokens), len_sents: (batch, max_num_sents)
        batch_size = text_inputs.size(0)

        sent_mask = torch.sign(len_sents)  # (batch_size, len_sent)
        sent_mask = utils.cast_type(sent_mask, FLOAT, self.use_gpu)
        num_sents = sent_mask.sum(dim=1)  # (batch_size)

        #### Stage1 and 2: sentence repr and discourse segments parser
        # adj_mat, sent_repr, batch_adj_list, batch_root_list, batch_segMap, batch_cp_ind = self.centering_attn(text_inputs, mask_input, len_sents, num_sents, tid)
        adj_mat, sent_repr, batch_adj_list, batch_root_list, batch_segMap, batch_cp_nps, batch_center_trans, \
         trans_inputs, trans_mask, fwrd_repr, \
         np_avg_repr, np_all_repr, num_np_sents = self.centering_attn(text_inputs, mask_input, len_sents, num_sents, tid)

        # # #### doc-level encoding input text (disable the below 4 lines if GPU memory is not enough)
        # encoder_doc_out = self.base_encoder(text_inputs, mask_input, len_seq)
        # encoded_doc = encoder_doc_out[0]
        # if self.output_attentions:
        #     attn_doc_avg = encoder_doc_out[1]  # averaged mh attentions (batch, item, item)
        # sent_mask = torch.sign(len_sents)  # (batch_size, len_sent)
        # sent_mask = utils.cast_type(sent_mask, LONG, self.use_gpu)
        # num_sents = sent_mask.sum(dim=1)  # (batch_size)
        # sent_repr = self.sent_repr_avg(batch_size, encoded_doc, len_sents)

        #### Stage3-2: Test for centering transitions?
        ### baseline: naive transformer
        # fwrd_repr = fwrd_repr[:, :, 0, :]  # only consider cp
        del batch_center_trans
        del fwrd_repr
        del np_avg_repr
        del trans_inputs
        del trans_mask
        # fwrd_repr = torch.div(torch.sum(fwrd_repr, dim=2), self.topk_fwr)  # average all cf to fwrd

        #### Stage2: Avg
        ilc_vec = torch.div(torch.sum(sent_repr, dim=1), num_sents.unsqueeze(1))

        # # unified attention test
        # ilc_vec = ilc_vec.unsqueeze(1)
        # context_weight = self.context_weight_unified.expand(ilc_vec.shape[0], ilc_vec.shape[2], 1)
        # attn_weight = torch.bmm(ilc_vec, context_weight).squeeze(2)
        # attn_weight = self.tanh(attn_weight)
        # attn_weight = self.softmax_d0(attn_weight)
        # attn_vec = torch.bmm(ilc_vec.transpose(1, 2), attn_weight.unsqueeze(2))
        # attn_unified = attn_vec.squeeze(2)

        #### FC layer
        # fc_out = self.linear_1(attn_unified)

        fc_out = self.linear_1(ilc_vec)
        fc_out = self.leak_relu(fc_out)
        # fc_out = gelu(fc_out)
        fc_out = self.dropout_layer(fc_out)

        fc_out = self.linear_2(fc_out)
        fc_out = self.leak_relu(fc_out)
        # fc_out = gelu(fc_out)
        fc_out = self.dropout_layer(fc_out)

        fc_out = self.linear_out(fc_out)

        if self.output_size == 1:
            fc_out = self.sigmoid(fc_out)

        # prepare output to return
        outputs = []
        outputs.append(fc_out)

        if self.gen_logs:
            outputs.append(batch_adj_list)
            outputs.append(batch_root_list)
            outputs.append(batch_segMap)
            outputs.append(batch_cp_nps)
            outputs.append(num_sents.tolist())

            outputs.append(batch_center_trans)

        # return fc_out
        return outputs
    # end def forward

# end class