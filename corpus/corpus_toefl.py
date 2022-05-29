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
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os, io, time

from collections import Counter
import pandas as pd
import numpy as np
import sklearn.model_selection
import nltk
from unidecode import unidecode
import statistics
import re
import json

import corpus.corpus_base
from corpus.corpus_base import CorpusBase
from corpus.corpus_base import PAD, UNK, BOS, EOS, BOD, EOD, SEP, TIME, DATE

logger = logging.getLogger()


class CorpusTOEFL(CorpusBase):
    """ Corpus class for TOEFL dataset """
    """ prompt: 1 to 8 (p1 to p8 in the paper) """

    #
    def __init__(self, config):
        super(CorpusTOEFL, self).__init__(config)

        self.score_ranges = {
            1: (0, 2), 2: (0, 2), 3: (0, 2), 4: (0, 2), 5: (0, 2), 6: (0, 2), 7: (0, 2), 8: (0, 2)  # low, medium, high
        }

        if config.output_size < 0:
            config.output_size = 3

        # if config.num_fold < 0:
        #     config.num_fold = 5

        self.ratio_high_score = 0.66
        self.ratio_mid_score = 0.33

        self.train_total_pd = None
        self.valid_total_pd = None
        self.test_total_pd = None

        self.train_pd = None
        self.valid_pd = None
        self.test_pd = None

        self.is_scale_label = False
        if config.loss_type.lower() == "mseloss":
            self.is_scale_label = True

        self.output_bias = None

        # self.cur_prompt_id = config.essay_prompt_id
        self.prompt_id_train = config.essay_prompt_id_train
        self.prompt_id_test = config.essay_prompt_id_test

        self.k_fold = config.num_fold

        self.use_lower = config.use_lower
        self.use_coref = config.use_coref

        # self.train_corpus = None  # just for remind, already declared in the corpus_base
        # self.test_corpus = None
        if config.is_gen_cv:
            seed = np.random.seed(seed=int(time.time()))
            self.generate_kfold(config=config, seed=seed)  # generate k-fold file

        self._read_dataset(config)  # store whole dataset as pd.dataframe

        self._build_vocab(config.max_vocab_cnt)

        return

    # end __init__

    #
    def _get_model_friendly_scores(self, essay_pd, prompt_id_target):
        """ scale between 0 to 1 for MSE loss function"""

        scores_array = essay_pd['essay_score'].values
        min_rating, max_rating = self.score_ranges[prompt_id_target]
        scaled_label = (scores_array - min_rating) / (max_rating - min_rating)

        essay_pd.insert(len(essay_pd.columns), 'rescaled_label', scaled_label)

        return essay_pd

    #
    def _read_dataset(self, config):
        """ read asap dataset, assumed that splitted to "train.csv", "dev.csv", "test.csv" under "fold_" """

        path_data = config.data_dir
        # cur_path_asap = os.path.join(path_asap, "fold_" + str(config.cur_fold))
        cur_path_cv = os.path.join(path_data, config.data_dir_cv)

        # read
        str_cur_fold = str(config.cur_fold)

        if config.use_coref:
            self.train_total_pd = pd.read_csv(os.path.join(cur_path_cv, "train_coref_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c')  # skip the first line
            self.valid_total_pd = pd.read_csv(os.path.join(cur_path_cv, "valid_coref_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c')
            self.test_total_pd = pd.read_csv(os.path.join(cur_path_cv, "test_coref_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c')
        else:        
            self.train_total_pd = pd.read_csv(os.path.join(cur_path_cv, "train_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c')  # skip the first line
            self.valid_total_pd = pd.read_csv(os.path.join(cur_path_cv, "valid_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c')
            self.test_total_pd = pd.read_csv(os.path.join(cur_path_cv, "test_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c')

        # self.train_total_pd = pd.read_csv(os.path.join(cur_path_cv, "train.csv"), sep=",", header=0,
        #                                   encoding="utf-8", engine='c')  # skip the first line
        # self.valid_total_pd = pd.read_csv(os.path.join(cur_path_cv, "valid.csv"), sep=",", header=0, encoding="utf-8",
        #                                   engine='c')
        # self.test_total_pd = pd.read_csv(os.path.join(cur_path_cv, "test.csv"), sep=",", header=0, encoding="utf-8",
        #                                  engine='c')


        # ## without cross-validation test
        # self.train_total_pd = pd.read_csv(os.path.join(cur_path_cv, "train.csv"), sep=",", header=0, encoding="utf-8", engine='c')  # skip the first line
        # self.valid_total_pd = pd.read_csv(os.path.join(cur_path_cv, "valid.csv"), sep=",", header=0, encoding="utf-8", engine='c')
        # self.test_total_pd = pd.read_csv(os.path.join(cur_path_cv, "test.csv"), sep=",", header=0, encoding="utf-8", engine='c')

        self.train_pd = self.train_total_pd.loc[self.train_total_pd['prompt'] == self.prompt_id_train]
        self.valid_pd = self.valid_total_pd.loc[self.valid_total_pd['prompt'] == self.prompt_id_test]
        self.test_pd = self.test_total_pd.loc[self.test_total_pd['prompt'] == self.prompt_id_test]

        # ## sampling for the development
        # self.train_pd = self.train_pd[:35]
        # self.valid_pd = self.valid_pd[:35]
        # self.test_pd = self.test_pd[:35]

        # # test whole prompts
        # self.train_pd = self.train_total_pd
        # self.valid_pd = self.valid_total_pd
        # self.test_pd = self.test_total_pd

        self.merged_pd = pd.concat([self.train_pd, self.valid_pd, self.test_pd], sort=True)
        # print("Total # of essays:" + str(len(self.merged_pd)))

        # scaling score for training due to different score range as prompt_id
        if self.is_scale_label:
            self.train_pd = self._get_model_friendly_scores(self.train_pd, self.prompt_id_train)
            self.valid_pd = self._get_model_friendly_scores(self.valid_pd, self.prompt_id_test)
            self.test_pd = self._get_model_friendly_scores(self.test_pd, self.prompt_id_test)

        # put bias as mean of labels
        if self.is_scale_label:
            self.output_bias = self.train_pd['rescaled_label'].values.mean(axis=0)

        # convert to id
        # self.train_corpus = self._sent_split_corpus(self.train_pd['essay'].values)  # (doc_id, list of sents)
        # self.valid_corpus = self._sent_split_corpus(self.valid_pd['essay'].values)
        # self.test_corpus = self._sent_split_corpus(self.test_pd['essay'].values)

        tid_train = self.train_pd['essay_id'].values
        tid_valid = self.valid_pd['essay_id'].values
        tid_test = self.test_pd['essay_id'].values

        if config.use_coref:
            text_train = self.train_pd['essay_coref'].values
            text_valid = self.valid_pd['essay_coref'].values
            text_test = self.test_pd['essay_coref'].values
        else:
            text_train = self.train_pd['essay'].values
            text_valid = self.valid_pd['essay'].values
            text_test = self.test_pd['essay'].values

        self.train_corpus, self.num_sents_train = self._sent_split_corpus(text_train, tid_train)  # (doc_id, list of sents)
        self.valid_corpus, self.num_sents_valid = self._sent_split_corpus(text_valid, tid_valid)
        self.test_corpus, self.num_sents_test = self._sent_split_corpus(text_test, tid_test)

        # self.train_corpus, self.num_sents_train, map_np_sbw_train, list_num_np_train = self._sent_split_corpus(self.train_pd['essay'].values, tid_train)  # (doc_id, list of sents)
        # self.valid_corpus, self.num_sents_valid, map_np_sbw_valid, list_num_np_valid = self._sent_split_corpus(self.valid_pd['essay'].values, tid_valid)
        # self.test_corpus, self.num_sents_test, map_np_sbw_test, list_num_np_test = self._sent_split_corpus(self.test_pd['essay'].values, tid_test)

        # get statistics for training later
        self._get_stat_corpus()

       
        return

    # end _read_dataset

    # stanza tokenizer test
    def _sent_split_corpus(self, arr_input_text, tid):
        """ tokenize corpus given tokenizer by config file"""
        # arr_input_text = pd_input['essay'].values

        # num_over = 0
        # total_sent = 0

        import stanza  # stanford library for tokenizer
        tokenizer_stanza = stanza.Pipeline('en', processors='tokenize', use_gpu=True)

        #
        num_sents = []
        sent_corpus = []  # tokenized to form of [doc, list of sentences]

        #
        # sent_np_sbw = []  # np information in subword tokenization; each document consists of list of maps
        # sent_np_origin = []  # np information in origin
        # sent_mapping_loc = []  # mapping from origin locations to subword tokenization

        map_np_sbw = dict()  # np information in subword tokenization; each document consists of list of maps
        map_np_origin = dict()  # np information in origin # for error analysis later (unnecessary for testing)
        map_mapping_loc = dict()  # mapping from origin locations to subword tokenization # for error analysis later (unnecessary for testing)

        #
        # for cur_doc in arr_input_text:
        list_num_np = []  # each max # of np of a document
        for cur_ind, cur_doc in enumerate(arr_input_text):
            cur_doc = self._refine_text(cur_doc)  # cur_doc: single string
            
            # # if correference resolution is required (depreciated, cuz pp in advance now)
            # if self.use_coref == True:
            #     # 
            #     import spacy, neuralcoref
            #     nlp = spacy.load('en')
            #     coref = neuralcoref.NeuralCoref(nlp.vocab)
            #     nlp.add_pipe(coref, name='neuralcoref')

            #     # correference resolution (need to pull mentions to investigate intermediate process)
            #     cur_doc = nlp(cur_doc)._.coref_resolved
            # # end if coref


            # sent_list = [sent.string.strip() for sent in spacy_nlp(cur_doc).sents] # spacy style

            ## stanza version
            doc_stanza = tokenizer_stanza(cur_doc)
            sent_list = [sentence.text for sentence in doc_stanza.sentences]
           
            ## normal version
            # sent_list = self.sent_tokenzier(cur_doc)  # following exactly same way with previous works
            
            sent_corpus.append(sent_list)
            num_sents.append(len(sent_list))

            # ## extracting np information in the subword tokenization using np parser
            # cur_tid = int(tid[cur_ind])
            # list_sent_np_sbw, list_sent_loc_sbw, list_sent_map_np, max_num_np = self.np_parser.identify_loc_NP_sbw(sent_list)
            # # sent_np_sbw.append(list_sent_np_sbw)
            # # sent_np_origin.append(list_sent_loc_sbw)
            # # sent_mapping_loc.append(list_sent_map_np)

            # map_np_sbw[cur_tid] = list_sent_np_sbw
            # map_np_origin[cur_tid] = list_sent_loc_sbw
            # map_mapping_loc[cur_tid] = list_sent_map_np
            # list_num_np.append(max_num_np)

        # end for

        
        return sent_corpus, num_sents
        # return sent_corpus, num_sents, map_np_sbw, list_num_np

    #
    def _refine_text(self, input_text, ignore_uni=True, ignore_para=True):
        """ customized function for pre-processing raw text"""
        if self.use_lower:
            input_text = input_text.lower()

        ## because it causes a different length in sentpiece tokenization, and it makes us difficult to track NP
        input_text = re.sub(r'\s+([?.!"])', r'\1', input_text)

        out_text = input_text

        return out_text

    #
    def _make_hist(self, keys, list_DS_rels):
        hist_map = dict()
        for cur_DS in list_DS_rels:
            cur_list = cur_DS.list_sent_ds
            for cur_sent in cur_list:
                hist_val = 1

                cur_type = cur_sent[1]
                if cur_type in hist_map:
                    hist_val = hist_map[cur_type] + 1

                # in the hist, value which indicates ds relation becomes the key
                hist_map[cur_type] = hist_val
            # end for cur_map

        # end for target_map


        return hist_map

    #
    def get_np_info(self, config):
        #
        if config.use_nounp:
            np_file_path = os.path.join(config.data_dir, "nounp")  # stanza 1.1.1 ver

            cur_fold = str(config.cur_fold)

            # len_np = 2
            len_np = 3  # need to be passed from outer 
            # len_np = 4  # need to be passed from outer 
            len_np = str(len_np)

            if config.use_coref:
                file_map_np_sbw = config.corpus_target.lower() + "_nounp_" + len_np + "_lower_coref" + "_fold_" + cur_fold + ".csv"
            else:
                file_map_np_sbw = config.corpus_target.lower() + "_nounp_" + len_np + "_lower" + "_fold_" + cur_fold + ".csv"
            
            #
            with open(os.path.join(np_file_path, file_map_np_sbw), 'r') as fin:
                # key: tid, value: list of list of map; describes noun phrases in subword tokens
                map_np_sbw_total = json.load(fin)  
            
            #
            max_num_np = map_np_sbw_total["max_num_np"][0]
        else:
            map_np_sbw_total=None
            max_num_np=None

        return map_np_sbw_total, max_num_np
    # end def

    #
    def get_id_corpus(self, num_fold=-1):
        """
        return id-converted corpus which is read in the earlier stage
        :param num_fold:
        :return: map of id-converted sentence
        """

        train_corpus = None
        valid_corpus = None
        test_corpus = None
        y_train = None
        y_valid = None
        y_test = None

        # ASAP is already divided
        train_corpus = self.train_corpus
        valid_corpus = self.valid_corpus
        test_corpus = self.test_corpus

        # change each sentence to id-parsed
        x_id_train, max_len_doc_train, list_len_train = self._to_id_corpus(train_corpus)
        x_id_valid, max_len_doc_valid, list_len_valid = self._to_id_corpus(valid_corpus)
        x_id_test, max_len_doc_test, list_len_test = self._to_id_corpus(test_corpus)

        max_len_doc = max(max_len_doc_train, max_len_doc_valid, max_len_doc_test)
        # avg_len_doc = avg_len_doc_train

        list_len = list_len_train + list_len_valid + list_len_test

        max_len_doc = max(list_len)
        avg_len_doc = statistics.mean(list_len)
        std_len_doc = statistics.stdev(list_len)

        # print(max_len_doc)
        # print(avg_len_doc)
        # print(std_len_doc)

        # print(welkfjweklf)


        max_num_sents = max(max(self.num_sents_train), max(self.num_sents_valid), max(self.num_sents_test))  

        # paragraph info
        len_para_train = None
        len_para_valid = None
        len_para_test = None
        
        max_num_para = -1
        # max_num_sents = -1
        len_para_train = []
        len_para_valid = []
        len_para_test = []
        # if self.use_paragraph:
        #     list_num_para_train, len_para_train, list_num_sents_train = self._get_para_info(self.train_pd['essay'])
        #     list_num_para_valid, len_para_valid, list_num_sents_valid = self._get_para_info(self.valid_pd['essay'])
        #     list_num_para_test, len_para_test, list_num_sents_test = self._get_para_info(self.test_pd['essay'])
        #     max_num_para = max(max(list_num_para_train), max(list_num_para_valid), max(list_num_para_test))
        #     max_num_sents = max(max(list_num_sents_train), max(list_num_sents_valid), max(list_num_sents_test))            

        if 'rescaled_label' in self.train_pd:
            y_train = self.train_pd['rescaled_label'].values
            y_valid = self.valid_pd['rescaled_label'].values
            y_test = self.test_pd['rescaled_label'].values
        else:
            y_train = self.train_pd['essay_score'].values
            y_valid = self.valid_pd['essay_score'].values
            y_test = self.test_pd['essay_score'].values

        score_train = self.train_pd['essay_score'].values
        score_valid = self.valid_pd['essay_score'].values
        score_test = self.test_pd['essay_score'].values

        tid_train = self.train_pd['essay_id'].values
        tid_valid = self.valid_pd['essay_id'].values
        tid_test = self.test_pd['essay_id'].values

        train_data_id = {'x_data': x_id_train, 'y_label': y_train, 'tid': tid_train, 'len_para': len_para_train, 'origin_score': score_train}  # origin score is needed for qwk
        valid_data_id = {'x_data': x_id_valid, 'y_label': y_valid, 'tid': tid_valid, 'len_para': len_para_valid, 'origin_score': score_valid}
        test_data_id = {'x_data': x_id_test, 'y_label': y_test, 'tid': tid_test, 'len_para': len_para_test, 'origin_score': score_test}
        
        id_corpus = {'train': train_data_id, 'valid': valid_data_id, 'test': test_data_id}

        return id_corpus, max_len_doc, avg_len_doc, max_num_para, max_num_sents

    # end def get_id_corpus

    #
    def generate_kfold(self, config, seed):
        """ generate new k-fold"""
        path_data_dir = config.data_dir
        path_cv_dir = config.data_dir_cv

        if not os.path.exists(os.path.join(path_data_dir, path_cv_dir)):
            os.makedirs(os.path.join(path_data_dir, path_cv_dir))

        file_path = os.path.join(path_data_dir, "pp_toefl_essay.csv")
        pd_toefl_essay = pd.read_csv(file_path)  # essay_id, prompt, native_lang, essay_score, essay

        # # splitting by manually
        # make chunks to define train, valid, test

        list_prompts_folded = []
        for prompt_id in range(1,9):
            cur_prompt_pd = pd_toefl_essay.loc[pd_toefl_essay['prompt'] == prompt_id]
            cur_prompt_np = cur_prompt_pd.values

            np.random.seed(int(time.time()))
            rand_index = np.array(range(len(cur_prompt_np)))
            np.random.shuffle(rand_index)
            shuffled_input = cur_prompt_np[rand_index]
            list_chunk_input = np.array_split(shuffled_input, config.num_fold)

            cur_map_prompt_fold = dict()
            for cur_fold in range(config.num_fold):
                # prepare chunks for train according to each k-fold
                train_chunks = []
                valid_chunks = []
                test_chunks = []
                for cur_ind in range(config.num_fold):
                    cur_test_ind = (config.num_fold - 1) - cur_fold
                    cur_valid_ind = cur_test_ind - 1
                    if cur_valid_ind < 0:
                        cur_valid_ind = cur_valid_ind + config.num_fold

                    if cur_ind == cur_test_ind:
                        test_chunks.append(list_chunk_input[cur_ind])
                    elif cur_ind == cur_valid_ind:
                        valid_chunks.append(list_chunk_input[cur_ind])
                    else:
                        train_chunks.append(list_chunk_input[cur_ind])

                #
                cur_train_np = np.concatenate(train_chunks, axis=0)
                cur_valid_np = np.concatenate(valid_chunks, axis=0)
                cur_test_np = np.concatenate(test_chunks, axis=0)
                cur_map_prompt_fold[cur_fold] = {"train": cur_train_np, "valid": cur_valid_np, "test": cur_test_np}
            # end for cur_fold

            list_prompts_folded.append(cur_map_prompt_fold)
        # end for prompt_id

        col_toefl = list(pd_toefl_essay)
        for cur_fold in range(config.num_fold):
            cur_fold_train = []
            cur_fold_valid = []
            cur_fold_test = []

            # merge prompts again for cur-fold level
            for prompt_id in range(0,8):
                cur_prompt_train = list_prompts_folded[prompt_id][cur_fold]["train"]
                cur_prompt_valid = list_prompts_folded[prompt_id][cur_fold]["valid"]
                cur_prompt_test = list_prompts_folded[prompt_id][cur_fold]["test"]

                cur_fold_train.append(cur_prompt_train)
                cur_fold_valid.append(cur_prompt_valid)
                cur_fold_test.append(cur_prompt_test)
            # end for prompt_id

            fold_train_np = np.concatenate(cur_fold_train, axis=0)
            fold_valid_np = np.concatenate(cur_fold_valid, axis=0)
            fold_test_np = np.concatenate(cur_fold_test, axis=0)

            cur_train_file = "train" + "_fold_" + str(cur_fold) + ".csv"
            cur_valid_file = "valid" + "_fold_" + str(cur_fold) + ".csv"
            cur_test_file = "test" + "_fold_" + str(cur_fold) + ".csv"

            pd.DataFrame(fold_train_np, columns=col_toefl).to_csv(os.path.join(path_data_dir, path_cv_dir, cur_train_file), index=None)
            pd.DataFrame(fold_valid_np, columns=col_toefl).to_csv(os.path.join(path_data_dir, path_cv_dir, cur_valid_file), index=None)
            pd.DataFrame(fold_test_np, columns=col_toefl).to_csv(os.path.join(path_data_dir, path_cv_dir, cur_test_file), index=None)
        # end for cur_fold

        return

    # end generate_kfold

    #
    def read_kfold(self, config):

        """ not used for asap, to follow the same setting with previous work"""

        return

# end class
