# -*- coding: utf-8 -*-
# It a corpus reader for "Discourse Coherence in the Wild: A Dataset, Evaluation and Methods", SIGDIAL18

import logging
import os
import re

from collections import Counter

import pandas as pd
import numpy as np
import sklearn.model_selection
import nltk
from unidecode import unidecode
import statistics
from statistics import mean 
import json

import corpus.corpus_base
from corpus.corpus_base import CorpusBase
from corpus.corpus_base import PAD, UNK, BOS, EOS, BOD, EOD, SEP, TIME, DATE

logger = logging.getLogger(__name__)


class CorpusGCDC(CorpusBase):
    """ Corpus class for GCDC dataset"""

    #
    def __init__(self, config):
        super(CorpusGCDC, self).__init__(config)

        self.gcdc_ranges = (1, 3)  # used if it needs to re-scale
        self.gcdc_domain = config.gcdc_domain
        
        if config.output_size < 0:
            config.output_size = 3  # 3-class classification

        if config.num_fold < 0:
            config.num_fold = 10

        self.ratio_high_score = 0.66
        self.ratio_mid_score = 0.33

        if config.is_gen_cv:
            seed = np.random.seed()
            self.generate_kfold(config=config, seed=seed)  # generate k-fold file

        # self.pd_gcdc = None  # store whole dataset info (depreciated, because there is no need to use whole)
        self._read_dataset(config)  # store whole dataset as pd.dataframe

        # build vocab
        # if self.tokenizer_type.startswith('nltk'):
        #     self._build_vocab(config.max_vocab_cnt)
        # elif self.tokenizer_type.startswith('bert-'):
        #     self.vocab = self.tokenizer.vocab
        #     self.rev_vocab = self.tokenizer.ids_to_tokens

        #
        self.read_kfold(config)

        self._build_vocab(config.max_vocab_cnt)

    def _parsing_yahoo_text(self, target_pd):
        input_texts = []
        q_title = target_pd['question_title'].values
        ques = target_pd['question'].values
        texts = target_pd['text'].values
        for cur_ind in range(len(texts)):
            cur_titl = q_title[cur_ind]
            cur_ques = ques[cur_ind]
            cur_text = texts[cur_ind]

            # cur_str = cur_titl + " " + cur_ques + " " + cur_text
            cur_str = cur_ques + " " + cur_text

            # cur_str = cur_titl + "\n\n" + cur_ques + "\n\n" + cur_text
            # cur_str = cur_ques + "\n\n" + cur_text
            # cur_str = cur_ques
            # cur_str = cur_text
            input_texts.append(cur_str)

        return input_texts

    #
    def _read_dataset(self, config):
        """ read dataset """

        ## first, read dataframe from the file
        path_gcdc = config.data_dir
        cur_path_gcdc = os.path.join(path_gcdc, config.data_dir_cv)

        # file_name = config.gcdc_domain + "_train.csv"
        # train_pd = pd.read_csv(os.path.join(path_gcdc, file_name))
        # file_name = config.gcdc_domain + "_test.csv"
        # test_pd = pd.read_csv(os.path.join(path_gcdc, file_name))

        # str_cur_fold = str(config.cur_fold)
        # file_name = config.gcdc_domain + "_train_fold_" + str_cur_fold + ".csv"
        # train_pd = pd.read_csv(os.path.join(cur_path_gcdc, file_name))
        # file_name = config.gcdc_domain + "_test_fold_" + str_cur_fold + ".csv"
        # test_pd = pd.read_csv(os.path.join(cur_path_gcdc, file_name))

        str_cur_fold = str(config.cur_fold)
        file_name = config.gcdc_domain + "_train_fold_" + str_cur_fold + ".csv"
        train_pd = pd.read_csv(os.path.join(cur_path_gcdc, file_name))
        file_name = config.gcdc_domain + "_test_fold_" + str_cur_fold + ".csv"
        test_pd = pd.read_csv(os.path.join(cur_path_gcdc, file_name))

        # # put original index for conveinence
        # ind_origin = list(range(0, len(train_pd)))
        # train_pd.insert(0, 'ind_origin', ind_origin)
        # ind_origin = list(range(0, len(test_pd)))
        # test_pd.insert(0, 'ind_origin', ind_origin)

        self.train_pd = train_pd
        self.valid_pd = None
        self.test_pd = test_pd

        self.merged_pd = pd.concat([self.train_pd, self.test_pd], sort=True)

        # rescaling (for mse loss)
        # self.train_pd = self._get_rescaled_scores(train_pd)
        # self.test_pd = self._get_rescaled_scores(test_pd)
        # self.output_bias = self.train_pd['rescaled_label'].values.mean(axis=0)

        ## extract only corpus with sentence level from train and test
        text_train = ""
        if self.gcdc_domain.lower() == "yahoo_q":
            text_train = self._parsing_yahoo_text(self.train_pd)
        else:
            text_train = self.train_pd['text'].values

        # self.train_corpus = self._sent_split_corpus(text_train)  # sentence level tokenized
        self.train_corpus, self.num_sents_train = self._sent_split_corpus(text_train)  # sentence level tokenized
        self.valid_corpus = None

        text_test = ""
        if self.gcdc_domain.lower() == "yahoo_q":
            text_test = self._parsing_yahoo_text(self.test_pd)
        else:
            text_test = self.test_pd['text'].values

        # self.test_corpus = self._sent_split_corpus(text_test)
        self.test_corpus, self.num_sents_test = self._sent_split_corpus(text_test)

        ## get stat corpus
        self._get_stat_corpus()

        print(self.avg_len_doc)

        return

    # end _read_dataset

    #
    def generate_kfold(self, config, seed):
        """ Generate k-fold CV"""
        num_fold = config.num_fold

        # prepare target directory and file
        path_gcdc = config.data_dir
        path_fold_dir = config.data_dir_cv
        if not os.path.exists(os.path.join(path_gcdc, path_fold_dir)):
            os.makedirs(os.path.join(path_gcdc, path_fold_dir))

        file_name = self.gcdc_domain + "_train.csv"
        pd_train = pd.read_csv(os.path.join(path_gcdc, file_name))
        file_name = self.gcdc_domain + "_test.csv"
        pd_test = pd.read_csv(os.path.join(path_gcdc, file_name))
        pd_valid = None

        # add original index
        ind_origin = list(range(0, len(pd_train)))
        pd_train.insert(0, 'ind_origin', ind_origin)
        ind_origin = list(range(0, len(pd_test)))
        pd_test.insert(0, 'ind_origin', ind_origin)

        # convert to numpy array
        arr_train = pd_train.values
        arr_test = pd_test.values
        arr_input_combined = np.vstack([arr_train, arr_test])
        col_gcdc = list(pd_train)

        ## splitting by KFold (seperated version)
        # generate chunk
        seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        rand_index = np.array(range(len(cur_train_np)))
        np.random.shuffle(rand_index)
        shuffled_train = cur_train_np[rand_index]
        list_chunk_train = np.array_split(shuffled_train, num_fold)

        seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        rand_index = np.array(range(len(cur_test_np)))
        shuffled_test = cur_test_np[rand_index]
        list_chunk_test = np.array_split(shuffled_test, num_fold)

        # concat chunks to num_fold
        # cur_map_fold = dict()
        for cur_fold in range(num_fold):
            train_chunks = []
            test_chunks = []

            for cur_ind in range(num_fold):
                cur_skip_ind = (num_fold - 1) - cur_fold
                if cur_ind == cur_skip_ind:
                    continue
                else:
                    train_chunks.append(list_chunk_train[cur_ind])
                    test_chunks.append(list_chunk_test[cur_ind])
            # end for cur_ind
            print(len(train_chunks))

            cur_train_np = np.concatenate(train_chunks, axis=0)
            cur_test_np = np.concatenate(test_chunks, axis=0)
            # cur_map_fold[cur_fold] = {"train": cur_train_np, "test": cur_test_np}

            # save CV partition
            cur_train_file = self.gcdc_domain + "_train" + "_fold_" + str(cur_fold) + ".csv"
            cur_test_file = self.gcdc_domain + "_test" + "_fold_" + str(cur_fold) + ".csv"
            # print(pd.DataFrame(input_train, columns=col_gcdc).head())
            pd.DataFrame(cur_train_np, columns=col_gcdc).to_csv(os.path.join(path_gcdc, path_out_dir, cur_train_file), index=None)
            pd.DataFrame(cur_test_np, columns=col_gcdc).to_csv(os.path.join(path_gcdc, path_out_dir, cur_test_file), index=None)


        # # splitting by KFold (combined version)
        # kf = sklearn.model_selection.KFold(n_splits=config.num_fold, random_state=seed, shuffle=True)
        # ind = 0
        # for ind_train, ind_test in kf.split(arr_input_combined):
        #     input_train, input_test = arr_input_combined[ind_train], arr_input_combined[ind_test]

        #     # save for reproduction later
        #     cur_train_file = self.gcdc_domain + "_train" + "_fold_" + str(ind) + ".csv"
        #     cur_test_file = self.gcdc_domain + "_test" + "_fold_" + str(ind) + ".csv"
        #     # print(pd.DataFrame(input_train, columns=col_gcdc).head())
        #     pd.DataFrame(input_train, columns=col_gcdc).to_csv(
        #         os.path.join(path_gcdc, path_fold_dir, cur_train_file),
        #         index=None)
        #     pd.DataFrame(input_test, columns=col_gcdc).to_csv(os.path.join(path_gcdc, path_fold_dir, cur_test_file),
        #                                                       index=None)
        #     ind = ind + 1

        return

    #
    def read_kfold(self, config):
        fold_train = []  # list of structured input, (num_fold, num_docs)
        fold_test = []  # list of structured input, (num_fold, num_docs)

        for ind_fold in range(0, config.num_fold):
            cur_train_file = self.gcdc_domain + "_train" + "_fold_" + str(ind_fold) + ".csv"
            cur_test_file = self.gcdc_domain + "_test" + "_fold_" + str(ind_fold) + ".csv"

            # read each kfold from files
            train_pd = pd.read_csv(os.path.join(config.data_dir, config.data_dir_cv, cur_train_file))
            test_pd = pd.read_csv(os.path.join(config.data_dir, config.data_dir_cv, cur_test_file))

            fold_train.append(train_pd)
            fold_test.append(test_pd)

        self.fold_train = fold_train
        self.fold_test = fold_test

        return fold_train, fold_test

    #
    def _get_rescaled_scores(self, gcdc_pd):
        scores_array = gcdc_pd['labelA'].values
        min_rating, max_rating = self.gcdc_ranges
        rescaled_label = (scores_array - min_rating) / (max_rating - min_rating)

        gcdc_pd.insert(len(gcdc_pd.columns), 'rescaled_label', rescaled_label)  # just 8, after the original score

        return gcdc_pd
    # end def

    #
    def get_np_info(self, config):
        #
        if config.use_nounp:
            np_file_path = os.path.join(config.data_dir, "nounp")  # stanza 1.1.1 ver
            cur_fold = str(config.cur_fold)

            len_np = 3  # need to be passed from outer 
            len_np = str(len_np)

            if config.use_coref:
                file_map_np_sbw = config.gcdc_domain + "_nounp_" + len_np + "_lower_coref" + "_fold_" + cur_fold + ".csv"
            else:
                file_map_np_sbw = config.gcdc_domain + "_nounp_" + len_np + "_lower" + "_fold_" + cur_fold + ".csv"
            
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
        """Return id-converted corpus
        :param num_fold:
        :return: map of id-converted sentence
        """
        train_corpus = None
        valid_corpus = None
        test_corpus = None
        y_train = None
        y_valid = None
        y_test = None

        if num_fold < 0:  # use whole dataset
            train_corpus = self.train_corpus
            test_corpus = self.test_corpus
            if self.valid_corpus is not None:
                valid_corpus = self.valid_corpus

            y_train = self.train_pd['labelA'].values
            y_test = self.test_pd['labelA'].values
            if self.valid_pd is not None:
                y_valid = self.valid_pd['labelA'].values

            y_train = np.subtract(y_train, 1)  # for the loss function, scale to 0-2 from 1-3
            y_test = np.subtract(y_test, 1)
            if y_valid is not None:
                y_valid = np.subtract(y_valid, 1)

            # y_train = self.train_pd['rescaled_label'].values
            # y_test = self.test_pd['rescaled_label'].values
            # score_train = self.train_pd['labelA'].values
            # score_test = self.test_pd['labelA'].values

        else:  # use specific fold at this time
            # cur_fold_train = self.fold_train[num_fold]
            # cur_fold_test = self.fold_test[num_fold]

            # cur_fold_train = self.train_corpus[num_fold]
            # cur_fold_test = self.test_corpus[num_fold]
            # train_corpus = self._tokenize_corpus(cur_fold_train)
            # test_corpus = self._tokenize_corpus(cur_fold_test)

            y_train = self.train_pd['labelA'].values
            y_test = self.test_pd['labelA'].values
            y_train = np.subtract(y_train, 1)  # for the loss function, scale to 0-2 from 1-3
            y_test = np.subtract(y_test, 1)

            # # minority test introduced in GCDC paper (very low coherence)
            # print(Counter(y_train))
            # df_ratings = self.train_pd[['ratingA1', 'ratingA2', 'ratingA3']]
            # rr = df_ratings.values
            # for ind, cur_label in enumerate(y_train):
            #     # num_over = np.where(rr[ind]>1)
            #     num_over = sum(i < 2 for i in rr[ind])
            #     if num_over > 1:
            #         y_train[ind] = 0
            #     else:
            #         y_train[ind] = 1

            # print(Counter(y_train))

            # print(Counter(y_test))
            # df_ratings = self.test_pd[['ratingA1', 'ratingA2', 'ratingA3']]
            # rr = df_ratings.values
            # for ind, cur_label in enumerate(y_test):
            #     # num_over = np.where(rr[ind]>1)
            #     num_over = sum(i < 2 for i in rr[ind])
            #     if num_over > 1:
            #         y_test[ind] = 0
            #     else:
            #         y_test[ind] = 1
            # print(Counter(y_test))


            # self.output_bias = mean(y_train) - 1

            # y_train = cur_fold_train['rescaled_label'].values
            # y_test = cur_fold_test['rescaled_label'].values
            # score_train = cur_fold_train['labelA'].values
            # score_test = cur_fold_test['labelA'].values

            # y_train = y_train.reshape(len(y_train), 1)
            # y_test = y_test.reshape(len(y_test), 1)
            # score_train = score_train.reshape(len(score_train), 1)
            # score_test = score_test.reshape(len(score_test), 1)

        # change each sentence to id-parsed
        x_id_train, max_len_doc_train, list_len_train = self._to_id_corpus(self.train_corpus)
        x_id_test, max_len_doc_test, list_len_test = self._to_id_corpus(self.test_corpus)

        list_len = list_len_train + list_len_test

        max_len_doc = max(list_len)
        avg_len_doc = statistics.mean(list_len)
        std_len_doc = statistics.stdev(list_len)

        tid_train = self.train_pd['text_id'].values
        tid_valid = None
        tid_test = self.test_pd['text_id'].values

        # # paragraph info
        # list_num_para_train, len_para_train, list_num_sents_train = self._get_para_info(self.train_pd['text'])
        # list_num_para_test, len_para_test, list_num_sents_test = self._get_para_info(self.test_pd['text'])
        # max_num_para = max(max(list_num_para_train), max(list_num_para_test))

        # max_num_sents = max(max(list_num_sents_train), max(list_num_sents_test))        
        max_num_sents = max(max(self.num_sents_train), max(self.num_sents_test))

        # padding to len_para
        # li = [[1,2,3], [1], [1,2,3,5], [2,3]]
        # len_para_train = [cur_li + ([0] * (max_num_para - len(cur_li))) for cur_li in len_para_train]
        # len_para_test = [cur_li + ([0] * (max_num_para - len(cur_li))) for cur_li in len_para_test]
        len_para_train = []
        len_para_test = []
        
        train_data_id = {'x_data': x_id_train, 'y_label': y_train, 'tid': tid_train, 'len_para': len_para_train}
        test_data_id = {'x_data': x_id_test, 'y_label': y_test, 'tid': tid_test, 'len_para': len_para_test}
        # # train_data_id = {'x_data':x_id_train, 'y_label':y_train, 'origin_score':score_train}
        # test_data_id = {'x_data':x_id_test, 'y_label':y_test, 'origin_score':score_test}
        id_corpus = {'train': train_data_id, 'test': test_data_id}

        max_num_para = -1  # depreicated at this moment

        return id_corpus, max_len_doc, avg_len_doc, max_num_para, max_num_sents

    #
    def get_id_corpus_target(self, target_pd):
        """Return id-converted corpus
        :param num_fold:
        :return: map of id-converted sentence
        """

        corpus_data = self._sent_split_corpus(target_pd['text'].values)

        y_data = target_pd['labelA'].values
        y_data = np.subtract(y_data, 1)

        tid_data = target_pd['text_id'].values

        id_data, max_len_doc, avg_len_doc = self._to_id_corpus(corpus_data)

        return id_data, max_len_doc, avg_len_doc

    # stanza tokenizer test
    def _sent_split_corpus(self, arr_input_text):
        """ tokenize corpus given tokenizer by config file"""
        # arr_input_text = pd_input['essay'].values

        # num_over = 0
        # total_sent = 0

        import stanza  # stanford library for tokenizer
        tokenizer_stanza = stanza.Pipeline('en', processors='tokenize', use_gpu=True)

        num_sents = []
        sent_corpus = []  # tokenized to form of [doc, list of sentences]
        for cur_doc in arr_input_text:
            cur_doc = self._refine_text(cur_doc)  # cur_doc: single string
            
            # sent_list = [sent.string.strip() for sent in spacy_nlp(cur_doc).sents] # spacy style

            ## stanza version
            doc_stanza = tokenizer_stanza(cur_doc)
            sent_list = [sentence.text for sentence in doc_stanza.sentences]
           
            ## normal version
            # sent_list = self.sent_tokenzier(cur_doc)  # following exactly same way with previous works
            
            sent_corpus.append(sent_list)
            num_sents.append(len(sent_list))

        # return sent_corpus, num_sents
        return sent_corpus, num_sents

    #
    def _refine_text(self, input_text, ignore_uni=False, ignore_para=True):
        """
        custom function for pre-processing text

        :param input_text:
        :param ignore_uni: whether ignore unicode or not
        :param ignore_para: whether ignore paragraph or not (disabled now)
        :return: refined text
        """

        if ignore_uni:
            # input_text = input_text.encode('ascii', 'ignore').decode("utf-8")
            input_text = unidecode(input_text)
        input_text = input_text.lower()

        # # put space for sepearator when there is quote
        # input_text = input_text.replace('?\"', '? \"').replace('!\"', '! \"').replace('.\"', '. \"')
        # input_text = input_text.replace("...", ".")  # e.g., etc... -> etc.
        #

        # input_text = input_text.replace("- ", ". ")
        # input_text = input_text.replace("gf", "girl friend")
        # input_text = input_text.replace("--", ".")
        # input_text = input_text.replace("..", ". ")
        # input_text = input_text.replace("...", ". ")
        # input_text = input_text.replace("....", ". ")
        # input_text = input_text.replace(".....", ". ")
        # input_text = input_text.replace(",,,,,", ". ")
        # input_text = input_text.replace(" coz ", " because ")
        # input_text = input_text.replace(" w/o ", " without ")
        # input_text = input_text.replace(" w/out ", " without ")
        # input_text = input_text.replace(" w/the ", " with the ")

        # input_text = input_text.replace('"', '')

        #
        # input_text = input_text.replace("wouldn\\'t", "would not")
        # input_text = input_text.replace("couldn\\'t", "could not")
        # input_text = input_text.replace("haven\\'t", "have not")
        # input_text = input_text.replace("hasn\\'t", "has not")

        # input_text = input_text.replace('.', '. ', input_text.count('.')).replace(',', ', ', input_text.count(','))


        out_text = input_text.strip()

        #
        # # filter at word level
        # out_text = ""
        # for word in nltk.word_tokenize(input_text):
        #     if self.is_time(word):
        #         word = TIME
        #     elif self.is_date(word):
        #         word = DATE
        #
        #     elif word.endswith('-'):
        #         word = word[:-1]
        #     elif word.startswith('-'):
        #         word = word[1:]
        #     elif word.endswith('\''):
        #         word = word[:-1]
        #     elif word.startswith('\''):
        #         word = word[1:]
        #
        #     if word == "w/": word = "with"
        #
        #     out_text = out_text + word + " "
        # # end for word
        # out_text = out_text.strip()

        return out_text
    # end _refine_text
