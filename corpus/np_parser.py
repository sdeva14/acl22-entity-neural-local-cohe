# import benepar
# benepar_pos_parser = benepar.Parser("benepar_en2")  # or "benepar_en2_large"

import nltk
# import re

class NP_Parser(object):

    def __init__(self, tokenizer, len_threshold_np, parser=None):
        
        if parser is None:
            self.parser = benepar_pos_parser
        else:
            self.parser = parser

        self.tokenizer = tokenizer
        self.len_threshold_np = len_threshold_np

    # end __init__

    #
    def find_sub_list(self, sub_list, origin_list):
        results=[]  # consists of pairs of (first index and the last)
        len_sub_list=len(sub_list)
        for ind in (i for i,e in enumerate(origin_list) if e==sub_list[0]):
            if origin_list[ind : ind+len_sub_list]==sub_list:
                results.append((ind, ind+len_sub_list-1))

        return results
    # end def find_sub_list

    # mapping the location of NPs from original index to the index in the subword tokenization (stage3)
    def map_np_sbw(self, sents, list_sent_map_np, list_sent_loc_subword):

        max_num_np = 0
        list_sent_np_subword = []  # list of map, ith sentence, map; key: NP, value: list of pairs for spans
        for ind_sent, cur_sent in enumerate(sents):
            # map_nps_subword = dict()  # mapping NPs in subword tokenized results, key: NP, value: list of span pair
            list_nps_subword = []  # list of pair (NP, a pair of span)
            cur_map_np = list_sent_map_np[ind_sent]  # mapping for NPs at the ith sentence
            cur_map_subword = list_sent_loc_subword[ind_sent]  # mapping for changed loc in the subword tokenization

            for cur_np, list_span_origin in cur_map_np.items():
                # print(cur_np)

                for cur_span in list_span_origin:
                    # print(cur_span)
                    start_origin = cur_span[0]
                    end_origin = cur_span[1]

                    # print(cur_map_subword[start_origin])
                    start_span_subword = cur_map_subword[start_origin][0]
                    end_span_subword = cur_map_subword[end_origin][1]

                    np_span_subword = (start_span_subword, end_span_subword)

                    # list_np_spans = []
                    # if cur_np in map_nps_subword:
                    #     list_np_spans = map_nps_subword[cur_np]
                    # list_np_spans.append(np_span_subword)
                    # map_nps_subword[cur_np] = list_np_spans
                    list_nps_subword.append((cur_np, np_span_subword))

            # cur_num_np = len(list_np_spans)
            cur_num_np = len(list_nps_subword)
            if cur_num_np > max_num_np:
                max_num_np = cur_num_np

            # end for cur_np

            # print(cur_map_np)
            # print(map_nps_subword)
            # print("\n")

            # list_sent_np_subword.append(map_nps_subword)
            list_sent_np_subword.append(list_nps_subword)

        # end for ind_sent

        return list_sent_np_subword, max_num_np

    # track the location in the subword tokenization (stage2)
    def track_sbw_loc(self, sents):
        list_sent_loc_subword = []  # list of list of pairs, ith sentence, list of (origin_ind, subword spans)
        for ind_sent, cur_sent in enumerate(sents):
            # print(cur_sent)

            words_origin = nltk.word_tokenize(cur_sent)
            list_map_subword = []  # current sentence; list of pairs, (loc in origin, loc in subword)

            # tokenize word-by-word to track when it is dividied by subwords
            subword_ind = 0
            for cur_ind_origin, cur_word in enumerate(words_origin):
                # subword tokenization
                list_subword = self.tokenizer.tokenize(cur_word)
                len_subwords = len(list_subword)

                if cur_word == "." or cur_word == "?" or cur_word == "!":
                    len_subwords = 1

                # store
                cur_pair = (subword_ind, subword_ind + len_subwords - 1)  # (start_index of the current sbw, end_index of the current sbw)
                list_map_subword.append(cur_pair)

                subword_ind = subword_ind + len_subwords  # the start of the next item

            # end for cur_word
            # print(list_map_subword)
            list_sent_loc_subword.append(list_map_subword)

        # end for ind_sent

        return list_sent_loc_subword

    # extract the location of NPs in the original sentences (stage1)
    def extract_noun_pharse(self, sents):

        # tree = parser.parse(sent)
        trees_sent = self.parser.parse_sents(sents)

        # for subtree in tree.subtrees(filter=lambda x: x.label() == 'NP'):
        #     print(subtree.leaves())

        list_sent_map_np = []  # list of dict, each dict represents nps in the ith sentence
        # len_threshold_np = 10  # only consider np consists of less than 3 words
        ind_np_origin = []  ## index of NPs in the original sentence before subword
        np_sents = []  ## list of list: NPs in sentences
        for ind_sent, cur_tree in enumerate(trees_sent):

            cur_map_np = dict()  # key: np, value: list of indexes in the origianl sentence

            cur_list_np = []  # list of list, each list represents NP
            for subtree in cur_tree.subtrees(filter=lambda x: x.label() == 'NP'):
                # print(subtree.leaves())

                cur_np = subtree.leaves()
                if len(cur_np) <= self.len_threshold_np:
                    # cur_np = ' '.join(cur_np)
                    cur_list_np.append(cur_np)  # np is a list
                # end if
            # end for subtree

            # track the index of each NP in the original sentence
            # we need ith word information instead of exact location w.r.t characters
            cur_sent = sents[ind_sent]
            # padded_sent = " " + cur_sent + " "
            # print(cur_sent)
            cur_words = nltk.word_tokenize(cur_sent)

            for cur_np in cur_list_np:
                ## extract ith word information
                ind_list = self.find_sub_list(cur_np, cur_words)

                # print(cur_np)
                # print(ind_list)
                # print(" ")

                str_cur_np = " ".join(cur_np)
                cur_map_np[str_cur_np] = ind_list
            # end for cur_np

            # print(cur_map_np)
            list_sent_map_np.append(cur_map_np)

        # end for cur_tree

        return list_sent_map_np
    # end def

    # identify the loc of NPs in the subword tokenization for a given document (wrapper)
    def identify_loc_NP_sbw(self, sents):

        # stage1; # list of dict, each dict represents nps in the ith sentence
        list_sent_map_np = self.extract_noun_pharse(sents)
        # stage2; # list of list of pairs, ith sentence, list of (origin_ind, subword spans)
        list_sent_loc_subword = self.track_sbw_loc(sents)
        # stage3; # list of map, ith sentence, map; key: NP, value: list of pairs for spans
        list_sent_np_subword, max_num_np = self.map_np_sbw(sents, list_sent_map_np, list_sent_loc_subword)

        return list_sent_np_subword, list_sent_loc_subword, list_sent_map_np, max_num_np
    # end def

# end class