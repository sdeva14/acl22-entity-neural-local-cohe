import numpy as np
import pandas as pd

import utils
from utils import LONG, FLOAT

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from corpus.dataset_base import Dataset_Base

class Dataset_GCDC(Dataset_Base):
	def __init__(self, data_id, config, pad_id):

		super().__init__(data_id, config, pad_id)

		# self.origin_score = data_id["origin_score"]

		return

	#
	def __len__(self):
		return len(self.y_label)

	#
	def __getitem__(self, idx):
		cur_x = self.x_data[idx]
		cur_y = self.y_label[idx]
		cur_tid = self.tid[idx]
		# cur_origin_score = self.origin_score[idx]
		# cur_len_para = self.len_para[idx]

		vec_text_input = None
		if self.pad_level == "sent" or self.pad_level == "sentence":
			vec_text_input, mask_input, len_seq, len_sents = self._pad_sent_level(cur_x)
		else:
			vec_text_input, mask_input, len_seq, len_sents = self._pad_doc_level(cur_x)
			# vec_text_input, seq_lens = self._pad_doc_level(cur_x)

		label_y = torch.LongTensor([cur_y])
		# print(label_y)
		# print(label_y.shape)

		# cur_len_para = torch.LongTensor(cur_len_para)

		# return vec_text_input, label_y, mask_input, len_seq, len_sents, cur_tid, cur_len_para
		return vec_text_input, label_y, mask_input, len_seq, len_sents, cur_tid
		# return vec_text_input, label_y, seq_lens, idx

