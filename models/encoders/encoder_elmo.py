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

import torch.nn as nn
import torch

# from pytorch_transformers import XLNetConfig, XLNetModel  # old-version
# from transformers import T5Model, T5Tokenizer
# from transformers import BertTokenizer, BertModel
from allennlp.modules.elmo import Elmo

class Encoder_ELMo(nn.Module):

    def __init__(self, config, x_embed):
        super().__init__()

        # elmo small
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"

        self.model = Elmo(options_file, weight_file, num_output_representations=2,
                dropout=0.5, requires_grad=False)  # use this as a embedding layer
        # self.encoder_out_size = self.model.config.d_model  # 1024 for t-large
        self.encoder_out_size = 256

        return
    # end __init__

    #
    def forward(self, text_inputs, mask_input, len_seq, mode=""):
        encoder_out = []
        self.model.eval()

        with torch.no_grad():
            # encoder_out = self.model(input_ids=text_inputs, attention_mask=mask_input)[0]
            encoder_out = self.model(text_inputs)
            encoded_repr = encoder_out["elmo_representations"]  ## it is a list of tensors
            encoded_repr = torch.stack(encoded_repr)

            encoded_repr = encoded_repr.transpose(0, 1)  # (batch, 2, token_num, dim)

            mask = encoder_out["mask"]
            mask = mask.repeat(1, 2)  # copy to mask bidirectional
            mask = mask.reshape(mask.size(0), 2, -1)
            mask = mask.unsqueeze(3)
            encoded_repr = encoded_repr * mask

        return encoded_repr

    #
    def forward_skip(self, x_input, mask, len_seq, mode=""):
        # ''' skip embedding part when embedded input is given '''
        encoder_out = x_input

        return encoder_out
    # end forward