import torch
from torch import nn
from tqdm import tqdm
import logging
from collections import namedtuple
import random
import json
from os.path import join
import time

from pytorch_transformers.tokenization_bert import BertTokenizer

import config
from utils import utils, datautils, model_utils

ModelSample = namedtuple('ModelSample', [
    'mention_id',
    'mstr_token_seq',
    'context_token_seq_bert',
    'mention_token_idx_bert',
    'labels'
])


class GlobalRes:
    def __init__(self, config):

        self.only_general_types = config.only_general_types
        self.without_general_types = config.without_general_types
        if config.dataset == 'ufet':
            self.ANSWER_NUM_DICT = {"open": 10331, "onto": 89, "wiki": 4600, "kb": 130, "gen": 9}
            self.n_types = self.ANSWER_NUM_DICT[config.dataset_type]
            with open(config.UFET_FILES['ufet_training_type_set'], 'r') as r:
                self.type_id2type_dict = {i: x.strip() for i, x in enumerate(r.readlines()) if i < self.n_types}
            self.type2type_id_dict = {tp: id for id, tp in self.type_id2type_dict.items()}
            self.general_type_set = set(
                [v for k, v in self.type_id2type_dict.items() if k < self.ANSWER_NUM_DICT['gen']])

        elif config.GENERAL_TYPES_MAPPING:
            self.types2general_types_mapping = datautils.load_pickle_data(config.GENERAL_TYPES_MAPPING)
            self.general_type_set = set([v for k, vs in self.types2general_types_mapping.items() for v in vs])
            if self.only_general_types:
                self.n_types = len(self.general_type_set)
                self.type_id2type_dict = {i: x for i, x in enumerate(self.general_type_set)}

            else:
                if self.without_general_types:
                    self.types2general_types_mapping = {k: v for k, v in self.types2general_types_mapping.items()}
                    self.type_id2type_dict = {i: x for i, x in enumerate(self.types2general_types_mapping)}
                else:
                    self.type_id2type_dict = {i: x for i, x in enumerate(self.types2general_types_mapping)}
                self.n_types = len(self.type_id2type_dict)

            self.type2type_id_dict = {tp: id for id, tp in self.type_id2type_dict.items()}
            # print(self.type2type_id_dict['其他'])
            self.gen_idxs = [k for k, v in self.type_id2type_dict.items() if v in self.general_type_set]
            self.fine_idxs = [k for k, v in self.type_id2type_dict.items() if v not in self.general_type_set]

        tic = time.time()
        WORD_VECS_FILE = f'/data/cleeag/word_embeddings/{config.mention_tokenizer_name}/' \
                         f'{config.mention_tokenizer_name}_tokenizer&vecs.pkl'
        print('loading {} ...'.format(WORD_VECS_FILE), end=' ', flush=True)
        self.token2token_id_dict, self.token_vecs = datautils.load_pickle_data(WORD_VECS_FILE)
        print(f'done, {time.time() - tic :.2f} secs taken.', flush=True)
        self.zero_pad_token_id = self.token2token_id_dict[config.TOKEN_ZERO_PAD]
        self.mention_token_id = self.token2token_id_dict[config.TOKEN_MENTION]
        self.unknown_token_id = self.token2token_id_dict[config.TOKEN_UNK]
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(self.token_vecs))
        self.embedding_layer.padding_idx = self.token2token_id_dict[config.TOKEN_ZERO_PAD]
        self.embedding_layer.weight.requires_grad = False
        self.embedding_layer.share_memory()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)


def model_samples_from_json(json_file_path):
    with open(json_file_path, 'r') as r:
        samples = [json.loads(line) for line in r.readlines()]
    return samples


def samples_to_tensor(config, gres, samples):
    """

    :param config:
    :param gres:
    :param samples:
    :return:
    context_token_tensor: tokenized context sequence. mention is tokenized with [MASK]
    mention_token_idx_tensor: [MASK] token's index in the sequence. use to retrieve computed context representation
    mstr_token_seqs: tokenized sequence of the mention string. applied embedding vectors already, so in tensor form
    mstr_token_seqs_lens: length of each mention string
    type_vecs: label vector

    for lstm:
    seq_lens: sorted by length of sequence
    back_idxs: id to recover sequence by batch (dim 1)
    """


    context_token_tensor = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s['sentence'][:config.max_seq_length]) if config.use_bert else torch.tensor(s['sentence'])
         for s in samples],
        batch_first=True,
        padding_value=0)

    mstr_token_seqs = [torch.tensor([gres.token2token_id_dict.get(x, gres.unknown_token_id)
                                     for x in gres.tokenizer.tokenize(s['mention'])])
                       for s in samples]
    mstr_token_seqs_lens = torch.tensor([len(seq) if len(seq) > 0 else 1 for seq in mstr_token_seqs], dtype=torch.float32).view(-1, 1)
    mstr_token_seqs = nn.utils.rnn.pad_sequence(mstr_token_seqs, batch_first=True, padding_value=0)
    mstr_token_seqs = gres.embedding_layer(mstr_token_seqs)

    max_seq_length = config.max_seq_length if config.use_bert else context_token_tensor.shape[1]
    mention_token_idx_tensor = torch.tensor([s['mention_token_index_in_sent']
                                             if s['mention_token_index_in_sent'] < max_seq_length
                                             else max_seq_length - 1 for s in samples], dtype=torch.long)

    if config.dataset == 'ufet':
        type_vecs = torch.tensor([
            utils.onehot_encode(
                [gres.type2type_id_dict.get(x) for x in s['types'] if x in gres.type2type_id_dict],
                len(gres.type2type_id_dict))
            for s in samples],
            dtype=torch.float32)
    else:
        type_vecs = torch.tensor([
            utils.onehot_encode(
                [gres.type2type_id_dict[x] for x in general_mapping(s['types'], gres)], len(gres.type2type_id_dict))
            for s in samples],
            dtype=torch.float32)
    return (context_token_tensor, mention_token_idx_tensor, mstr_token_seqs, mstr_token_seqs_lens), type_vecs



def general_mapping(type_ls, gres):
    if config.GENERAL_TYPES_MAPPING and config.dataset != 'ufet':
        # tmp1 = [x for x in type_ls if x not in gres.general_type_set and x in gres.types2general_types_mapping]
        tmp1 = [x for x in type_ls if x in gres.types2general_types_mapping]
        tmp2 = [z for y in [gres.types2general_types_mapping.get(x)
                            for x in type_ls if x in gres.types2general_types_mapping] for z in y]
        if gres.only_general_types:
            return list(set(tmp2))
        elif gres.without_general_types:
            return list(set(tmp1))
        else:
            return list(set(tmp1 + tmp2))
    else:
        return type_ls


def get_general_types(type_ls, gres):
    tmp2 = [z for y in [gres.types2general_types_mapping.get(x)
                        for x in type_ls if x in gres.types2general_types_mapping] for z in y]
    return list(set(tmp2))
