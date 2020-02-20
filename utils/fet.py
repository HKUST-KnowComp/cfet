import numpy as np
from utils import model_utils
import os

import torch
from torch import nn
import torch.nn.functional as F
# from pytorch_pretrained_bert.modeling import BertModel
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.modeling_bert import BertModel
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from utils import model_utils


class fet_model(nn.Module):
    def __init__(self, config, device, gres):
        super().__init__()
        self.config = config
        self.dataset = config.dataset

        self.dropout = config.dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        self.use_mlp = config.use_mlp
        self.use_bert = config.use_bert
        self.use_lstm = config.use_lstm

        self.device = device
        self.max_seq_length = config.max_seq_length
        self.mlp_hidden_dim = config.mlp_hidden_dim
        self.batch_size = config.batch_size // len(config.gpu_ids)
        self.inference_threshhold = config.inference_threshhold if config.dataset == 'cufe' else 0.5

        self.n_types = gres.n_types
        self.type_embed_dim = config.type_embed_dim
        self.type_embeddings = torch.tensor(
            np.random.normal(scale=0.01, size=(self.type_embed_dim, self.n_types)).astype(np.float32),
            device=self.device, requires_grad=True)
        self.type_embeddings = nn.Parameter(self.type_embeddings)

        self.word_vec_dim = gres.embedding_layer.embedding_dim

        self.bert_hdim = config.bert_hdim
        self.bert_use_four = config.bert_use_four
        self.concat_lstm = config.concat_lstm
        self.lstm_hidden_dim = config.lstm_hidden_dim

        if self.use_bert:
            self.bert_model = BertModel.from_pretrained(pretrained_model_name_or_path=config.model_name,
                                                        cache_dir=config.MODEL_CACHE_PATH)
                                                        # cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                        #                        'distributed_{}'.format('-1')))
            if config.freeze_bert:
                for p in self.bert_model.parameters():
                    p.requires_grad = False
            if not self.bert_use_four:
                linear_map_input_dim = self.bert_hdim + self.word_vec_dim
            else:
                linear_map_input_dim = self.bert_hdim * 4 + self.word_vec_dim

        elif self.use_lstm:
            self.bi_lstm_1 = nn.LSTM(input_size=self.word_vec_dim,
                                     hidden_size=self.lstm_hidden_dim, bidirectional=True)
            self.bi_lstm_2 = nn.LSTM(input_size=self.lstm_hidden_dim * 2,
                                     hidden_size=self.lstm_hidden_dim, bidirectional=True)

            self.zeros = torch.zeros(2, self.batch_size, self.lstm_hidden_dim, requires_grad=True)
            # self.lstm_hidden_1 = (torch.zeros(2, self.batch_size, self.lstm_hidden_dim,
            #                                   requires_grad=True),
            #                       torch.zeros(2, self.batch_size, self.lstm_hidden_dim,
            #                                   requires_grad=True))
            # self.lstm_hidden_2 = (torch.zeros(2, self.batch_size, self.lstm_hidden_dim,
            #                                   requires_grad=True),
            #                       torch.zeros(2, self.batch_size, self.lstm_hidden_dim,
            #                                   requires_grad=True))

            if self.concat_lstm:
                linear_map_input_dim = self.lstm_hidden_dim * 4 + self.word_vec_dim
            else:
                linear_map_input_dim = self.lstm_hidden_dim * 2 + self.word_vec_dim

        # build DNN
        if self.dataset == 'ufet':
            self.decoder = model_utils.MultiSimpleDecoder(linear_map_input_dim)
            self.sig_func = torch.nn.Sigmoid()

        if not self.use_mlp:
            self.linear_map1 = nn.Linear(linear_map_input_dim, self.type_embed_dim, bias=False)
            # self.linear_map = nn.Linear(linear_map_input_dim, self.n_types, bias=False)
        else:
            mlp_hidden_dim = linear_map_input_dim // 2 if self.mlp_hidden_dim is None else self.mlp_hidden_dim
            self.linear_map1 = nn.Linear(linear_map_input_dim, mlp_hidden_dim)
            self.lin1_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
            self.lin2_bn = nn.BatchNorm1d(mlp_hidden_dim)
            self.linear_map3 = nn.Linear(mlp_hidden_dim, self.type_embed_dim)

        self.loss_func = nn.BCEWithLogitsLoss()

    def run_lstm(self, context_token_seqs, lens):

        # some notes: pack padded sequence for quicker computation. rnn don't need all sequences to be the same length,
        # packing them can omit computing the padding part.
        x = torch.nn.utils.rnn.pack_padded_sequence(context_token_seqs, lens, batch_first=True)
        self.lstm_hidden_1 = (torch.zeros(2, len(context_token_seqs), self.lstm_hidden_dim, device=self.device,
                                          requires_grad=True),
                              torch.zeros(2, len(context_token_seqs), self.lstm_hidden_dim, device=self.device,
                                          requires_grad=True))
        self.lstm_hidden_2 = (torch.zeros(2, len(context_token_seqs), self.lstm_hidden_dim, device=self.device,
                                          requires_grad=True),
                              torch.zeros(2, len(context_token_seqs), self.lstm_hidden_dim, device=self.device,
                                          requires_grad=True))

        lstm_output_1, self.lstm_hidden_1 = self.bi_lstm_1(x, self.lstm_hidden_1)
        lstm_output_2, self.lstm_hidden_2 = self.bi_lstm_2(lstm_output_1, self.lstm_hidden_2)

        lstm_output_1, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output_1, batch_first=True)
        lstm_output_2, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output_2, batch_first=True)

        if self.concat_lstm:
            lstm_output = torch.cat((lstm_output_1, lstm_output_2), dim=2)
        else:
            lstm_output = lstm_output_1 + lstm_output_2
        return lstm_output

    def get_avg_token_vecs(self, token_seqs):
        lens = torch.tensor([len(seq) for seq in token_seqs], dtype=torch.float32, device=self.device).view(-1, 1)
        seqs = [torch.tensor(seq, dtype=torch.long, device=self.device) for seq in token_seqs]
        seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True,
                                               padding_value=self.embedding_layer.padding_idx)
        token_vecs = self.embedding_layer(seqs)
        vecs_avg = torch.div(torch.sum(token_vecs, dim=1), lens)
        return vecs_avg

    def define_loss(self, logits, targets, data_type):
        self.device = self.linear_map1.weight.device
        if data_type == 'onto':
            loss = self.loss_func(logits, targets)
            return loss
        if data_type == 'wiki':
            gen_cutoff, fine_cutoff, final_cutoff = self.config.ANSWER_NUM_DICT['gen'], self.config.ANSWER_NUM_DICT['kb'], \
                                                    self.config.ANSWER_NUM_DICT[data_type]
        else:
            gen_cutoff, fine_cutoff, final_cutoff = self.config.ANSWER_NUM_DICT['gen'], self.config.ANSWER_NUM_DICT[
                'kb'], None
        loss = 0.0
        comparison_tensor = torch.Tensor([1.0]).to(self.device)
        gen_targets = targets[:, :gen_cutoff]
        fine_targets = targets[:, gen_cutoff:fine_cutoff]
        gen_target_sum = torch.sum(gen_targets, 1)
        fine_target_sum = torch.sum(fine_targets, 1)

        if torch.sum(gen_target_sum.data) > 0:
            gen_mask = torch.squeeze(torch.nonzero(torch.min(gen_target_sum.data, comparison_tensor)), dim=1)
            gen_logit_masked = logits[:, :gen_cutoff][gen_mask, :]
            gen_target_masked = gen_targets.index_select(0, gen_mask)
            gen_loss = self.loss_func(gen_logit_masked, gen_target_masked)
            # gen_loss = self.get_loss(gen_logit_masked, gen_target_masked)
            loss += gen_loss
        if torch.sum(fine_target_sum.data) > 0:
            fine_mask = torch.squeeze(torch.nonzero(torch.min(fine_target_sum.data, comparison_tensor)), dim=1)
            fine_logit_masked = logits[:, gen_cutoff:fine_cutoff][fine_mask, :]
            fine_target_masked = fine_targets.index_select(0, fine_mask)
            fine_loss = self.loss_func(fine_logit_masked, fine_target_masked)
            # fine_loss = self.get_loss(fine_logit_masked, fine_target_masked)

            loss += fine_loss

        if not data_type == 'kb':
            if final_cutoff:
                finer_targets = targets[:, fine_cutoff:final_cutoff]
                logit_masked = logits[:, fine_cutoff:final_cutoff]
            else:
                logit_masked = logits[:, fine_cutoff:]
                finer_targets = targets[:, fine_cutoff:]
            if torch.sum(torch.sum(finer_targets, 1).data) > 0:
                finer_mask = torch.squeeze(
                    torch.nonzero(torch.min(torch.sum(finer_targets, 1).data, comparison_tensor)), dim=1)
                finer_target_masked = finer_targets.index_select(0, finer_mask)
                logit_masked = logit_masked[finer_mask, :]
                layer_loss = self.loss_func(logit_masked, finer_target_masked)
                # layer_loss = self.get_loss(logit_masked, finer_target_masked)

                loss += layer_loss
        return loss

    def get_uw_loss(self, logits, targets, gres):
        loss = 0.0
        # comparison_tensor = torch.Tensor([1.0]).cuda()
        gen_logits = logits[:, gres.gen_idxs]
        fine_logits = logits[:, gres.fine_idxs]
        gen_targets = targets[:, gres.gen_idxs]
        fine_targets = targets[:, gres.fine_idxs]
        gen_target_sum = torch.sum(gen_targets, 1)
        fine_target_sum = torch.sum(fine_targets, 1)

        if torch.sum(gen_target_sum.data) > 0:
            # gen_mask = torch.squeeze(torch.nonzero(torch.min(gen_target_sum.data, comparison_tensor)), dim=1)
            batch_gen_mask = torch.squeeze(torch.nonzero(gen_target_sum.data), dim=1)
            gen_logit_masked = gen_logits[batch_gen_mask, :]
            gen_target_masked = gen_targets[batch_gen_mask, :]
            # gen_loss = self.loss_func(gen_logit_masked, gen_target_masked)
            if len(batch_gen_mask) > 0:
                gen_loss = self.get_loss(gen_logit_masked, gen_target_masked)
                loss += gen_loss
        if torch.sum(fine_target_sum.data) > 0:
            # fine_mask = torch.squeeze(torch.nonzero(torch.min(fine_target_sum.data, comparison_tensor)), dim=1)
            batch_fine_mask = torch.squeeze(torch.nonzero(fine_target_sum.data), dim=1)
            fine_logit_masked = fine_logits[batch_fine_mask, :]
            fine_target_masked = fine_targets[batch_fine_mask, :]
            # fine_loss = self.loss_func(fine_logit_masked, fine_target_masked)
            if len(batch_fine_mask) > 0:
                fine_loss = self.get_loss(fine_logit_masked, fine_target_masked)
                loss += fine_loss

        return loss


    def get_loss(self, scores, true_type_vecs, margin=1.0):
        tmp1 = torch.sum(true_type_vecs * F.relu(margin - scores), dim=1)
        tmp2 = torch.sum((1 - true_type_vecs) * F.relu(margin + scores), dim=1)

        loss = torch.mean(torch.add(tmp1, tmp2))
        return loss

    def inference_full(self, logits):
        if self.dataset == 'ufet':
            logits = self.sig_func(logits)
        label_preds = list()
        for i in range(len(logits)):
            pred_idxs = (logits[i] > self.inference_threshhold).nonzero().squeeze(1).tolist()
            if len(pred_idxs) == 0:
                v, pred_idxs = logits[i].max(0)
                pred_idxs = [pred_idxs.tolist()]
            # pred_idxs = sorted(range(len(logits[i])), key=lambda sub: logits[i][sub])[-5:]

            label_preds.append(pred_idxs)
        return label_preds

    def forward(self, input_dataset, gres):
        """

        :param context_token_list: list of sequences of different length
        :param mention_token_idxs:
        :param mstr_token_seqs:
        :return: logits: (batch size, n_types)
        """
        self.device = self.linear_map1.weight.device

        if self.use_bert:
            context_token_list, mention_token_idxs, mstr_token_seqs, mstr_token_seqs_lens = input_dataset
            context_token_tensor = torch.zeros(len(context_token_list), self.max_seq_length,
                                               device=self.device, dtype=torch.long)
            context_token_tensor[:, :min(context_token_list.size(1), self.max_seq_length)] \
                = context_token_list[:, :self.max_seq_length]

            bert_output = self.bert_model(context_token_tensor)
            # bert_context_hidden = bert_output
            if not self.bert_use_four:
                context_hidden = bert_output[0]
            else:
                bert_context_hidden = bert_output[0] + bert_output[2:5]
                context_hidden = torch.cat(bert_context_hidden, dim=2)
            context_hidden = context_hidden[list(range(context_hidden.size(0))), mention_token_idxs, :]

        elif self.use_lstm:
            (context_token_tensor, mention_token_idx_tensor, mstr_token_seqs, mstr_token_seqs_lens) = input_dataset

            context_token_seqs, seq_lens, mention_token_idxs, back_idxs \
                = model_utils.get_len_sorted_context_seqs_input(self.device, context_token_tensor, mention_token_idx_tensor)
            context_token_seqs = gres.embedding_layer(context_token_seqs.cpu()).to(self.device)

            context_hidden = self.run_lstm(context_token_seqs, seq_lens)
            context_hidden = context_hidden[list(range(context_token_seqs.size(0))), mention_token_idxs, :]
            context_hidden = context_hidden[back_idxs]

        mstr_vecs_avg = torch.div(torch.sum(mstr_token_seqs, dim=1), mstr_token_seqs_lens)

        cat_output = torch.cat((context_hidden, mstr_vecs_avg), dim=1)


        if self.dataset == 'ufet':
        # if False:
            # with decoder has better result??
            logits = self.decoder(cat_output, self.config.dataset_type)
        else:
            if not self.use_mlp:
                mention_reps = self.linear_map1(self.dropout_layer(cat_output))
            else:
                l1_output = self.linear_map1(self.dropout_layer(cat_output))
                # l1_output = F.relu(l1_output)
                l1_output = self.lin1_bn(F.relu(l1_output))
                # mention_reps = self.linear_map2(F.dropout(l1_output, self.dropout, training))
                l2_output = self.linear_map2(self.dropout_layer(l1_output))
                l2_output = self.lin2_bn(F.relu(l2_output))
                mention_reps = self.linear_map3(self.dropout_layer(l2_output))


            logits = torch.matmul(mention_reps.view(-1, 1, self.type_embed_dim),
                                  self.type_embeddings.view(-1, self.type_embed_dim, self.n_types))
            logits = logits.view(-1, self.n_types)
        return logits
