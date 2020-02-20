import numpy as np
from tqdm import tqdm
import pickle as pkl
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F

from utils import exp_utils, utils, fet
import config


class MultiSimpleDecoder(nn.Module):
    """
      Simple decoder in multi-task setting.
    """

    def __init__(self, output_dim):
        super(MultiSimpleDecoder, self).__init__()
        self.linear = nn.Linear(output_dim, config.ANSWER_NUM_DICT['open'],
                                bias=False)  # (out_features x in_features)

    def forward(self, inputs, output_type):
        if output_type == "open":
            return self.linear(inputs)
        elif output_type == 'wiki':
            return F.linear(inputs, self.linear.weight[:config.ANSWER_NUM_DICT['wiki'], :], self.linear.bias)
        elif output_type == 'kb':
            return F.linear(inputs, self.linear.weight[:config.ANSWER_NUM_DICT['kb'], :], self.linear.bias)
        elif output_type == 'gen':
            return F.linear(inputs, self.linear.weight[:config.ANSWER_NUM_DICT['gen'], :], self.linear.bias)
        else:
            raise ValueError('Decoder error: output type not one of the valid')

def build_lstm_hidden(config):
    batch_size = config.batch_size
    lstm_hidden_dim = config.lstm_hidden_dim
    lstm_hidden_1 = (torch.zeros(batch_size, 2 , lstm_hidden_dim,
                                 requires_grad=True),
                     torch.zeros(batch_size, 2 , lstm_hidden_dim,
                                 requires_grad=True))
    lstm_hidden_2 = (torch.zeros(batch_size, 2 , lstm_hidden_dim,
                                 requires_grad=True),
                     torch.zeros(batch_size, 2 , lstm_hidden_dim,
                                 requires_grad=True))
    return lstm_hidden_1, lstm_hidden_2

def build_hierarchy_vecs(type_vocab, type_to_id_dict):
    from utils import utils

    n_types = len(type_vocab)
    l1_type_vec = np.zeros(n_types, np.float32)
    l1_type_indices = list()
    child_type_vecs = np.zeros((n_types, n_types), np.float32)
    for i, t in enumerate(type_vocab):
        p = utils.get_parent_type(t)
        if p is None:
            l1_type_indices.append(i)
            l1_type_vec[type_to_id_dict[t]] = 1
        else:
            child_type_vecs[type_to_id_dict[p]][type_to_id_dict[t]] = 1
    l1_type_indices = np.array(l1_type_indices, np.int32)
    return l1_type_indices, l1_type_vec, child_type_vecs

def get_len_sorted_context_seqs_input(device, context_token_list, mention_token_idxs):
    data_tups = list(enumerate(zip(context_token_list, mention_token_idxs)))
    data_tups.sort(key=lambda x: -len(x[1][0]))
    sorted_seqs = [x[1][0] for x in data_tups]
    mention_token_idxs = [x[1][1] for x in data_tups]
    idxs = [x[0] for x in data_tups]
    back_idxs = [0] * len(idxs)
    for i, idx in enumerate(idxs):
        back_idxs[idx] = i

    back_idxs = torch.tensor(back_idxs, dtype=torch.long, device=device)
    # seqs, seq_lens = get_seqs_torch_input(device, seqs)
    seq_lens = torch.tensor([len(seq) for seq in sorted_seqs], dtype=torch.long, device=device)
    sorted_seqs_tensor_list = [torch.tensor(seq, dtype=torch.long, device=device) for seq in sorted_seqs]
    padded_sorted_seqs_tensor = torch.nn.utils.rnn.pad_sequence(sorted_seqs_tensor_list, batch_first=True)
    mention_token_idxs = torch.tensor(mention_token_idxs, dtype=torch.long, device=device)
    return padded_sorted_seqs_tensor, seq_lens, mention_token_idxs, back_idxs


def check_breakdown_performance():
    # run = '11_21_2333'
    # run = '11_22_1324'

    run = '11_21_2314'
    # run = '11_22_1328'


    run = '11_21_2338'
    # run = '11_21_2340'

    # model_path = f'/data/cleeag/distant_supervision_data/training_data/train-wiki_data_zh-cufe_type2general_zh-bert/models/bert-base-chinese-{run}.pt'
    # res_dir = f'/data/cleeag/distant_supervision_data/training_data/train-wiki_data_zh-cufe_type2general_zh-bert/{run}-results'
    # sample_path = '/data/cleeag/distant_supervision_data/training_data/train-wiki_data_zh-cufe_type2general_zh-bert/test.pkl'

    # model_path = f'/data/cleeag/distant_supervision_data/training_data/train-wiki_data_zh-cufe_type2general_zh-fasttext/models/bi-lstm-{run}.pt'
    # res_dir = f'/data/cleeag/distant_supervision_data/training_data/train-wiki_data_zh-cufe_type2general_zh-fasttext/{run}-results'
    # sample_path = '/data/cleeag/distant_supervision_data/training_data/train-wiki_data_zh-cufe_type2general_zh-fasttext/test.pkl'

    model_path = f'/data/cleeag/ufet_data/training_data/ufet-mixed-glove/models/bi-lstm-{run}.pt'
    res_dir = f'/data/cleeag/ufet_data/training_data/ufet-mixed-glove/{run}-results'
    sample_path = '/data/cleeag/ufet_data/training_data/ufet-mixed-glove/test.pkl'

    model_path = f'/data/cleeag/ufet_data/training_data/ufet-mixed-bert/models/bert-base-cased-{run}.pt'
    res_dir = f'/data/cleeag/ufet_data/training_data/ufet-mixed-bert/{run}-results'
    sample_path = '/data/cleeag/ufet_data/training_data/ufet-mixed-bert/test.pkl'

    run = '11_21_1848'

    model_path = f'/data/cleeag/distant_supervision_data/training_data/crowd_data-fasttext/models/bi-lstm-{run}.pt'
    res_dir = f'/data/cleeag/distant_supervision_data/training_data/crowd_data-fasttext/{run}-results'
    sample_path = '/data/cleeag/distant_supervision_data/training_data/crowd_data-fasttext/test.pkl'

    run = '11_23_1135'

    model_path = f'/data/cleeag/distant_supervision_data/training_data/train-wiki_data_zh-cufe_type2general_zh-babylon/models/bi-lstm-{run}.pt'
    res_dir = f'/data/cleeag/distant_supervision_data/training_data/train-wiki_data_zh-cufe_type2general_zh-babylon/{run}-results'
    sample_path = '/data/cleeag/distant_supervision_data/training_data/train-wiki_data_zh-cufe_type2general_zh-babylon/test.pkl'

    gres = exp_utils.GlobalRes(config)

    samples = pkl.load(file=open(sample_path, 'rb'))
    true_labels_dict = {s['mention_id']: [gres.type2type_id_dict.get(x) for x in
                                               exp_utils.general_mapping(s['types'], gres)] for s in samples}
    if config.dataset == 'cufe':
        gen_true_labels_dict = {k:[v for v in vs if v in gres.gen_idxs] for k, vs in true_labels_dict.items()}
        gen_true_labels_dict = {k: v for k, v in gen_true_labels_dict.items() if len(v) > 0}
        fine_true_labels_dict = {k:[v for v in vs if v not in gres.gen_idxs] for k, vs in true_labels_dict.items()}
        fine_true_labels_dict = {k: v for k, v in fine_true_labels_dict.items() if len(v) > 0}

    if config.dataset == 'ufet':
        fine_true_labels_dict = {k: v for k, v in true_labels_dict.items() if len(v) > 0}



    device = torch.device('cuda:3')
    model = fet.fet_model(config, device, gres)
    model_state_dict = torch.load(model_path)
    model_state_dict = {'.'.join(k.split('.')[1:]): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    batch_size = 32
    n_batches = (len(samples) + batch_size - 1) // batch_size
    gen_pred_labels_dict, fine_pred_labels_dict = dict(), dict()
    logits_ls = []
    gold_ls = []
    for i in tqdm(range(n_batches)):
        batch_beg, batch_end = i * batch_size, min((i + 1) * batch_size, len(samples))
        batch_samples = samples[batch_beg:batch_end]
        input_dataset, type_vecs = exp_utils.samples_to_tensor(config, gres, batch_samples)
        input_dataset = tuple(x.to(device) for x in input_dataset)
        gold_ls.extend([x.data.numpy() for x in type_vecs])
        with torch.no_grad():
            logits = model(input_dataset, gres)
            preds = model.inference_full(logits)

        if config.dataset == 'cufe':

            for j, (sample, type_ids_pred, sample_logits) in enumerate(
                    zip(batch_samples, preds, logits.data.cpu().numpy())):
                gen_labels = [x for x in type_ids_pred if x in gres.gen_idxs]
                fine_labels = [x for x in type_ids_pred if x not in gres.gen_idxs]
                gen_pred_labels_dict[sample['mention_id']] = gen_labels
                fine_pred_labels_dict[sample['mention_id']] = fine_labels
                logits_ls.append(sample_logits)

        elif config.dataset == 'ufet':
            for j, (sample, type_ids_pred, sample_logits) in enumerate(
                    zip(batch_samples, preds, logits.data.cpu().numpy())):
                fine_labels = type_ids_pred
                fine_pred_labels_dict[sample['mention_id']] = fine_labels
                logits_ls.append(sample_logits)

    maf1_gen, ma_p_gen, ma_r_gen = utils.macrof1(gen_true_labels_dict, gen_pred_labels_dict, return_pnr=True)
    maf1_fine, ma_p_fine, ma_r_fine = utils.macrof1(fine_true_labels_dict, fine_pred_labels_dict, return_pnr=True)
    # mrr = utils.mrr(logits_ls, gold_ls)
    # mrr = utils.mrr(logits_ls, gold_ls)
    mrr = utils.mrr(logits_ls, gold_ls)

    with open(join(res_dir, 'breakdown.txt'), 'w') as w:
        w.write(f'test gen evaluation result: macro_p={ma_p_gen:.4f}, macro_r={ma_r_gen:.4f}, macro_f={maf1_gen:.4f}\n')
        w.write(f'test fine evaluation result: macro_p={ma_p_fine:.4f}, macro_r={ma_r_fine:.4f}, macro_f={maf1_fine:.4f}\n')
        w.write(f'mrr: {mrr:.4f}')


def get_avg_token_vecs(device, embedding_layer: nn.Embedding, token_seqs):
    lens = torch.tensor([len(seq) for seq in token_seqs], dtype=torch.float32, device=device).view(-1, 1)
    seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in token_seqs]
    seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True,
                                           padding_value=embedding_layer.padding_idx)
    token_vecs = embedding_layer(seqs)
    vecs_avg = torch.div(torch.sum(token_vecs, dim=1), lens)
    return vecs_avg

def eval_fetel(config, gres, model, samples, true_labels_dict):
    batch_size = config.batch_size
    n_batches = (len(samples) + batch_size - 1) // batch_size
    losses = list()
    pred_labels_dict = dict()
    pred_labels_dict_test = dict()
    result_objs, incorrect_result_objs = list(), list()
    if hasattr(model, 'module'):
        device = model.module.device
        model.module.eval()
    else:
        device = model.device
        model.eval()

    for i in range(n_batches):
        batch_beg, batch_end = i * batch_size, min((i + 1) * batch_size, len(samples))
        batch_samples = samples[batch_beg:batch_end]
        input_dataset, type_vecs = exp_utils.samples_to_tensor(config, gres, batch_samples)
        input_dataset = tuple(x.to(device) for x in input_dataset)
        type_vecs = type_vecs.to(device)

        with torch.no_grad():
            logits = model(input_dataset, gres)
            if hasattr(model, 'module'):
                loss = model.module.get_loss(logits, type_vecs)
                if config.dataset == 'cufe':
                    preds = model.module.inference_full(logits)

                else:
                    preds = model.module.inference_full(logits)

            else:
                loss = model.get_loss(logits, type_vecs)
                preds = model.inference_full(logits)

        losses.append(loss)

        for j, (sample, type_ids_pred, sample_logits) in enumerate(
                zip(batch_samples, preds, logits.data.cpu().numpy())):
            labels = type_ids_pred
            pred_labels_dict[sample['mention_id']] = labels

            if len(type_ids_pred) > 0:
                if config.dataset == 'cufe' and not config.without_general_types:
                    general_types = exp_utils.get_general_types(sample['types'], gres)
                    result_dict = {'mention_id': sample['mention_id'],
                                   'mention': sample['mention'],
                                   'labels': list(set(sample['types'] + general_types)),
                                   'general_types': general_types,
                                   'preds': [gres.type_id2type_dict[x] for x in type_ids_pred], }
                else:
                    result_dict = {'mention_id': sample['mention_id'],
                                   'mention': sample['mention'],
                                   'labels': [x for x in sample['types']
                                              if x in gres.type2type_id_dict],
                                   'preds': [gres.type_id2type_dict[x] for x in type_ids_pred] }
                    if len(result_dict['labels']) == 0:
                        continue
                if config.only_general_types:
                    if set(result_dict['general_types']) == set(result_dict['preds']):
                        result_objs.append(result_dict)
                    else:
                        incorrect_result_objs.append(result_dict)
                else:
                    if set(result_dict['labels']) == set(result_dict['preds']):
                        result_objs.append(result_dict)
                    else:
                        incorrect_result_objs.append(result_dict)
                # 'logits': [float(v) for v in sample_logits]})

    # print(f'if all predict {gres.type2type_id_dict["人"]}: strict: {utils.strict_acc(true_labels_dict, pred_labels_dict_test)}, '
    #       f'partial: {utils.strict_acc(true_labels_dict, pred_labels_dict_test)}')
    true_labels_dict = {k:v for k, v in true_labels_dict.items() if len(v)>0}
    strict_acc = utils.strict_acc(true_labels_dict, pred_labels_dict)
    partial_acc = utils.partial_acc(true_labels_dict, pred_labels_dict)
    maf1, ma_p, ma_r = utils.macrof1(true_labels_dict, pred_labels_dict, return_pnr=True)
    mif1 = utils.microf1(true_labels_dict, pred_labels_dict)
    return sum(losses), strict_acc, partial_acc, maf1, ma_p, ma_r, mif1, result_objs, incorrect_result_objs