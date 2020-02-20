import numpy as np
from tqdm import tqdm

def get_parent_type(t):
    p = t.rfind('/')
    if p <= 1:
        return None
    return t[:p]


def get_parent_types(t):
    parents = list()
    while True:
        p = get_parent_type(t)
        if p is None:
            return parents
        parents.append(p)
        t = p


def get_parent_type_ids_dict(type_id_dict):
    d = dict()
    for t, type_id in type_id_dict.items():
        d[type_id] = [type_id_dict[p] for p in get_parent_types(t)]
    return d


def get_full_type_ids(type_ids, parent_type_ids_dict):
    full_type_ids = set()
    for type_id in type_ids:
        full_type_ids.add(type_id)
        for tid in parent_type_ids_dict[type_id]:
            full_type_ids.add(tid)
    return list(full_type_ids)


def json_objs_to_kvlistdict(objs, key_str):
    d = dict()
    for x in objs:
        cur_key = x[key_str]
        cur_key_objs = d.get(cur_key, list())
        if not cur_key_objs:
            d[cur_key] = cur_key_objs
        cur_key_objs.append(x)
    return d


def onehot_encode(type_ids, n_types):
    tmp = np.zeros(n_types)
    for t in type_ids:
        tmp[t] = 1.0
    return tmp


def __super_types(t):
    types = [t]
    tmpt = t
    while True:
        pos = tmpt.rfind('/')
        if pos == 0:
            break
        tmpt = tmpt[:pos]
        types.append(tmpt)
    return types


def get_full_types(labels):
    types = set()
    for l in labels:
        super_types = __super_types(l)
        for t in super_types:
            types.add(t)
    return list(types)


def count_match(label_true, label_pred):
    cnt = 0
    for l in label_true:
        if l in label_pred:
            cnt += 1
    return cnt


def microf1(true_labels_dict, pred_labels_dict):
    # assert len(true_labels_dict) == len(pred_labels_dict)
    l_true_cnt, l_pred_cnt, hit_cnt = 0, 0, 0
    for mention_id, labels_true in true_labels_dict.items():
        labels_pred = pred_labels_dict[mention_id]
        hit_cnt += count_match(labels_true, labels_pred)
        l_true_cnt += len(labels_true)
        l_pred_cnt += len(labels_pred)
    p = hit_cnt / l_pred_cnt
    r = hit_cnt / l_true_cnt
    return 2 * p * r / (p + r + 1e-7)


def macrof1(true_labels_dict, pred_labels_dict, return_pnr=False):
    # assert len(true_labels_dict) == len(pred_labels_dict)
    p_acc, r_acc = 0, 0
    for mention_id, labels_true in true_labels_dict.items():
        labels_pred = pred_labels_dict[mention_id]
        if len(labels_true) == 0 or len(labels_pred) == 0:
            continue
        match_cnt = count_match(labels_true, labels_pred)
        p_acc += match_cnt / len(labels_pred)
        r_acc += match_cnt / len(labels_true)
    p, r = p_acc / len(pred_labels_dict), r_acc / len(true_labels_dict)
    f1 = 2 * p * r / (p + r + 1e-7)
    if return_pnr:
        return f1, p, r
    else:
        return f1


def strict_acc(true_labels_dict, pred_labels_dict):
    hit_cnt = 0
    for wid, labels_true in true_labels_dict.items():
        labels_pred = pred_labels_dict[wid]
        if labels_full_match(labels_true, labels_pred):
            hit_cnt += 1
    return hit_cnt / len(true_labels_dict)


def partial_acc(true_labels_dict, pred_labels_dict):
    hit_cnt = 0
    for wid, labels_true in true_labels_dict.items():
        labels_pred = pred_labels_dict[wid]
        for l in labels_pred:
            if l in labels_true:
                hit_cnt += 1
                break
    return hit_cnt / len(true_labels_dict)


def strict_acc_with_probs(true_labels_dict, result_objs):
    assert len(true_labels_dict) == len(result_objs)
    hit_cnt = 0
    for r in result_objs:
        probs = r['probs']
        y_pred = np.argmax(np.array(probs))
        labels_true = true_labels_dict[r['mention_id']]
        if labels_full_match(labels_true, [y_pred]):
            hit_cnt += 1
    return hit_cnt / len(result_objs)


def labels_full_match(labels_true, labels_pred):
    if len(labels_true) != len(labels_pred):
        return False

    for l in labels_true:
        if l not in labels_pred:
            return False
    return True


def mrr(dist_list, gold):
    """
    dist_list: list of list of label probability for all labels.
    gold: list of gold indexes.

    Get mean reciprocal rank. (this is slow, as have to sort for 10K vocab)
    """
    mrr_per_example = []
    dist_arrays = np.array(dist_list)
    dist_sorted = np.argsort(-dist_arrays, axis=1)
    for ind, gold_i in tqdm(enumerate(gold)):
        gold_i_where = [i for i in range(len(gold_i)) if gold_i[i] == 1]
        rr_per_array = []
        sorted_index = dist_sorted[ind, :]
        for gold_i_where_i in gold_i_where:
            for k in range(len(sorted_index)):
                if sorted_index[k] == gold_i_where_i:
                    rr_per_array.append(1.0 / (k + 1))
        if np.isnan(np.mean(rr_per_array)): continue
        mrr_per_example.append(np.mean(rr_per_array))
    return sum(mrr_per_example) * 1.0 / len(mrr_per_example)
