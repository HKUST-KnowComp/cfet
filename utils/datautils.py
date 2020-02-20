import pickle
import json
import pandas as pd
import numpy as np
np.random.seed(42)


def parse_typed_mention_file_line(line):
    parts = line.strip().split('\t')
    wid = int(parts[0])
    mention_str = parts[1]
    sent_id = int(parts[2])
    pos_beg, pos_end = int(parts[3]), int(parts[4])
    target_wid = int(parts[5])
    type_ids = [int(t) for t in parts[6].split(' ')]
    return wid, mention_str, sent_id, pos_beg, pos_end, target_wid, type_ids


def load_wid_types_file(filename, type_to_id_dict=None):
    wid_types_dict = dict()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            types = x['types']
            if type_to_id_dict:
                types = [type_to_id_dict[l] for l in types]
            wid_types_dict[x['wid']] = types
    return wid_types_dict


def load_pickle_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_pickle_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_type_vocab(type_vocab_file):
    type_vocab = list()
    type_to_id_dict = dict()
    with open(type_vocab_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            t = line.strip()
            type_vocab.append(t)
            type_to_id_dict[t] = i
    return type_vocab, type_to_id_dict


def read_sents_to_token_id_seq_dict(sents_file, token2token_id_dict, unknown_token_id):
    import json

    sent_tokens_dict = dict()
    f = open(sents_file, encoding='utf-8')
    for line in f:
        sent = json.loads(line)
        tokens = sent['text'].split(' ')
        sent_tokens_dict[sent['sent_id']] = [token2token_id_dict.get(t, unknown_token_id) for t in tokens]
    f.close()
    return sent_tokens_dict


def read_json_objs(filename):
    objs = list()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            objs.append(json.loads(line))
    return objs


def save_json_objs(objs, output_file):
    with open(output_file, 'w', encoding='utf-8', newline='\n') as fout:
        for v in objs:
            fout.write('{}\n'.format(json.dumps(v, ensure_ascii=False)))


def read_pred_results_file(filename, type_vocab=None):
    results = read_json_objs(filename)
    results_dict = dict()
    for r in results:
        labels = r['labels'] if type_vocab is None else [type_vocab[l] for l in r['labels']]
        results_dict[r['mention_id']] = labels
    return results_dict


def load_csv(file, na_filter=True):
    with open(file, encoding='utf-8') as f:
        return pd.read_csv(f, na_filter=na_filter)


def load_word_vec_data(file):
    with open(file, 'r') as r:
        token_vocab, token_vecs = [], []
        for i, line in enumerate(r):
            tokens = line.rstrip().split(' ')
            if len(tokens) == 2:
                n, dim = tokens
                continue
            vecs = np.array(tokens[1:], dtype='f')
            assert len(vecs) == 300
            token_vocab.append(tokens[0])
            token_vecs.append(vecs)

    token_vocab.extend(['<UNK>', '<ZPAD>', '<MEN>'])
    token_vecs.extend([np.random.rand(int(dim)) for _ in range(3)])
    token_vecs = np.array(token_vecs, dtype='f')
    tokenizer = {t: i for i, t in enumerate(token_vocab)}
    return tokenizer, np.array(token_vecs, dtype='f')
