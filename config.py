from platform import platform
from os.path import join
import socket

DATA_DIR = '/data/cleeag/distant_supervision_data'
UFET_DATA_DIR = '/data/cleeag/ufet_data'

TOKEN_UNK = '<UNK>'
TOKEN_ZERO_PAD = '<ZPAD>'
TOKEN_EMPTY_PAD = '<EPAD>'
TOKEN_MENTION = '[MASK]'

RANDOM_SEED = 771
NP_RANDOM_SEED = 7711
PY_RANDOM_SEED = 9973

MACHINE_NAME = socket.gethostname()
LOG_DIR = join(DATA_DIR, 'log')


CUFE_FILES = {
    'training_data_prefix': join(DATA_DIR, 'training_data', 'train-wiki_data_zh-cufe_type2general_zh'),
    'crowd_training_data_prefix':join(DATA_DIR, 'training_data', 'crowd_data'),
    'test_data_file_prefix':'/data/cleeag/distant_supervision_data/training_data/crowd_data'
}

UFET_FILES = {
    'training_data_prefix': join(UFET_DATA_DIR, 'training_data', 'ufet-mixed'),
    'ufet_training_type_set': join(UFET_DATA_DIR, 'type_set', 'types.txt'),
}

dir_suffix = None

GENERAL_TYPES_MAPPING = f'/data/cleeag/distant_supervision_data/type_sets/cufe_type2general_zh.pkl'

MODEL_CACHE_PATH =  '/data/cleeag/transformers_model_cache'


"""------------------------- training configurations -------------------------"""
max_seq_length = 128
# eval_batch_size = 50
dropout = 0.5

# dimensions
lstm_hidden_dim = 150
type_embed_dim = 500
pred_mlp_hdim = 400
mlp_hidden_dim = 400
bert_hdim = 768

# training parameters
n_iter = 30
bert_adam_warmup = 0.001
lr_gamma = 0.7
nil_rate = 0.5
rand_per = True
per_penalty = 2.0
inference_threshhold = 0

# model structure
bert_use_four = 0
concat_lstm = False
use_mlp = 0

use_bert = 1
use_lstm = 0

# dataset = 'ufet'
dataset = 'cufe'

batch_size = 32 if use_bert else 256

if dataset == 'ufet':
    # dataset_type = 'kb'
    # dataset_type = 'wiki'
    dataset_type = 'open'
    # dataset_type = 'gen'
    # dataset_type = 'onto'
    model_name = 'bert-base-cased' if use_bert else 'bi-lstm'
    WORD_VECS_FILE = '/data/cleeag/word_embeddings/glove/glove_tokenizer&vecs.pkl'
    CROWD_TRAIN_DATA_PREFIX = '/data/cleeag/ufet_data/training_data/raw/ufet-crowd_train'

elif dataset == 'cufe':
    model_name = 'bert-base-chinese' if use_bert else 'bi-lstm'
    WORD_VECS_FILE = '/data/cleeag/word_embeddings/fasttext/fasttext_tokenizer&vecs-zh.pkl'
    CROWD_TRAIN_DATA_PREFIX = '/data/cleeag/distant_supervision_data/training_data/crowd_data'


only_general_types = 0
without_general_types = 0
ANSWER_NUM_DICT = {"open": 10331, "onto": 89, "wiki": 4600, "kb": 130, "gen": 9}

if use_bert:
    fine_tune = 0
    freeze_bert = 0
else:
    fine_tune = 0
    freeze_bert = 0

use_gpu = 1
# gpu_ids = [0, 1, 2, 3]
# gpu_ids = [0, 1]
if use_bert:
    gpu_ids = [2]
    # gpu_ids = [1]
    # gpu_ids = [3]
    # gpu_ids = [1, 2]
elif use_lstm:
    # gpu_ids = [1]
    # gpu_ids = [2, 3]
    gpu_ids = [2]
    # gpu_ids = [0, 1, 2, 3]

if use_bert:
    seq_tokenizer_name = 'bert'
    if dataset == 'ufet':
        mention_tokenizer_name = 'glove'
    else:
        mention_tokenizer_name = 'fasttext'
elif use_lstm:
    if dataset == 'ufet':
        seq_tokenizer_name = 'glove'
        mention_tokenizer_name = 'glove'
    else:
        seq_tokenizer_name = 'fasttext'
        mention_tokenizer_name = 'fasttext'
# seq_tokenizer_name = 'babylon'
# mention_tokenizer_name = 'babylon'
transfer = 0
train_on_crowd = 0

continue_train = 0
# continue_train = 'ufet_lstm'
CONTINUE_TRAINING_PATH = {
    'lstm_no_gen':'',
    'lstm_with_gen': '/data/cleeag/distant_supervision_data/training_data/train-wiki_data_zh-cufe_type2general_zh-fasttext/models/bi-lstm-11_21_0400.pt',
    'lstm_with_gen_babylon': '/data/cleeag/distant_supervision_data/training_data/train-wiki_data_zh-cufe_type2general_zh-fasttext/models/bi-lstm-11_21_0400.pt',
    'bert_no_gen': '',
    'bert_with_gen': '/data/cleeag/distant_supervision_data/training_data/train-wiki_data_zh-cufe_type2general_zh-bert/models/bert-base-chinese-11_21_0356.pt',
    'ufet_lstm':'/data/cleeag/ufet_data/training_data/ufet-mixed-glove/models/bi-lstm-11_17_2156.pt',
    # 'ufet_bert': '/data/cleeag/ufet_data/training_data/ufet-mixed-bert/models/bert-base-cased-11_20_1749.pt'

}

TRANSFER_MODEL_PATH = '/data/cleeag/ufet_data/training_data/ufet-mixed-babylon/models/bi-lstm-11_16_0032.pt'

if use_bert:
    # learning_rate = 3e-6 if train_on_crowd  else 3e-5
    learning_rate = 3e-5
else:
    learning_rate = 0.001

if dataset == 'ufet':
    # eval_cycle = 300 if not use_bert else 1000
    eval_cycle = 100
elif dataset == 'cufe' and train_on_crowd:
    eval_cycle = 5
else :
    eval_cycle = 100

test = 0
if test:
    n_iter = 100000
    lr_gamma = 1