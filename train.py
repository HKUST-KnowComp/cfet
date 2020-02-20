import datetime
import torch
import numpy as np
import os
import logging
import random
from os.path import join
from time import time

import torch
from torch import nn


from utils import exp_utils, datautils, model_utils, utils
from utils.loggingutils import init_universal_logging
from utils.fet import fet_model
import config



def train_model():
    if config.dataset == 'cufe':
        data_prefix = config.CUFE_FILES['training_data_prefix']
        if config.train_on_crowd:
            data_prefix = config.CUFE_FILES['crowd_training_data_prefix']

    elif config.dataset == 'ufet':
        data_prefix = config.UFET_FILES['training_data_prefix']


    data_prefix += f'-{config.seq_tokenizer_name}'
    if config.dir_suffix:
        data_prefix += '-' + config.dir_suffix
    if not os.path.isdir(data_prefix): os.mkdir(data_prefix)

    str_today = datetime.datetime.now().strftime('%m_%d_%H%M')
    if not os.path.isdir(join(data_prefix, 'log')): os.mkdir(join(data_prefix, 'log'))
    if not config.test:
        log_file = os.path.join(join(data_prefix, 'log'), '{}-{}.log'.format(str_today, config.model_name))
        print(log_file)
    else:
        log_file = os.path.join(config.LOG_DIR, '{}-{}-test.log'.format(str_today, config.model_name))
    # if not os.path.isdir(log_file) and not config.test: os.mkdir(log_file)
    init_universal_logging(log_file, mode='a', to_stdout=True)

    save_model_dir = join(data_prefix, 'models')
    if not os.path.isdir(save_model_dir): os.mkdir(save_model_dir)

    gres = exp_utils.GlobalRes(config)

    run_name = f'{config.dataset}-{config.model_name}-{config.seq_tokenizer_name}-{str_today}'
    logging.info(f'run_name: {run_name}')
    logging.info(f'/data/cleeag/word_embeddings/{config.mention_tokenizer_name}/{config.mention_tokenizer_name}_tokenizer&vecs.pkl -- loaded')
    logging.info(f'training on {config.dataset}')
    logging.info(f'total type count: {len(gres.type2type_id_dict)}, '
                 f'general type count: {0 if config.without_general_types else len(gres.general_type_set)}')

    if config.dataset == 'ufet':
        crowd_training_samples = f'{config.CROWD_TRAIN_DATA_PREFIX}-{config.seq_tokenizer_name}.pkl'
        if config.test:

            train_data_pkl = join(data_prefix, 'dev.pkl')
            training_samples = datautils.load_pickle_data(train_data_pkl)
            crowd_training_samples = datautils.load_pickle_data(crowd_training_samples)

        else:
            train_data_pkl = join(data_prefix, 'train.pkl')
            print('loading training data {} ...'.format(train_data_pkl), end=' ', flush=True)
            training_samples = datautils.load_pickle_data(train_data_pkl)
            print('done', flush=True)
            logging.info('training data {} -- loaded'.format(train_data_pkl))

            crowd_training_samples = datautils.load_pickle_data(crowd_training_samples)

            if config.fine_tune and config.use_bert:
                # training_samples = random.choices(len(training_samples) // 10, training_samples)
                random.shuffle(training_samples)
                training_samples = training_samples[:len(training_samples) // 10]
                logging.info(f'fining tuning with {len(training_samples)} samples')

        # dev_data_pkl = join(data_prefix, 'dev.pkl')
        # dev_samples = datautils.load_pickle_data(dev_data_pkl)
        dev_json_path = join(data_prefix, 'dev.json')
        dev_samples = exp_utils.model_samples_from_json(dev_json_path)
        dev_true_labels_dict = {s['mention_id']: [gres.type2type_id_dict.get(x)
                                                  for x in s['types'] if x in gres.type2type_id_dict] for s in
                                dev_samples}

        test_data_pkl = join(data_prefix, 'test.pkl')
        test_samples = datautils.load_pickle_data(test_data_pkl)
        test_true_labels_dict = {s['mention_id']: [gres.type2type_id_dict.get(x)
                                                   for x in s['types'] if x in gres.type2type_id_dict] for s in
                                 test_samples}

    else:
        dev_data_pkl = join(data_prefix, 'dev.pkl')
        test_data_pkl = join(data_prefix, 'test.pkl')
        if config.test:
            train_data_pkl = join(data_prefix, 'dev.pkl')
        else:
            train_data_pkl = join(data_prefix, 'train.pkl')
        # test_data_pkl = config.CUFE_FILES['test_data_file_prefix'] + f'-{config.seq_tokenizer_name}/test.pkl'

        print('loading training data {} ...'.format(train_data_pkl), end=' ', flush=True)
        training_samples = datautils.load_pickle_data(train_data_pkl)
        print('done', flush=True)
        logging.info('training data {} -- loaded'.format(train_data_pkl))

        if not config.train_on_crowd:
            # crowd_training_samples = f'{config.CROWD_TRAIN_DATA_PREFIX}-{config.seq_tokenizer_name}/train.pkl'
            crowd_training_samples = datautils.load_pickle_data(join(data_prefix, 'crowd-train.pkl'))

        print('loading dev data {} ...'.format(dev_data_pkl), end=' ', flush=True)
        dev_samples = datautils.load_pickle_data(dev_data_pkl)
        print('done', flush=True)
        dev_true_labels_dict = {s['mention_id']: [gres.type2type_id_dict.get(x) for x in
                                                  exp_utils.general_mapping(s['types'], gres)] for s in dev_samples}

        test_samples = datautils.load_pickle_data(test_data_pkl)
        test_true_labels_dict = {s['mention_id']: [gres.type2type_id_dict.get(x) for x in
                                                   exp_utils.general_mapping(s['types'], gres)] for s in test_samples}

    logging.info(f'total training samples: {len(training_samples)}, '
                 f'dev samples: {len(dev_samples)}, testing samples: {len(test_samples)}')

    if not config.test:
        result_dir = join(data_prefix, f'{str_today}-results')
        if config.dataset == 'cufe':
            type_scope = 'general_types' if config.only_general_types else 'all_types'
        else:
            type_scope = config.dataset_type
        dev_results_file = join(result_dir,
                                f'dev-{config.model_name}-{type_scope}-results-{config.inference_threshhold}.txt')
        dev_incorrect_results_file = join(result_dir,
                                          f'dev-{config.model_name}-{type_scope}-incorrect_results-{config.inference_threshhold}.txt')
        test_results_file = join(result_dir,
                                 f'test-{config.model_name}-{type_scope}-results-{config.inference_threshhold}.txt')
        test_incorrect_results_file = join(result_dir,
                                           f'test-{config.model_name}-{type_scope}-incorrect_results-{config.inference_threshhold}.txt')
    else:
        result_dir = join(data_prefix, f'test-results')
        dev_results_file = join(result_dir, f'dev-results.txt')
        dev_incorrect_results_file = join(result_dir, f'dev-incorrect_results.txt')
        test_results_file = join(result_dir, f'test-results.txt')
        test_incorrect_results_file = join(result_dir, f'test-incorrect_results.txt')

    if not os.path.isdir(result_dir): os.mkdir(result_dir)

    logging.info('use_bert = {}, use_lstm = {}, use_mlp={}, bert_param_frozen={}, bert_fine_tune={}'
                 .format(config.use_bert, config.use_lstm, config.use_mlp, config.freeze_bert, config.fine_tune))
    logging.info(
        'type_embed_dim={} contextt_lstm_hidden_dim={} pmlp_hdim={}'.format(
            config.type_embed_dim, config.lstm_hidden_dim, config.pred_mlp_hdim))

    # setup training
    device = torch.device(f'cuda:{config.gpu_ids[0]}') if torch.cuda.device_count() > 0 else None
    device_name = torch.cuda.get_device_name(config.gpu_ids[0])

    logging.info(f'running on device: {device_name}')
    logging.info('building model...')

    model = fet_model(config, device, gres)
    logging.info(f'transfer={config.transfer}')

    if config.continue_train:
        model_path = config.CONTINUE_TRAINING_PATH[config.continue_train]
        logging.info(f'loading checkpoint from {model_path}')
        trained_weights = torch.load(model_path)
        trained_weights = {'.'.join(k.split('.')[1:]): v for k, v in trained_weights.items()}
        cur_model_dict = model.state_dict()
        cur_model_dict.update(trained_weights)
        model.load_state_dict(cur_model_dict)

    if config.transfer and config.use_lstm:
        logging.info(f'loading checkpoint from {config.TRANSFER_MODEL_PATH}')
        cur_model_dict = model.state_dict()
        trained_weights = torch.load(config.TRANSFER_MODEL_PATH)
        trained_weights_bilstm = {'.'.join(k.split('.')[1:]): v for k, v in trained_weights.items() if 'bi_lstm' in k}
        cur_model_dict.update(trained_weights_bilstm)
        model.load_state_dict(cur_model_dict)



    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    batch_size = 32 if config.dataset == 'cufe' and config.train_on_crowd else config.batch_size
    n_iter = 150 if config.dataset == 'cufe' and config.train_on_crowd else config.n_iter
    n_batches = (len(training_samples) + batch_size - 1) // batch_size
    n_steps = n_iter * n_batches
    eval_cycle = config.eval_cycle

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses = list()
    best_dev_acc = -1
    best_maf1_v = -1
    step = 0
    steps_since_last_best = 0

    # start training
    logging.info('{}'.format(model.__class__.__name__))
    logging.info('training batch size: {}'.format(batch_size))
    logging.info('{} epochs, {} steps, {} steps per iter, learning rate={}, lr_decay={}, start training ...'.format(
        n_iter, n_steps, n_batches, config.learning_rate, config.lr_gamma))

    while True:
        if step == n_steps:
            break

        batch_idx = step % n_batches
        batch_beg, batch_end = batch_idx * batch_size, min((batch_idx + 1) * batch_size,
                                                                  len(training_samples))
        if config.dataset == 'ufet':
            batch_samples = training_samples[batch_beg:batch_end - batch_size * 2 // 3] \
                            + random.choices(crowd_training_samples, k=batch_size * 1 // 3)
        elif config.dataset == 'cufe':
            if not config.train_on_crowd:
                batch_samples = training_samples[batch_beg:batch_end - batch_size * 2 // 3] \
                                + random.choices(crowd_training_samples, k=batch_size * 1 // 3)
            else:
                batch_samples = training_samples[batch_beg:batch_end]

        try:
            input_dataset, type_vecs = exp_utils.samples_to_tensor(config, gres, batch_samples)

            input_dataset = tuple(x.to(device) for x in input_dataset)
            type_vecs = type_vecs.to(device)
            model.module.train()
            logits = model(input_dataset, gres)
        except:
            step += 1
            continue

        if config.dataset == 'ufet':
            loss = model.module.define_loss(logits, type_vecs, config.dataset_type)
        elif config.GENERAL_TYPES_MAPPING and not config.only_general_types:
            loss = model.module.get_uw_loss(logits, type_vecs, gres)
        else:
            loss = model.module.get_loss(logits, type_vecs)
        optimizer.zero_grad()

        loss.backward()
        if config.use_lstm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, float('inf'))
        optimizer.step()
        losses.append(loss.data.cpu().numpy())
        step += 1
        if step % eval_cycle == 0 and step > 0:
            print('\nevaluating...')
            l_v, acc_v, pacc_v, maf1_v, ma_p_v, ma_r_v, mif1_v, dev_results, incorrect_dev_result_objs = \
                model_utils.eval_fetel(config, gres, model, dev_samples, dev_true_labels_dict)

            l_t, acc_t, pacc_t, maf1_t, ma_p_t, ma_r_t, mif1_t, test_results, incorrect_test_result_objs = \
                model_utils.eval_fetel(config, gres, model, test_samples, test_true_labels_dict)

            if maf1_v > best_maf1_v:
                steps_since_last_best = 0
            best_tag = '*' if maf1_v > best_maf1_v else ''
            logging.info('run name={}, step={}/{}, learning rate={}, losses={:.4f}, steps_since_last_best={}'
                         .format(run_name, step, n_steps,
                                 optimizer.param_groups[0]['lr'], sum(losses), steps_since_last_best))
            logging.info('dev evaluation result: '
                         'l_v={:.4f} acc_v={:.4f} pacc_v={:.4f} macro_f1_v={:.4f} micro_f1_v={:.4f}{}'
                         .format(l_v, acc_v, pacc_v, maf1_v, mif1_v, best_tag))
            logging.info(f'dev evaluation result: macro_p={ma_p_v:.4f}, macro_r={ma_r_v:.4f}')

            logging.info('test evaluation result: '
                         'l_v={:.4f} acc_t={:.4f} pacc_t={:.4f} macro_f1_t={:.4f} micro_f1_t={:.4f}{}'
                         .format(l_t, acc_t, pacc_t, maf1_t, mif1_t, best_tag))
            logging.info(f'test evaluation result: macro_p={ma_p_t:.4f}, macro_r={ma_r_t:.4f}')

            if maf1_v > best_maf1_v:
                if save_model_dir and not config.test:
                    save_model_file = join(save_model_dir, f'{config.model_name}-{str_today}.pt')
                    torch.save(model.state_dict(), save_model_file)
                    logging.info('model saved to {}'.format(save_model_file))

                logging.info('prediction result saved to {}'.format(result_dir))
                datautils.save_json_objs(dev_results, dev_results_file)
                datautils.save_json_objs(incorrect_dev_result_objs, dev_incorrect_results_file)
                datautils.save_json_objs(test_results, test_results_file)
                datautils.save_json_objs(incorrect_test_result_objs, test_incorrect_results_file)
                # best_dev_acc = acc_v
                best_maf1_v = maf1_v

            losses = list()

        steps_since_last_best += 1


if __name__ == '__main__':
    torch.random.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.NP_RANDOM_SEED)
    random.seed(config.PY_RANDOM_SEED)

    str_today = datetime.datetime.now().strftime('%m_%d_%H%M')
    model_used = 'use_bert' if config.use_bert else 'use_lstm'
    if not os.path.isdir(config.LOG_DIR): os.mkdir(config.LOG_DIR)
    if not config.test:
        log_file = os.path.join(config.LOG_DIR, '{}-{}_{}.log'.format(os.path.splitext(
            os.path.basename(__file__))[0], str_today, model_used))
    else:
        log_file = os.path.join(config.LOG_DIR, '{}-{}_{}_test.log'.format(os.path.splitext(
            os.path.basename(__file__))[0], str_today, model_used))
    init_universal_logging(log_file, mode='a', to_stdout=True)
    train_model()
    # model_utils.check_breakdown_performance()
