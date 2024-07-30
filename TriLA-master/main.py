# coding:utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import sys, traceback
import datetime
import random
import numpy as np
import torch
import argparse

from torch.utils.data import DataLoader
from train_NoAdapt import TrainNoAdapt
from train_DA import TrainDA
from test_DA import TestDA
from dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.transform import collate_fn_wo_transform, collate_fn_w_transform


torch.set_num_threads(1)


def seed_torch(rand_seed):
    os.environ['PYTHONHASHSEED'] = str(rand_seed)   # 为了禁止hash随机化，使得实验可复现
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)   # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.log = open(filename, 'w')
        self.hook = sys.excepthook
        sys.excepthook = self.kill

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def kill(self, ttype, tvalue, ttraceback):
        for trace in traceback.format_exception(ttype, tvalue, ttraceback):
            print(trace)
        os.remove(self.filename)

    def flush(self):
        pass


def print_information(config):
    print('GPUs: ' + str(torch.cuda.device_count()))
    print('time: ' + str(config.time_now))
    print('mode: ' + str(config.mode))
    print('source domain: ' + str(config.Source_Dataset))
    print('target domain: ' + str(config.Target_Dataset))
    print('model: ' + str(config.model_type) + ', reload=' + str(config.reload))

    print('loss: ' + str(config.lossmap))

    print('input size: ' + str(config.image_size))
    print('batch size: ' + str(config.batch_size))

    print('VAE_coef: ' + str(config.vae_coef))
    print('Style_coef: ' + str(config.style_coef))
    print('Content_coef: ' + str(config.content_coef))
    print('Output_coef: ' + str(config.output_coef))

    print('LFDA Upper bound ' + str(config.upbound))
    print('LFDA Lower bound ' + str(config.lowbound))

    print('optimizer: ' + str(config.optimizer))
    print('lr_scheduler: ' + str(config.lr_scheduler))
    print('lr: ' + str(config.lr))
    print('momentum: ' + str(config.momentum))
    print('weight_decay: ' + str(config.weight_decay))
    print('***' * 10)


def main(config):
    loss_name = '('
    for loss in config.lossmap:
        loss_name += loss
        if loss != config.lossmap[-1]:
            loss_name += ', '

    # loss_name += (config.Source_Dataset + '2' + config.Target_Dataset)
    loss_name += ')'
    config.time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")

    if config.reload is not None:
        config.model_path = os.path.join(config.path_save_model, loss_name, config.load_time)
        config.result_path = os.path.join(config.path_save_result, config.model_type, loss_name, config.mode,
                                          config.Target_Dataset + '-' + config.load_time)
    else:
        config.model_path = os.path.join(config.path_save_model, loss_name, config.time_now)
        config.result_path = os.path.join(config.path_save_result, config.model_type, loss_name, config.mode,
                                          config.Target_Dataset + '-' + config.time_now)
    config.log_path = os.path.join(config.path_save_log, config.mode, str(config.Source_Dataset)+'2'+config.Target_Dataset)
    config.savefig = config.model_type+loss_name

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)

    config.log_path = os.path.join(config.log_path, config.model_type+'-' + config.time_now + '.log')
    sys.stdout = Logger(config.log_path, sys.stdout)

    print_information(config)
    if 'train' in config.mode:
        source_name = config.Source_Dataset
        source_csv = []
        for s_n in source_name:
            source_csv.append(s_n + '_test.csv')
            source_csv.append(s_n + '_train.csv')
        target_train_csv = ['MESSIDOR_' + config.Target_Dataset + '_unlabeled.csv']
        target_test_csv = ['MESSIDOR_' + config.Target_Dataset + '_test.csv']

        print('Source dataset' + str(source_csv))
        print('Target train dataset' + str(target_train_csv))
        print('Target test dataset' + str(target_test_csv))
        print('Training... Domain Adaptation Phase')
        print('{} 2 {}'.format(config.Source_Dataset, config.Target_Dataset))

        sr_img_list, sr_label_list = convert_labeled_list(config.dataset_root, source_csv, r=1)
        tr_img_list, _ = convert_labeled_list(config.dataset_root, target_train_csv, r=1)
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv, r=1)
        # print(len(sr_img_list), len(tr_img_list), len(ts_img_list))

        source_dataset = RIGA_labeled_set(config.dataset_root, sr_img_list, sr_label_list,
                                          config.image_size, img_normalize=False)
        source_dataloader = DataLoader(dataset=source_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       collate_fn=collate_fn_w_transform,
                                       num_workers=1,
                                       drop_last=True)

        target_train_dataset = RIGA_unlabeled_set(config.dataset_root, tr_img_list, config.image_size)
        target_train_dataloader = DataLoader(dataset=target_train_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             collate_fn=collate_fn_wo_transform,
                                             num_workers=1,
                                             drop_last=True)

        target_valid_dataset = RIGA_labeled_set(config.dataset_root, ts_img_list, ts_label_list,
                                                config.image_size, img_normalize=True)
        test_dataloader = DataLoader(dataset=target_valid_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=1,
                                     collate_fn=collate_fn_wo_transform)

        train = TrainDA(config, source_dataloader, target_train_dataloader, test_dataloader)
        train.run()

    elif config.mode == 'NoAdapt':
        source_name = config.Source_Dataset
        train_csv = []
        for s_n in source_name:
            train_csv.append(s_n + '_test.csv')
            train_csv.append(s_n + '_train.csv')
        test_csv = ['MESSIDOR_' + config.Target_Dataset + '_test.csv']

        print('NoAdapt train dataset' + str(train_csv))
        print('NoAdapt test dataset' + str(test_csv))

        tr_img_list, tr_label_list = convert_labeled_list(config.dataset_root, train_csv, r=1)
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, test_csv, r=1)

        train_dataset = RIGA_labeled_set(config.dataset_root, tr_img_list, tr_label_list,
                                         config.image_size, img_normalize=False)
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      collate_fn=collate_fn_w_transform,
                                      num_workers=1,
                                      drop_last=True)

        test_dataset = RIGA_labeled_set(config.dataset_root, ts_img_list, ts_label_list, config.image_size)
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=1,
                                     collate_fn=collate_fn_wo_transform)

        train = TrainNoAdapt(config, train_dataloader, test_dataloader)
        train.run()

    elif config.mode == 'IntraDomain':
        train_csv = ['MESSIDOR_' + config.Target_Dataset + '_train.csv']
        test_csv = ['MESSIDOR_' + config.Target_Dataset + '_test.csv']

        print('Intra-domain train dataset' + str(train_csv))
        print('Intra-domain test dataset' + str(test_csv))

        tr_img_list, tr_label_list = convert_labeled_list(config.dataset_root, train_csv, r=1)
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, test_csv, r=1)

        train_dataset = RIGA_labeled_set(config.dataset_root, tr_img_list, tr_label_list,
                                         config.image_size, img_normalize=False)
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      collate_fn=collate_fn_w_transform,
                                      num_workers=1,
                                      drop_last=True)

        test_dataset = RIGA_labeled_set(config.dataset_root, ts_img_list, ts_label_list, config.image_size)
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=1,
                                     collate_fn=collate_fn_wo_transform)

        train = TrainNoAdapt(config, train_dataloader, test_dataloader)
        train.run()

    elif config.mode == 'test':
        target_test_csv = ['MESSIDOR_' + config.Target_Dataset + '_test.csv']
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv, r=1)

        print('Target test dataset' + str(target_test_csv))
        print('{} 2 {}'.format(config.Source_Dataset, config.Target_Dataset))
        print('Loading model: ' + str(config.load_time) + '/' + str(config.reload) + '-' + str(config.model_type) + '.pth')

        target_valid_dataset = RIGA_labeled_set(config.dataset_root, ts_img_list, ts_label_list,
                                                config.image_size, img_normalize=True)
        test_dataloader = DataLoader(dataset=target_valid_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=1,
                                     collate_fn=collate_fn_wo_transform)

        test = TestDA(config, test_dataloader)
        test.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_DA',
                        help='train_DA/NoAdapt/IntraDomain/test')
    parser.add_argument('--lossmap', type=str, default=['bce', 'dice'])

    parser.add_argument('--reload', type=int, default=100)
    parser.add_argument('--load_time', type=str)
    parser.add_argument('--model_type', type=str, default='Res_Unet',
                        help='Deeplab_Mobile/Deeplab_Res/Res_Unet')  # choose the model

    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=256)

    parser.add_argument('--optimizer', type=str, default='AdamW', help='SGD/Adam/AdamW')
    parser.add_argument('--lr_scheduler', type=str, default='Epoch',
                        help='Cosine/Step/Epoch')   # choose the decrease strategy of lr
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # weight_decay in SGD
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)  # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)  # beta2 in Adam
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--vae_coef', type=float, default=0.1)
    parser.add_argument('--style_coef', type=float, default=0.1)
    parser.add_argument('--content_coef', type=float, default=0.001)
    parser.add_argument('--output_coef', type=float, default=0.1)

    parser.add_argument('--upbound', type=int, default=4)
    parser.add_argument('--lowbound', type=int, default=1)

    parser.add_argument('--Source_Dataset', nargs='+', type=str, default=['BinRushed'], help='BinRushed/Magrabia')
    parser.add_argument('--Target_Dataset', type=str, default='Base1', help='Base1/Base2/Base3')

    parser.add_argument('--path_save_result', type=str, default='./results/')
    parser.add_argument('--path_save_model', type=str, default='./models/')
    parser.add_argument('--path_save_log', type=str, default='./logs/')
    parser.add_argument('--dataset_root', type=str, default='/media/userdisk0/zychen/Datasets/RIGAPlus_512')

    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, default='cpu')

    config = parser.parse_args()
    main(config)

