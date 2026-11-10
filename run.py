import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed 1 2 2024 2025 2026')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, default='CySAF',
                    choices=['CySAF', 'CySAF_AdaptiveFilter', 'CySAF_Seg',
                             'CySAF_feedforward', 'CySAF_module_ablation', 'CySAF_decompositon',
                            ],
                    help='model name')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/未分类/', help='location of model checkpoints')
parser.add_argument('--test_results', type=str, default='./test_results/未分类/',
                    help='location of test prediction plots and text files')
parser.add_argument('--results', type=str, default='./results/',
                    help='location of test metric artifacts')
parser.add_argument('--result_log', type=str, default='./result.txt',
                    help='append concise test metrics to this file')
parser.add_argument('--save_test_plots', type=int, default=1, choices=[0, 1],
                    help='save one PDF and prediction text files per test batch')
parser.add_argument('--test_only', action='store_true',
                    help='load the checkpoint and export full test-set arrays without training')
parser.add_argument(
    '--pretrained_checkpoint', type=str, default='',
    help='optional checkpoint used to warm-start structurally compatible CySAF parameters',
)

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# Cycle prior and normalization.
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')
parser.add_argument(
    '--use_x_period_filter', type=int, default=0, choices=[0, 1],
    help='apply the adaptive threshold filter to the input cycle template',
)
parser.add_argument(
    '--use_y_period_filter', type=int, default=1, choices=[0, 1],
    help='apply the adaptive threshold filter to the prediction cycle template',
)
parser.add_argument(
    '--use_period_decomposition', '--use_cycle_decomposition', '--use_period',
    dest='use_period_decomposition', type=int, default=1, choices=[0, 1],
    help='1: separate cycle/non-cycle components (default); 0: feed normalized x directly to Seg/MLP/Linear',
)

# CySAF model parameters
parser.add_argument('--seg_window', type=int, default=6, help='segment window')
parser.add_argument('--seg_stride', type=int, default=6, help='segment stride')
parser.add_argument('--window_type', type=str, default='rect', help='rect, hamming, hann, gaussian')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--seg_filter', type=str, default='adaptive', choices=['adaptive', 'lowpass', 'paifilter'],
                    help='spectral filter used inside Seg')
parser.add_argument('--lowpass_cutoff', type=float, default=0.1,
                    help='normalized low-pass cutoff in cycles/sample for Seg lowpass ablation')
parser.add_argument('--mask_temperature', type=float, default=16.0,
                    help='temperature for straight-through spectral mask')
parser.add_argument('--keep_bins', type=int, default=2,
                    help='number of low-frequency bins always kept in the cycle filter')
parser.add_argument('--adaptive_threshold_floor', type=float, default=1.0,
                    help='minimum median-normalized threshold for CySAF_ConstrainedAdaptiveFilter')
parser.add_argument('--adaptive_threshold_ceiling', type=float, default=4.0,
                    help='maximum median-normalized threshold for CySAF_ConstrainedAdaptiveFilter')
parser.add_argument('--adaptive_threshold_init', type=float, default=1.25,
                    help='initial median-normalized threshold for CySAF_ConstrainedAdaptiveFilter')
parser.add_argument(
    '--decomp_method', type=str, default='moving_avg',
    choices=['moving_avg', 'dft', 'ld'],
    help='cycle/non-cycle decomposition used by CySAF_decompositon',
)
parser.add_argument(
    '--decomp_kernel_size', type=int, default=25,
    help='positive odd kernel size for moving-average and LD decomposition',
)
parser.add_argument(
    '--dft_top_k', type=int, default=5,
    help='number of non-DC temporal frequency bins retained by DFT decomposition',
)
parser.add_argument(
    '--adaptive_filter_mode', type=str, default='adaptive',
    choices=['adaptive', 'fixed_energy', 'lowpass', 'linear',
             'legacy', 'energy_threshold'],
    help='cycle-template filter used by the CySAF_AdaptiveFilter ablation model',
)
parser.add_argument(
    '--adaptive_filter_energy_threshold', type=float, default=1.0,
    help='fixed median-normalized energy threshold (default: 1.0)',
)
parser.add_argument(
    '--adaptive_filter_cutoff', type=float, default=0.1,
    help='fixed low-pass cutoff in cycles/sample; valid range is [0, 0.5]',
)
parser.add_argument('--seg_variant', type=str, default='seg', choices=['seg', 'mlp', 'linear'],
                    help='prediction layer: original Seg, whole-layer MLP, or whole-layer Linear')
parser.add_argument('--Cycle_num', type=int, default=1, help='number of cycles (CySAF supports one)')
parser.add_argument('--every_Cycle', type=int,nargs='+', default=[672,96], help='for example(672 96）')
parser.add_argument('--recent_num', type=int,nargs='+', default=[672,96], help='')
parser.add_argument('--Cycle_stride', type=int,nargs='+', default=[8,1], help='')
parser.add_argument('--Cycle_window', type=int,nargs='+', default=[16,1], help='')
# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# SegRNN
parser.add_argument('--rnn_type', default='gru', help='rnn_type')
parser.add_argument('--dec_way', default='pmf', help='decode way')
parser.add_argument('--seg_len', type=int, default=48, help='segment length')
parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

# SparseTSF
parser.add_argument('--period_len', type=int, default=24, help='period_len')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')

parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training and not args.test_only:
    for ii in range(args.itr):

        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.cycle,
            fix_seed)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.cycle,
        fix_seed)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    if args.test_only:
        exp.test(setting, test=1)
    else:
        exp.predict(setting, True)
    torch.cuda.empty_cache()
