from multiprocessing.util import log_to_stderr

import numpy as np
import argparse, math, sys, os
import torch
from torch import nn
import distributed as dist
from transformers import DistilBertForMaskedLM, DistilBertConfig
from matplotlib.lines import Line2D

from archive.mask_reconstruct_img_batched import denormalize
from vqvae import FlatVQVAE
import neptune
from pathlib import Path
import torch
import socket
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim

import random
import os
import numpy as np
import torch
import refactored.util as util
import wandb

RUNS_DIR = Path('/Users/rathjjgf/repos/mask_image_reconstruct_image/data/runs/')

FIG_DIR = '/Users/rathjjgf/Desktop/zahra_fig/'


def load_pkl(file_name):
    with open(util.DATA_DIR / 'plots' / (file_name + '.pkl'), "rb") as f:
        recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without = pickle.load(
            f)
    return recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without


def plot_x(data, axs, num_images, img_id, reverse=False):
    recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without = data
    if reverse:
        recons.reverse(), masks.reverse(), perc.reverse(), confidences.reverse(), normed_confidences.reverse(), correctly_classified.reverse(), recon_without.reverse(), correctly_classified_without.reverse()
    for j in range(num_images):
        for k in range(len(masks)):
            # axs[k][].set_ylabel(perc[k])

            axs[3 * j + 2][k].imshow(masks[k].reshape(-1, 20, 20, 1)[img_id], vmin=0, vmax=1)
            util.set_border(axs[j * 3 + 2][k], correctly_classified[k][img_id], no_color=True)

            axs[j * 3 + 1][k].imshow(util.denormalize(recons[k])[img_id].permute(1, 2, 0))
            util.set_border(axs[j * 3 + 1][k], correctly_classified[k][img_id])

            axs[j * 3 + 0][k].imshow(util.denormalize(recon_without[k])[img_id].permute(1, 2, 0))
            util.set_border(axs[j * 3 + 0][k], correctly_classified_without[k][img_id])


def plot_x_2(data, axs, num_images, img_id, reverse=False):
    recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without = data
    if reverse:
        recons.reverse(), masks.reverse(), perc.reverse(), confidences.reverse(), normed_confidences.reverse(), correctly_classified.reverse(), recon_without.reverse(), correctly_classified_without.reverse()
    for j in range(num_images):
        for k in range(0, len(masks) - 1, 2):
            axs[j * 2 + 0][k // 2].imshow(util.denormalize(recons[k])[img_id].permute(1, 2, 0))
            util.set_border(axs[j * 2 + 0][k // 2], correctly_classified[k][img_id])

            axs[2 * j + 1][k // 2].imshow(masks[k].reshape(-1, 20, 20, 1)[img_id], vmin=0, vmax=1)
            util.set_border(axs[j * 2 + 1][k // 2], correctly_classified[k][img_id])


# def get_series(key_prefix, key_suffix, run_id, update=False, load=True):
#     path = util.RUNS_DIR / run_id / (key_suffix + '.npy')
#     if update or not path.exists():
#         run = neptune.init_run(with_id=run_id, mode='read-only', project='Vqvae-transformer')
#         values = np.array(list(run[key_prefix + '/' + key_suffix].fetch_values()['value']))
#         np.save(str(path), values)
#         run.stop()
#     if load:
#         return np.load(path)
#     return None


def make_local_run_path(run_id, sub_dir=None):
    local_run_path = RUNS_DIR / run_id
    local_run_path.mkdir(exist_ok=True)
    if sub_dir is not None:
        local_run_path = RUNS_DIR / run_id / sub_dir
        local_run_path.mkdir(exist_ok=True)
    return local_run_path


def get_run_by_name(run_id, entity='ini-tns', project='Vqvae-transformer'):
    api = wandb.Api()
    return api.runs(f'{entity}/{project}', filters={"display_name": run_id})[0]


def get_series(run_id, key_suffix, key_prefix='val', load=True, update=False):
    make_local_run_path(run_id)
    path = util.RUNS_DIR / run_id / (key_suffix + '.npy')
    if not path.exists():
        run = get_run_by_name(run_id)
        values = np.array(run.history(keys=[key_prefix + '_' + key_suffix]).iloc[:, 1].tolist())
        np.save(str(path), values)
    if load:
        return np.load(path)
    return None


def plot_token_val():
    n_rows, n_cols = 2, 2
    fontsize = 12
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 1.5 * n_rows))
    # run_ids = [('VQVAET-501', 'selective'), ('VQVAET-511', 'selective'), ('VQVAET-516', 'selective'),
    #            ('VQVAET-515', 'selective'), ('VQVAET-523', 'selective'), ('VQVAET-524', 'selective'),
    #            ('VQVAET-525', 'selective'), ('VQVAET-526', 'selective'),
    #            ('VQVAET-518', 'rnd sel'), ('VQVAET-517', 'rnd sel'),
    #            ('VQVAET-521', 'rnd sel'),  ('VQVAET-522', 'rnd sel'),
    #            ('VQVAET-502', 'random')]
    # 511
    run_ids = [('VQVAET-502', 'random'), ('VQVAET-511', 'selective'),
               # ('VQVAET-518', 'rnd sel'), ('VQVAET-517', 'rnd sel'),
               # ('VQVAET-521', 'rnd sel'), ('VQVAET-522', 'rnd sel'),
               ]

    update = False
    plot_up_to = 201
    for run_id in run_ids:
        if run_id[1][:2] == 'se':
            linestyle = '--'
            color = '#ff7f0e'
        else:
            linestyle = '-'
            color = '#2ca02c'
        ser = get_series(key_prefix='val', key_suffix='rnd_acc_masked_only', run_id=run_id[0], load=True, update=update)
        ax[0][0].plot(1 - ser[:plot_up_to], label=run_id[1], linestyle=linestyle, color=color)
        ax[0][0].grid(True)
        # ax[0][0].set_title('rnd val codebook accuracy')

    for run_id in run_ids:
        if run_id[1][:2] == 'se':
            linestyle = '--'
            color = '#ff7f0e'
        else:
            linestyle = '-'
            color = '#2ca02c'
        ser = get_series(key_prefix='val', key_suffix='rnd_class_acc', run_id=run_id[0], load=True, update=update)
        ax[1][0].plot(1 - ser[:plot_up_to], label=run_id[1], linestyle=linestyle, color=color)
        ax[1][0].grid(True)

        # ax[1][0].set_title('rnd val class accuracy')

    for run_id in run_ids:
        if run_id[1][:2] == 'se':
            linestyle = '--'
            color = '#ff7f0e'
        else:
            linestyle = '-'
            color = '#2ca02c'
        ser = get_series(key_prefix='val', key_suffix='selective_desc_acc_masked_only', run_id=run_id[0], load=True,
                         update=update)
        ax[0][1].plot(1 - ser[:plot_up_to], label=run_id[1], linestyle=linestyle, color=color)
        ax[0][1].grid(True)

        # ax[0][1].set_title('sel codebook acc')

    for run_id in run_ids:
        if run_id[1][:2] == 'se':
            linestyle = '--'
            color = '#ff7f0e'
        else:
            linestyle = '-'
            color = '#2ca02c'
        ser = get_series(key_prefix='val', key_suffix='selective_desc_class_acc', run_id=run_id[0], load=True,
                         update=update)
        ax[1][1].plot(1 - ser[:plot_up_to], label=run_id[1], linestyle=linestyle, color=color)
        ax[1][1].grid(True)

        # ax[1][1].set_title('sel class acc')

    ax[0][0].set_ylim(0.75, 1)
    ax[0][0].set_yticks([0.8, 0.85, 0.9, 0.95])
    # ax[0][1].tick_params(axis='y', left=False, labelleft=False)
    ax[0][0].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[0][0].tick_params(axis='both', which='both', labelsize=fontsize)

    ax[0][1].set_ylim(0.75, 1)
    ax[0][1].set_yticks([0.8, 0.85, 0.9, 0.95])
    ax[0][0].set_yticklabels(['.80', '.85', '.90', '.95'])

    ax[0][1].tick_params(axis='y', left=False, labelleft=False)
    ax[0][1].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[0][1].tick_params(axis='both', which='both', labelsize=fontsize)

    ax[1][0].set_ylim(0.25, 1)
    ax[1][0].set_yticks([0.4, 0.55, 0.7, 0.85])

    # ax[0][1].tick_params(axis='y', left=False, labelleft=False)
    # ax[0][1].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[1][0].tick_params(axis='both', which='both', labelsize=fontsize)
    ax[1][0].set_yticklabels(['.40', '.55', '.70', '.85'])

    ax[1][1].set_ylim(0.25, 1)
    ax[1][1].set_yticks([0.4, 0.55, 0.7, 0.85])

    ax[1][1].tick_params(axis='y', left=False, labelleft=False)
    # ax[1][1].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[1][1].tick_params(axis='both', which='both', labelsize=fontsize)

    ax[1][0].set_xlabel('# Epochs', fontsize=fontsize)
    ax[1][1].set_xlabel('# Epochs', fontsize=fontsize)

    ax[0][0].set_ylabel('error rate', fontsize=fontsize)
    ax[1][0].set_ylabel('error rate', fontsize=fontsize)

    ax[1][1].legend(fontsize=fontsize)
    plt.subplots_adjust(wspace=0.015, hspace=0.03)

    ax[0, 0].annotate('validated with\nrandom masking', xy=(0.5, 1.2), xycoords='axes fraction',
                      fontsize=fontsize, ha='center', va='center', fontweight='bold')
    ax[0, 1].annotate('validated with\nselective masking', xy=(0.5, 1.2), xycoords='axes fraction',
                      fontsize=fontsize, ha='center', va='center', fontweight='bold')
    ax[0, 0].annotate('compl.', xy=(-0.32, 0.5), xycoords='axes fraction',
                      fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    ax[1, 0].annotate('recog.', xy=(-0.31, 0.5), xycoords='axes fraction',
                      fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    #
    # for run_id in run_ids:
    #     ser = get_series(key_prefix='val', key_suffix='selective_desc_acc_masked_only', run_id=run_id[0], load=True)
    #     ax[1][0].plot(ser, label=run_id[1])
    # for run_id in run_ids:
    #     ser = get_series(key_prefix='val', key_suffix='selective_desc_class_acc', run_id=run_id[0], load=True)
    #     ax[1][1].plot(ser, label=run_id[1])
    fig.savefig(FIG_DIR + 'val_error.pdf', format='pdf', bbox_inches="tight")
    plt.show()


def stabilizing_plot():
    n_rows, n_cols = 1, 2
    fontsize = 12
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 2 * n_rows))
    # run_ids = [('VQVAET-501', 'selective'), ('VQVAET-511', 'selective'), ('VQVAET-516', 'selective'),
    #            ('VQVAET-515', 'selective'), ('VQVAET-523', 'selective'), ('VQVAET-524', 'selective'),
    #            ('VQVAET-525', 'selective'), ('VQVAET-526', 'selective'),
    #            ('VQVAET-518', 'rnd sel'), ('VQVAET-517', 'rnd sel'),
    #            ('VQVAET-521', 'rnd sel'),  ('VQVAET-522', 'rnd sel'),
    #            ('VQVAET-502', 'random')]

    run_ids = [('VQVAET-511', 'selective'),
               # ('VQVAET-518', 'rnd sel'), ('VQVAET-517', 'rnd sel'),
               # ('VQVAET-521', 'rnd sel'), ('VQVAET-522', 'rnd sel'),
               ('VQVAET-502', 'random')]

    update = False
    for run_id in run_ids:
        if run_id[1][:2] == 'se':
            linestyle = '--'
        else:
            linestyle = '-'
        ser = get_series(key_prefix='val', key_suffix='rnd_acc_masked_only', run_id=run_id[0], load=True, update=update)
        ax[0][0].plot(1 - ser, label=run_id[1], linestyle=linestyle)
        ax[0][0].grid(True)
        # ax[0][0].set_title('rnd val codebook accuracy')

    for run_id in run_ids:
        if run_id[1][:2] == 'se':
            linestyle = '--'
        else:
            linestyle = '-'
        ser = get_series(key_prefix='val', key_suffix='rnd_class_acc', run_id=run_id[0], load=True, update=update)
        ax[1][0].plot(1 - ser, label=run_id[1], linestyle=linestyle)
        ax[1][0].grid(True)

        # ax[1][0].set_title('rnd val class accuracy')

    for run_id in run_ids:
        if run_id[1][:2] == 'se':
            linestyle = '--'
        else:
            linestyle = '-'
        ser = get_series(key_prefix='val', key_suffix='selective_desc_acc_masked_only', run_id=run_id[0], load=True,
                         update=update)
        ax[0][1].plot(1 - ser, label=run_id[1], linestyle=linestyle)
        ax[0][1].grid(True)

        # ax[0][1].set_title('sel codebook acc')

    for run_id in run_ids:
        if run_id[1][:2] == 'se':
            linestyle = '--'
        else:
            linestyle = '-'
        ser = get_series(key_prefix='val', key_suffix='selective_desc_class_acc', run_id=run_id[0], load=True,
                         update=False)
        ax[1][1].plot(1 - ser, label=run_id[1], linestyle=linestyle)
        ax[1][1].grid(True)

        # ax[1][1].set_title('sel class acc')

    ax[0, 0].annotate('random masking', xy=(0.5, 1.1), xycoords='axes fraction',
                      fontsize=fontsize, ha='center', va='center', fontweight='bold')
    ax[0, 1].annotate('selective masking', xy=(0.5, 1.1), xycoords='axes fraction',
                      fontsize=fontsize, ha='center', va='center', fontweight='bold')
    ax[0, 0].annotate('completion', xy=(-0.32, 0.5), xycoords='axes fraction',
                      fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    ax[1, 0].annotate('recognition', xy=(-0.32, 0.5), xycoords='axes fraction',
                      fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)

    ax[0][0].set_ylim(0.7, 1)
    ax[0][1].set_ylim(0.7, 1)

    ax[1][0].set_ylim(0.2, 1)
    ax[1][1].set_ylim(0.2, 1)

    ax[1][0].set_xlabel('# Epochs')
    ax[1][1].set_xlabel('# Epochs')

    ax[0][0].set_ylabel('error rate')
    ax[1][0].set_ylabel('error rate')

    ax[1][1].legend()

    #
    # for run_id in run_ids:
    #     ser = get_series(key_prefix='val', key_suffix='selective_desc_acc_masked_only', run_id=run_id[0], load=True)
    #     ax[1][0].plot(ser, label=run_id[1])
    # for run_id in run_ids:
    #     ser = get_series(key_prefix='val', key_suffix='selective_desc_class_acc', run_id=run_id[0], load=True)
    #     ax[1][1].plot(ser, label=run_id[1])
    fig.savefig(FIG_DIR + 'val_error.pdf', format='pdf', bbox_inches="tight")
    plt.show()


# def plot_masking_ratios(rnd_id=None, sel_id=None, rnd_sel_id=None):
#     # masking - trans
#     fontsize=12
#     n_rows, n_cols = 3, 3
#     fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 2 * n_rows))
#
#     axs[0][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_add_code_acc_m.pt').flip(0), label='additive')
#     axs[0][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_sel_code_acc_m.pt'), label='selective')
#     axs[0][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_rnd_code_acc_m.pt'), label='random')
#     axs[0][0].set_ylim(0.2, 1)
#     axs[0][0].set_ylabel('error rate', fontsize=fontsize)
#     axs[0][0].grid(True)
#     axs[0][0].tick_params(axis='x', bottom=False, labelbottom=False)
#     axs[0][0].tick_params(axis='both', which='both', labelsize=fontsize)
#
#
#     # axs[0][0].set_title('Random Transformer', fontsize=fontsize)
#
#     axs[1][0].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_add_mse.pt').flip(0), label='additive')
#     axs[1][0].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_sel_mse.pt'), label='selective')
#     axs[1][0].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_rnd_mse.pt'), label='random')
#     axs[1][0].set_ylim(0, 0.3)
#     # axs[1][0].set_title('reconstruction')
#     axs[1][0].set_ylabel('MSE', fontsize=fontsize)
#     axs[1][0].grid(True)
#     axs[1][0].tick_params(axis='x', bottom=False, labelbottom=False)
#     axs[1][0].tick_params(axis='both', which='both', labelsize=fontsize)
#
#
#
#     axs[2][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_add_acc.pt').flip(0), label='additive')
#     axs[2][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' /  f'{rnd_id}_sel_acc.pt'), label='selective')
#     axs[2][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_rnd_acc.pt'), label='random')
#     axs[2][0].set_ylim(0.2, 1)
#     axs[2][0].set_ylabel('error rate', fontsize=fontsize)
#     axs[2][0].grid(True)
#     # axs[2][0].set_title('image classification')
#     axs[2][0].set_xlabel('# masked codebooks', fontsize=fontsize)
#     axs[2][0].tick_params(axis='both', which='both', labelsize=fontsize)
#
#
#
#     axs[0][1].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' /  f'{sel_id}_add_code_acc_m.pt').flip(0), label='additive')
#     axs[0][1].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_sel_code_acc_m.pt'), label='selective')
#     axs[0][1].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_rnd_code_acc_m.pt'), label='random')
#     axs[0][1].set_ylim(0.2, 1)
#     axs[0][1].grid(True)
#     axs[0][1].tick_params(axis='y', left=False, labelleft=False)
#     axs[0][1].tick_params(axis='x', bottom=False, labelbottom=False)
#
#     # axs[0][1].set_title('Selective Transformer', fontsize=fontsize)
#
#     axs[1][1].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_add_mse.pt').flip(0), label='additive')
#     axs[1][1].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_sel_mse.pt'), label='selective')
#     axs[1][1].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_rnd_mse.pt'), label='random')
#     axs[1][1].grid(True)
#     axs[1][1].tick_params(axis='x', bottom=False, labelbottom=False)
#     axs[1][1].tick_params(axis='y', left=False, labelleft=False)
#     axs[1][1].set_ylim(0, 0.3)
#
#
#
#     axs[2][1].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_add_acc.pt').flip(0), label='additive')
#     axs[2][1].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_sel_acc.pt'), label='selective')
#     axs[2][1].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_rnd_acc.pt'), label='random')
#     axs[2][1].set_ylim(0.2, 1)
#     axs[2][1].grid(True)
#     axs[2][1].set_xlabel('# masked codebooks', fontsize=fontsize)
#     axs[2][1].tick_params(axis='y', left=False, labelleft=False)
#     axs[2][1].tick_params(axis='both', which='both', labelsize=fontsize)
#
#
#
#     axs[0][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_add_code_acc_m.pt').flip(0), label='additive')
#     axs[0][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_sel_code_acc_m.pt'), label='selective')
#     axs[0][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_rnd_code_acc_m.pt'), label='random')
#     axs[0][2].set_ylim(0.2, 1)
#     axs[0][2].grid(True)
#     axs[0][2].tick_params(axis='y', left=False, labelleft=False)
#     axs[0][2].tick_params(axis='x', bottom=False, labelbottom=False)
#
#
#
#     # axs[0][0].set_title('Random Transformer', fontsize=fontsize)
#
#     axs[1][2].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_add_mse.pt').flip(0), label='additive')
#     axs[1][2].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_sel_mse.pt'), label='selective')
#     axs[1][2].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_rnd_mse.pt'), label='random')
#     axs[1][2].set_ylim(0, 0.3)
#     # axs[1][0].set_title('reconstruction')
#     axs[1][2].grid(True)
#     axs[1][2].tick_params(axis='y', left=False, labelleft=False)
#     axs[1][2].tick_params(axis='x', bottom=False, labelbottom=False)
#
#
#
#
#     axs[2][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_add_acc.pt').flip(0), label='additive')
#     axs[2][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' /  f'{rnd_sel_id}_sel_acc.pt'), label='selective')
#     axs[2][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_rnd_acc.pt'), label='random')
#     axs[2][2].set_ylim(0.2, 1)
#     axs[2][2].grid(True)
#     # axs[2][0].set_title('image classification')
#     axs[2][2].set_xlabel('# masked codebooks', fontsize=fontsize)
#     axs[2][2].tick_params(axis='y', left=False, labelleft=False)
#     axs[2][2].tick_params(axis='both', which='both', labelsize=fontsize)
#
#     # axs[0][1].set_yticks([])
#
#
#
#     axs[2][2].legend(fontsize=fontsize)
#
#     axs[0, 0].annotate('Rnd. Transformer', xy=(0.5, 1.1), xycoords='axes fraction',
#                        fontsize=fontsize, ha='center', va='center', fontweight='bold')
#     axs[0, 1].annotate('Sel. Transformer', xy=(0.5, 1.1), xycoords='axes fraction',
#                        fontsize=fontsize, ha='center', va='center', fontweight='bold')
#
#     axs[0, 2].annotate('Rnd. Sel. Transformer', xy=(0.5, 1.1), xycoords='axes fraction',
#                        fontsize=fontsize, ha='center', va='center', fontweight='bold')
#
#
#     axs[0, 0].annotate('completion', xy=(-0.32, 0.5), xycoords='axes fraction',
#                        fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
#     axs[1, 0].annotate('reconstruction', xy=(-0.32, 0.5), xycoords='axes fraction',
#                        fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
#     axs[2, 0].annotate('recognition', xy=(-0.32, 0.5), xycoords='axes fraction',
#                        fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
#     plt.subplots_adjust(wspace=0.03, hspace=0.15)
#
#
#
#     fig.savefig(FIG_DIR + 'mask_eval.pdf', format='pdf', bbox_inches="tight")
#     plt.show()
#


def plot_recons_2(img_id_1, img_id_2, transformer='rnd', file_name='a'):
    n_rows, n_cols = 6 * 2, 5
    fontsize = 32
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))

    sel_data = load_pkl(f'{transformer}_sel')
    plot_x_2(data=sel_data, axs=axs[:(n_rows // 6)], num_images=1, img_id=img_id_1)
    plot_x_2(data=sel_data, axs=axs[(n_rows // 6): 2 * (n_rows // 6)], num_images=1, img_id=img_id_2)

    add_data = load_pkl(f'{transformer}_add')
    plot_x_2(data=add_data, axs=axs[2 * (n_rows // 6): 3 * (n_rows // 6)], num_images=1, img_id=img_id_1, reverse=True)
    plot_x_2(data=add_data, axs=axs[3 * (n_rows // 6): 4 * (n_rows // 6)], num_images=1, img_id=img_id_2)

    rnd_data = load_pkl(f'{transformer}_rnd')
    plot_x_2(data=rnd_data, axs=axs[4 * (n_rows // 6): 5 * (n_rows // 6)], num_images=1, img_id=img_id_1)
    plot_x_2(data=rnd_data, axs=axs[5 * (n_rows // 6): 6 * (n_rows // 6)], num_images=1, img_id=img_id_2)

    plt.subplots_adjust(wspace=0.03, hspace=0.03)

    mask_ratios = np.linspace(0, 80, 5, dtype=int)
    for i in range(n_cols):
        axs[0, i].set_title(mask_ratios[i], fontsize=fontsize)

    util.shift_rows_and_columns(axs, rows_to_shift=np.arange(0, n_rows, 4), vertical_shift=0.005)

    axs[1, 0].annotate('selective', xy=(-0.35, 0.0), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    axs[0, 0].set_ylabel('recon.', fontsize=fontsize)
    axs[1, 0].set_ylabel('mask', fontsize=fontsize)
    axs[2, 0].set_ylabel('recon.', fontsize=fontsize)
    axs[3, 0].set_ylabel('mask', fontsize=fontsize)

    axs[5, 0].annotate('additive', xy=(-0.35, 0.0), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    axs[4, 0].set_ylabel('recon.', fontsize=fontsize)
    axs[5, 0].set_ylabel('mask', fontsize=fontsize)
    axs[6, 0].set_ylabel('recon.', fontsize=fontsize)
    axs[7, 0].set_ylabel('mask', fontsize=fontsize)

    axs[9, 0].annotate('random', xy=(-0.35, 0.0), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    axs[8, 0].set_ylabel('recon.', fontsize=fontsize)
    axs[9, 0].set_ylabel('mask', fontsize=fontsize)
    axs[10, 0].set_ylabel('recon.', fontsize=fontsize)
    axs[11, 0].set_ylabel('mask', fontsize=fontsize)

    axs[0, 2].annotate('masking ratio in %', xy=(0.5, 1.35), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold')

    plt.show()
    fig.savefig(FIG_DIR + f'recons_imgnet_{file_name}.pdf', format='pdf', bbox_inches="tight")


# def plot_recons(img_id, transformer='rnd'):
#     n_rows, n_cols = 3 * 3, 11
#     fontsize = 26
#     fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))
#
#     add_data = load_pkl(f'{transformer}_add_m')
#     plot_x(data=add_data, axs=axs[:n_rows // 3, :], num_images=1, reverse=True, img_id=img_id)
#
#     sel_data = load_pkl(f'{transformer}_sel_m')
#     plot_x(data=sel_data, axs=axs[(n_rows // 3):(n_rows // 3) * 2 :], num_images=1, img_id=img_id)
#
#     rnd_data = load_pkl(f'{transformer}_rnd_m')
#     plot_x(data=rnd_data, axs=axs[(n_rows // 3) * 2 :, :], num_images=1, img_id=img_id)
#
#     plt.subplots_adjust(wspace=0.03, hspace=0.03)
#
#     mask_ratios = np.linspace(0, 100, 11, dtype=int)
#     for i in range(n_cols):
#         axs[0, i].set_title(mask_ratios[i], fontsize=fontsize)
#
#     util.shift_rows_and_columns(axs, rows_to_shift=np.arange(0, n_rows, 3), vertical_shift=0.005)
#
#
#     axs[1, 0].annotate('additive', xy=(-0.3, 0.5), xycoords='axes fraction',
#                        fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
#     axs[0, 0].set_ylabel('no compl.', fontsize=fontsize)
#     axs[1, 0].set_ylabel('compl.', fontsize=fontsize)
#     axs[2, 0].set_ylabel('mask', fontsize=fontsize)
#
#
#     axs[4, 0].annotate('selective', xy=(-0.3, 0.5), xycoords='axes fraction',
#                        fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
#     axs[3, 0].set_ylabel('no compl.', fontsize=fontsize)
#     axs[4, 0].set_ylabel('compl.', fontsize=fontsize)
#     axs[5, 0].set_ylabel('mask', fontsize=fontsize)
#
#     axs[7, 0].annotate('random', xy=(-0.3, 0.5), xycoords='axes fraction',
#                        fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
#     axs[6, 0].set_ylabel('no compl.', fontsize=fontsize)
#     axs[7, 0].set_ylabel('compl.', fontsize=fontsize)
#     axs[8, 0].set_ylabel('mask', fontsize=fontsize)
#
#     axs[0, 5].annotate('masking ratio in %', xy=(0.5, 1.3), xycoords='axes fraction',
#                        fontsize=fontsize, ha='center', va='center', fontweight='bold')
#     plt.show()
#     fig.savefig(FIG_DIR + 'recons_imgnet.pdf', format='pdf', bbox_inches="tight")

def plot_masking_ratios_2(rnd_id=None, sel_id=None, rnd_sel_id=None):
    # masking - trans
    fontsize = 8
    n_rows, n_cols = 3, 3
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 1 * n_rows))
    axs[0][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_rnd_code_acc_m.pt'), label='random',
                   linestyle='solid', color='#2ca02c')
    axs[0][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_sel_code_acc_m.pt'), label='selective',
                   linestyle='dashed', color='#ff7f0e')
    axs[0][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_add_code_acc_m.pt').flip(0),
                   label='additive', linestyle='dotted', color='#1f77b4')
    axs[0][0].set_ylim(0.2, 1)
    axs[0][0].set_ylabel('error rate', fontsize=fontsize)
    axs[0][0].grid(True)
    axs[0][0].tick_params(axis='x', bottom=False, labelbottom=False)
    axs[0][0].tick_params(axis='both', which='both', labelsize=fontsize)

    # axs[0][0].set_title('Random Transformer', fontsize=fontsize)
    axs[1][0].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_rnd_mse.pt'), label='random',
                   linestyle='solid', color='#2ca02c')
    axs[1][0].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_sel_mse.pt'), label='selective',
                   linestyle='dashed', color='#ff7f0e')
    axs[1][0].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_add_mse.pt').flip(0), label='additive',
                   linestyle='dotted', color='#1f77b4')
    axs[1][0].set_ylim(0, 0.3)
    axs[1][0].set_ylabel('MSE', fontsize=fontsize)
    axs[1][0].grid(True)
    axs[1][0].tick_params(axis='x', bottom=False, labelbottom=False)
    axs[1][0].tick_params(axis='both', which='both', labelsize=fontsize)

    axs[2][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_rnd_acc.pt'), label='random',
                   linestyle='solid', color='#2ca02c')
    axs[2][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_sel_acc.pt'), label='selective',
                   linestyle='dashed', color='#ff7f0e')
    axs[2][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_id}_add_acc.pt').flip(0), label='additive',
                   linestyle='dotted', color='#1f77b4')
    axs[2][0].set_ylim(0.2, 1)
    axs[2][0].set_ylabel('error rate', fontsize=fontsize)
    axs[2][0].grid(True)
    # axs[2][0].set_title('image classification')
    axs[2][0].set_xlabel('masking level', fontsize=fontsize)
    axs[2][0].tick_params(axis='both', which='both', labelsize=fontsize)

    axs[0][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_rnd_code_acc_m.pt'), label='random',
                   linestyle='solid', color='#2ca02c')
    axs[0][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_sel_code_acc_m.pt'), label='selective',
                   linestyle='dashed', color='#ff7f0e')
    axs[0][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_add_code_acc_m.pt').flip(0),
                   label='additive', linestyle='dotted', color='#1f77b4')
    axs[0][2].set_ylim(0.2, 1)
    axs[0][2].grid(True)
    axs[0][2].tick_params(axis='y', left=False, labelleft=False)
    axs[0][2].tick_params(axis='x', bottom=False, labelbottom=False)
    axs[0][2].tick_params(axis='both', which='both', labelsize=fontsize)

    # axs[0][1].set_title('Selective Transformer', fontsize=fontsize)
    axs[1][2].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_rnd_mse.pt'), label='random',
                   linestyle='solid', color='#2ca02c')
    axs[1][2].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_sel_mse.pt'), label='selective',
                   linestyle='dashed', color='#ff7f0e')
    axs[1][2].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_add_mse.pt').flip(0), label='additive',
                   linestyle='dotted', color='#1f77b4')
    axs[1][2].tick_params(axis='y', left=False, labelleft=False)
    axs[1][2].tick_params(axis='x', bottom=False, labelbottom=False)
    axs[1][2].tick_params(axis='both', which='both', labelsize=fontsize)
    axs[1][2].grid(True)

    # axs[1][1].set_ylim(0, 0.3)
    axs[1][0].set_ylabel('MSE', fontsize=fontsize)
    axs[1][1].grid(True)
    axs[1][1].tick_params(axis='x', bottom=False, labelbottom=False)
    axs[1][1].tick_params(axis='y', left=False, labelleft=False)
    axs[1][1].tick_params(axis='both', which='both', labelsize=fontsize)

    #
    # axs[1][1].grid(True)
    # axs[1][1].tick_params(axis='x', bottom=False, labelbottom=False)
    # axs[1][1].tick_params(axis='y', left=False, labelleft=False)
    # axs[1][1].set_ylim(0, 0.3)

    axs[2][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_rnd_acc.pt'), label='random',
                   linestyle='solid', color='#2ca02c')
    axs[2][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_sel_acc.pt'), label='selective',
                   linestyle='dashed', color='#ff7f0e')
    axs[2][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{sel_id}_add_acc.pt').flip(0), label='additive',
                   linestyle='dotted', color='#1f77b4')

    axs[2][2].set_ylim(0.2, 1)
    axs[2][2].grid(True)
    axs[2][2].set_xlabel('masking level', fontsize=fontsize)
    axs[2][2].tick_params(axis='y', left=False, labelleft=False)
    axs[2][2].tick_params(axis='both', which='both', labelsize=fontsize)

    axs[0][1].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_sel_code_acc_m.pt'),
                   label='selective', linestyle='dashed', color='#ff7f0e')
    axs[0][1].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_rnd_code_acc_m.pt'),
                   label='random', linestyle='solid', color='#2ca02c')
    axs[0][1].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_add_code_acc_m.pt').flip(0),
                   label='additive', linestyle='dotted', color='#1f77b4')

    axs[0][1].set_ylim(0.2, 1)

    axs[0][1].grid(True)
    axs[0][1].tick_params(axis='y', left=False, labelleft=False)
    axs[0][1].tick_params(axis='x', bottom=False, labelbottom=False)
    axs[0][1].tick_params(axis='both', which='both', labelsize=fontsize)

    # axs[0][0].set_title('Random Transformer', fontsize=fontsize)
    axs[1][1].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_rnd_mse.pt'), label='random',
                   linestyle='solid', color='#2ca02c')
    axs[1][1].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_sel_mse.pt'), label='selective',
                   linestyle='dashed', color='#ff7f0e')
    axs[1][1].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_add_mse.pt').flip(0), label='additive',
                   linestyle='dotted', color='#1f77b4')
    axs[1][1].set_ylim(0, 0.3)
    # axs[1][0].set_title('reconstruction')
    # axs[1][0].set_ylabel('MSE', fontsize=fontsize)
    axs[1][1].grid(True)
    axs[1][1].tick_params(axis='x', bottom=False, labelbottom=False)
    axs[1][1].tick_params(axis='y', left=False, labelleft=False)
    axs[1][1].tick_params(axis='both', which='both', labelsize=fontsize)

    axs[2][1].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_rnd_acc.pt'), label='random',
                   linestyle='solid', color='#2ca02c')
    axs[2][1].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_sel_acc.pt'), label='selective',
                   linestyle='dashed', color='#ff7f0e')
    axs[2][1].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / f'{rnd_sel_id}_add_acc.pt').flip(0),
                   label='additive', linestyle='dotted', color='#1f77b4')
    axs[2][1].set_ylim(0.2, 1)
    axs[2][1].grid(True)
    # axs[2][0].set_title('image classification')
    axs[2][1].set_xlabel('masking level', fontsize=fontsize)
    axs[2][1].tick_params(axis='y', left=False, labelleft=False)
    axs[2][1].tick_params(axis='both', which='both', labelsize=fontsize)

    # axs[0][1].set_yticks([])

    axs[2][2].legend(fontsize=fontsize)

    axs[0, 0].annotate('random transformer', xy=(0.5, 1.1), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold')
    axs[0, 2].annotate('selective transformer', xy=(0.5, 1.1), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold')

    axs[0, 1].annotate('random-selective transformer', xy=(0.5, 1.1), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold')

    axs[0, 0].annotate('compl.', xy=(-0.22, 0.5), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    axs[1, 0].annotate('recons.', xy=(-0.22, 0.5), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    axs[2, 0].annotate('recog.', xy=(-0.22, 0.5), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    plt.subplots_adjust(wspace=0.01, hspace=0.03)

    axs[0][0].set_yticks([0.4, 0.6, 0.8])
    axs[0][0].set_yticklabels(['.40', '.60', '.80'])

    axs[0][0].set_ylim(0.2, 1)
    axs[0][2].set_yticks([0.4, 0.6, 0.8])
    axs[0][2].set_ylim(0.2, 1)
    axs[0][1].set_yticks([0.4, 0.6, 0.8])
    axs[0][1].set_ylim(0.2, 1)

    axs[1][0].set_yticks([0.05, 0.1, 0.15])
    axs[1][0].set_yticklabels(['.05', '.01', '.15'])

    axs[1][0].set_ylim(0.0, 0.2)
    axs[1][2].set_yticks([0.05, 0.1, 0.15])
    axs[1][2].set_ylim(0.0, 0.2)
    axs[1][1].set_yticks([0.05, 0.1, 0.15])
    axs[1][1].set_ylim(0.0, 0.2)

    axs[2][0].set_yticks([0.4, 0.6, 0.8])
    axs[2][0].set_yticklabels(['.40', '.60', '.80'])
    axs[2][0].set_ylim(0.2, 1)
    axs[2][2].set_yticks([0.4, 0.6, 0.8])
    axs[2][2].set_ylim(0.2, 1)
    axs[2][1].set_yticks([0.4, 0.6, 0.8])
    axs[2][1].set_ylim(0.2, 1)

    axs[2][0].set_xticks([0, 100, 200, 300, 400])
    axs[2][0].set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])

    axs[2][2].set_xticks([0, 100, 200, 300, 400])
    axs[2][2].set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])

    axs[2][1].set_xticks([0, 100, 200, 300, 400])
    axs[2][1].set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])

    fig.savefig(FIG_DIR + 'mask_eval.pdf', format='pdf', bbox_inches="tight")
    plt.show()


def make_sel_comparisons():
    n_rows, n_cols = 2, 2
    fontsize = 12
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 1.5 * n_rows))
    # run_ids = [('VQVAET-501', 'selective'), ('VQVAET-511', 'selective'), ('VQVAET-516', 'selective'),
    #            ('VQVAET-515', 'selective'), ('VQVAET-523', 'selective'), ('VQVAET-524', 'selective'),
    #            ('VQVAET-525', 'selective'), ('VQVAET-526', 'selective'),
    #            ('VQVAET-518', 'rnd sel'), ('VQVAET-517', 'rnd sel'),
    #            ('VQVAET-521', 'rnd sel'),  ('VQVAET-522', 'rnd sel'),
    #            ('VQVAET-502', 'random')]
    # 511

    run_ids = [  # ('VQVAET-551', ''), ('VQVAET-550', ''), ('VQVAET-549', ''), ('VQVAET-548', ''),
        ('VQVAET-547', ''),
         ('VQVAET-545', ''),
        ('VQVAET-511', ''), ('VQVAET-518', 'rnd sel'), ('VQVAET-517', 'rnd sel'),
        ('VQVAET-521', 'rnd sel')]

    update = False
    plot_up_to = 81
    for run_id in run_ids:
        if run_id[1][:2] == 'rn':
            linestyle = '--'
            color = '#ff7f0e'
        else:
            linestyle = '-'
            color = '#2ca02c'
        ser = get_series(key_prefix='val', key_suffix='rnd_acc_masked_only', run_id=run_id[0], load=True, update=update)
        ax[0][0].plot(1 - ser[:plot_up_to], label=run_id[1], linestyle=linestyle, color=color)
        ax[0][0].grid(True)
        # ax[0][0].set_title('rnd val codebook accuracy')

    for run_id in run_ids:
        if run_id[1][:2] == 'rn':
            linestyle = '--'
            color = '#ff7f0e'
        else:
            linestyle = '-'
            color = '#2ca02c'
        ser = get_series(key_prefix='val', key_suffix='rnd_class_acc', run_id=run_id[0], load=True, update=update)
        ax[1][0].plot(1 - ser[:plot_up_to], label=run_id[1], linestyle=linestyle, color=color)
        ax[1][0].grid(True)

        # ax[1][0].set_title('rnd val class accuracy')

    for run_id in run_ids:
        if run_id[1][:2] == 'rn':
            linestyle = '--'
            color = '#ff7f0e'
        else:
            linestyle = '-'
            color = '#2ca02c'
        ser = get_series(key_prefix='val', key_suffix='selective_desc_acc_masked_only', run_id=run_id[0], load=True,
                         update=update)
        ax[0][1].plot(1 - ser[:plot_up_to], label=run_id[1], linestyle=linestyle, color=color)
        ax[0][1].grid(True)

        # ax[0][1].set_title('sel codebook acc')

    for run_id in run_ids:
        if run_id[1][:2] == 'rn':
            linestyle = '--'
            color = '#ff7f0e'
        else:
            linestyle = '-'
            color = '#2ca02c'
        ser = get_series(key_prefix='val', key_suffix='selective_desc_class_acc', run_id=run_id[0], load=True,
                         update=update)
        ax[1][1].plot(1 - ser[:plot_up_to], label=run_id[1], linestyle=linestyle, color=color)
        ax[1][1].grid(True)

        # ax[1][1].set_title('sel class acc')

    ax[0][0].set_ylim(0.75, 1)
    ax[0][0].set_yticks([0.8, 0.85, 0.9, 0.95])
    # ax[0][1].tick_params(axis='y', left=False, labelleft=False)
    ax[0][0].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[0][0].tick_params(axis='both', which='both', labelsize=fontsize)

    ax[0][1].set_ylim(0.75, 1)
    ax[0][1].set_yticks([0.8, 0.85, 0.9, 0.95])
    ax[0][0].set_yticklabels(['.80', '.85', '.90', '.95'])

    ax[0][1].tick_params(axis='y', left=False, labelleft=False)
    ax[0][1].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[0][1].tick_params(axis='both', which='both', labelsize=fontsize)

    ax[1][0].set_ylim(0.25, 1)
    ax[1][0].set_yticks([0.4, 0.55, 0.7, 0.85])

    # ax[0][1].tick_params(axis='y', left=False, labelleft=False)
    # ax[0][1].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[1][0].tick_params(axis='both', which='both', labelsize=fontsize)
    ax[1][0].set_yticklabels(['.40', '.55', '.70', '.85'])

    ax[1][1].set_ylim(0.25, 1)
    ax[1][1].set_yticks([0.4, 0.55, 0.7, 0.85])

    ax[1][1].tick_params(axis='y', left=False, labelleft=False)
    # ax[1][1].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[1][1].tick_params(axis='both', which='both', labelsize=fontsize)

    ax[1][0].set_xlabel('# Epochs', fontsize=fontsize)
    ax[1][1].set_xlabel('# Epochs', fontsize=fontsize)

    ax[0][0].set_ylabel('error rate', fontsize=fontsize)
    ax[1][0].set_ylabel('error rate', fontsize=fontsize)

    # ax[1][1].legend(fontsize=fontsize)
    custom_lines = [
        Line2D([0], [0], linestyle='--', label='random-selective', color='#ff7f0e'),
        Line2D([0], [0], linestyle='-', label='selective', color='#2ca02c')
    ]

    ax[0][0].legend(handles=custom_lines, fontsize=fontsize)
    plt.subplots_adjust(wspace=0.015, hspace=0.03)

    ax[0, 0].annotate('validated with\nrandom masking', xy=(0.5, 1.2), xycoords='axes fraction',
                      fontsize=fontsize, ha='center', va='center', fontweight='bold')
    ax[0, 1].annotate('validated with\nselective masking', xy=(0.5, 1.2), xycoords='axes fraction',
                      fontsize=fontsize, ha='center', va='center', fontweight='bold')
    ax[0, 0].annotate('compl.', xy=(-0.32, 0.5), xycoords='axes fraction',
                      fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    ax[1, 0].annotate('recog.', xy=(-0.31, 0.5), xycoords='axes fraction',
                      fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    #
    # for run_id in run_ids:
    #     ser = get_series(key_prefix='val', key_suffix='selective_desc_acc_masked_only', run_id=run_id[0], load=True)
    #     ax[1][0].plot(ser, label=run_id[1])
    # for run_id in run_ids:
    #     ser = get_series(key_prefix='val', key_suffix='selective_desc_class_acc', run_id=run_id[0], load=True)
    #     ax[1][1].plot(ser, label=run_id[1])
    fig.savefig(FIG_DIR + 'sel_comp.pdf', format='pdf', bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    #     # util.set_seed(0)
    #     # good ides
    #     # plot_recons_2(transformer='VQVAET-512', img_id_1=12, img_id_2=10, file_name='a') # 512 is rnd
    #     # plot_recons_2(transformer='VQVAET-512', img_id_1=8, img_id_2=9, file_name='b')  # 512 is rnd
    #     # plot_recons_2(transformer='VQVAET-512', img_id_1=33, img_id_2=19,  file_name='c')  # 512 is rnd
    #     # plot_recons_2(transformer='VQVAET-512', img_id_1=19, img_id_2=56,  file_name='e')  # 512 is rnd
    #     #
    make_sel_comparisons()
#     # 12 , 10, 8, 9, 19,  33, 56
#
#     # plot_token_val()
#     # plot_masking_ratios()
#     # plot_masking_ratios(rnd_id='VQVAET-512', sel_id='VQVAET-511', rnd_sel_id='VQVAET-522')
#     plot_masking_ratios_2(rnd_id='VQVAET-502', sel_id='VQVAET-511', rnd_sel_id='VQVAET-522')
#
#
#
#     pass
