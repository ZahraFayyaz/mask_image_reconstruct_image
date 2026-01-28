from multiprocessing.util import log_to_stderr

import numpy as np
import argparse, math, sys, os
import torch
from torch import nn
import distributed as dist
from transformers import DistilBertForMaskedLM, DistilBertConfig
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



FIG_DIR = '/Users/rathjjgf/Desktop/zahra_fig/'

def load_pkl(file_name):
    with open(util.DATA_DIR / 'plots' / (file_name + '.pkl'), "rb") as f:
        recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without = pickle.load(f)
    return recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without

def plot_x(data, axs, num_images, img_id, reverse=False):
    recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without = data
    if reverse:
        recons.reverse(), masks.reverse(), perc.reverse(), confidences.reverse(), normed_confidences.reverse(), correctly_classified.reverse(),  recon_without.reverse(), correctly_classified_without.reverse()
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
        recons.reverse(), masks.reverse(), perc.reverse(), confidences.reverse(), normed_confidences.reverse(), correctly_classified.reverse(),  recon_without.reverse(), correctly_classified_without.reverse()
    for j in range(num_images):
        for k in range(0, len(masks), 2):
            axs[j * 2 + 0][k].imshow(util.denormalize(recons[k])[img_id].permute(1, 2, 0))
            util.set_border(axs[j * 3 + 1][k], correctly_classified[k][img_id])

            axs[2 * j + 1][k].imshow(masks[k].reshape(-1, 20, 20, 1)[img_id], vmin=0, vmax=1)
            util.set_border(axs[j * 3 + 2][k], correctly_classified[k][img_id], no_color=True)



def get_series(key_prefix, key_suffix, run_id, update=False, load=True):
    path = util.RUNS_DIR / run_id / (key_suffix + '.npy')
    if update or not path.exists():
        run = neptune.init_run(with_id=run_id, mode='read-only', project='Vqvae-transformer')
        values = np.array(list(run[key_prefix + '/' + key_suffix].fetch_values()['value']))
        np.save(str(path), values)
        run.stop()
    if load:
        return np.load(path)
    return None


def plot_token_val():
    n_rows, n_cols = 2, 2
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))
    run_ids = [('VQVAET-501', 'selective m 1'), ('VQVAET-511', 'selective m 2'),
               ('VQVAET-502', 'random m 1'),  ('VQVAET-512', 'random m 2'),
               ('VQVAET-504', 'selective 1'),  ('VQVAET-514', 'selective 2'),
               ('VQVAET-505', 'random 1'),  ('VQVAET-513', 'random 2')]

    for run_id in run_ids:
        if run_id[1][:2] == 'se':
            linestyle = '--'
        else:
            linestyle = '-'
        ser = get_series(key_prefix='val', key_suffix='rnd_acc_masked_only', run_id=run_id[0], load=True, update=True)
        ax[0][0].plot(ser, label=run_id[1], linestyle=linestyle)

    for run_id in run_ids:
        if run_id[1][:2] == 'se':
            linestyle = '--'
        else:
            linestyle = '-'
        ser = get_series(key_prefix='val', key_suffix='rnd_class_acc', run_id=run_id[0], load=True, update=True)
        ax[0][1].plot(ser,  label=run_id[1], linestyle=linestyle)

    for run_id in run_ids:
        if run_id[1][:2] == 'se':
            linestyle = '--'
        else:
            linestyle = '-'
        ser = get_series(key_prefix='val', key_suffix='selective_desc_acc_masked_only', run_id=run_id[0], load=True, update=True)
        ax[1][0].plot(ser, label=run_id[1], linestyle=linestyle)

    for run_id in run_ids:
        if run_id[1][:2] == 'se':
            linestyle = '--'
        else:
            linestyle = '-'
        ser = get_series(key_prefix='val', key_suffix='selective_desc_class_acc', run_id=run_id[0], load=True, update=True)
        ax[1][1].plot(ser,  label=run_id[1], linestyle=linestyle)

    ax[0][0].set_ylim(0, 0.3)
    ax[1][0].set_ylim(0, 0.3)

    ax[0][1].set_ylim(0, 0.7)
    ax[1][1].set_ylim(0, 0.7)




    ax[0][0].set_title('val codebook accuracy')
    ax[0][1].set_title('val class accuracy')
    ax[0][0].set_ylabel('Accuracy')
    ax[1][0].set_xlabel('# Epochs')
    ax[1][1].set_xlabel('# Epochs')

    ax[1][0].set_ylabel('Accuracy')

    ax[1][1].legend()

    #
    # for run_id in run_ids:
    #     ser = get_series(key_prefix='val', key_suffix='selective_desc_acc_masked_only', run_id=run_id[0], load=True)
    #     ax[1][0].plot(ser, label=run_id[1])
    # for run_id in run_ids:
    #     ser = get_series(key_prefix='val', key_suffix='selective_desc_class_acc', run_id=run_id[0], load=True)
    #     ax[1][1].plot(ser, label=run_id[1])

    plt.show()


def plot_masking_ratios():
    # masking - trans
    n_rows, n_cols = 2, 3
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))

    axs[0][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'add_rnd_code_acc.pt'), label='additive')
    axs[0][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'sel_rnd_code_acc.pt').flip(0), label='selective')
    axs[0][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'rnd_rnd_code_acc.pt').flip(0), label='random')
    axs[0][0].set_ylim(0, 1)
    axs[0][0].set_ylabel('class. error')
    axs[0][0].set_title('codebook classification')


    axs[0][1].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'add_rnd_mse.pt'), label='additive')
    axs[0][1].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'sel_rnd_mse.pt').flip(0), label='selective')
    axs[0][1].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'rnd_rnd_mse.pt').flip(0), label='random')
    axs[0][1].set_ylim(0, 1)
    axs[0][1].set_title('reconstruction')
    axs[0][1].set_ylabel('MSE')


    axs[0][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'add_rnd_acc.pt'), label='additive')
    axs[0][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'sel_rnd_acc.pt').flip(0), label='selective')
    axs[0][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'rnd_rnd_acc.pt').flip(0), label='random')
    axs[0][2].set_ylim(0, 1)
    axs[0][2].set_ylabel('class. error')
    axs[0][2].set_title('image classification')

    axs[0][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'add_sel_code_acc.pt'), label='additive')
    axs[0][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'sel_sel_code_acc.pt').flip(0), label='selective')
    axs[0][0].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'rnd_sel_code_acc.pt').flip(0), label='random')
    axs[0][0].set_ylim(0, 1)
    axs[0][0].set_ylabel('class. error')
    axs[0][0].set_title('codebook classification')


    axs[1][1].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'add_sel_mse.pt'), label='additive')
    axs[1][1].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'sel_sel_mse.pt').flip(0), label='selective')
    axs[1][1].plot(torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'rnd_sel_mse.pt').flip(0), label='random')
    axs[1][1].set_ylim(0, 1)
    axs[1][1].set_ylabel('MSE')
    axs[1][1].set_xlabel('# masked codebooks')


    axs[1][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'add_sel_acc.pt'), label='additive')
    axs[1][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'sel_sel_acc.pt').flip(0), label='selective')
    axs[1][2].plot(1 - torch.load(util.DATA_DIR / 'mask_ratio_eval' / 'rnd_sel_acc.pt').flip(0), label='random')
    axs[1][2].set_ylim(0, 1)
    axs[1][2].set_ylabel('class. error')
    axs[1][2].set_xlabel('# masked codebooks')


    plt.legend()
    plt.show()



def plot_recons_2(img_id_1, img_id_2, transformer='rnd'):
    n_rows, n_cols = 6 * 2, 6
    fontsize = 26
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))

    sel_data = load_pkl(f'{transformer}_sel_m')
    plot_x_2(data=sel_data, axs=axs[:(n_rows // 2)], num_images=1, img_id=img_id_1)
    plot_x_2(data=sel_data, axs=axs[(n_rows // 2): 2 * (n_rows // 2)], num_images=1, img_id=img_id_2)

    add_data = load_pkl(f'{transformer}_add_m')
    plot_x_2(data=add_data, axs=axs[2 * (n_rows // 2): 3 *(n_rows // 2)], num_images=1, img_id=img_id_1)
    plot_x_2(data=add_data, axs=axs[3 * (n_rows // 2): 4 *(n_rows // 2)], num_images=1, img_id=img_id_2)

    rnd_data = load_pkl(f'{transformer}_rnd_m')
    plot_x_2(data=rnd_data, axs=axs[4 * (n_rows // 2): 5 *(n_rows // 2)], num_images=1, img_id=img_id_1)
    plot_x_2(data=rnd_data, axs=axs[5 * (n_rows // 2): 6 * (n_rows // 2)], num_images=1, img_id=img_id_2)


    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    mask_ratios = np.linspace(0, 100, 11, dtype=int)
    for i in range(n_cols):
        axs[0, i].set_title(mask_ratios[i], fontsize=fontsize)

    util.shift_rows_and_columns(axs, rows_to_shift=np.arange(0, n_rows, 3))





    axs[1, 0].annotate('selective', xy=(-0.3, 1.0), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    axs[0, 0].set_ylabel('compl.', fontsize=fontsize)
    axs[1, 0].set_ylabel('mask', fontsize=fontsize)
    axs[2, 0].set_ylabel('compl.', fontsize=fontsize)
    axs[3, 0].set_ylabel('mask', fontsize=fontsize)


    axs[5, 0].annotate('additive', xy=(-0.3, 1.0), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    axs[4, 0].set_ylabel('compl.', fontsize=fontsize)
    axs[5, 0].set_ylabel('mask', fontsize=fontsize)
    axs[6, 0].set_ylabel('compl.', fontsize=fontsize)
    axs[7, 0].set_ylabel('mask', fontsize=fontsize)

    axs[9, 0].annotate('random', xy=(-0.3, 0.0), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    axs[8, 0].set_ylabel('compl.', fontsize=fontsize)
    axs[9, 0].set_ylabel('mask', fontsize=fontsize)
    axs[10, 0].set_ylabel('compl.', fontsize=fontsize)
    axs[11, 0].set_ylabel('mask', fontsize=fontsize)

    axs[0, 2].annotate('masking ratio in %', xy=(0.5, 1.3), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold')

    plt.show()
    # fig.savefig(FIG_DIR + 'recons_imgnet.pdf', format='pdf', bbox_inches="tight")



def plot_recons(img_id, transformer='rnd'):
    n_rows, n_cols = 3 * 3, 11
    fontsize = 26
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))

    add_data = load_pkl(f'{transformer}_add_m')
    plot_x(data=add_data, axs=axs[:n_rows // 3, :], num_images=1, reverse=True, img_id=img_id)

    sel_data = load_pkl(f'{transformer}_sel_m')
    plot_x(data=sel_data, axs=axs[(n_rows // 3):(n_rows // 3) * 2 :], num_images=1, img_id=img_id)

    rnd_data = load_pkl(f'{transformer}_rnd_m')
    plot_x(data=rnd_data, axs=axs[(n_rows // 3) * 2 :, :], num_images=1, img_id=img_id)

    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    mask_ratios = np.linspace(0, 100, 11, dtype=int)
    for i in range(n_cols):
        axs[0, i].set_title(mask_ratios[i], fontsize=fontsize)

    util.shift_rows_and_columns(axs, rows_to_shift=np.arange(0, n_rows, 3))


    axs[1, 0].annotate('additive', xy=(-0.3, 0.5), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    axs[0, 0].set_ylabel('no compl.', fontsize=fontsize)
    axs[1, 0].set_ylabel('compl.', fontsize=fontsize)
    axs[2, 0].set_ylabel('mask', fontsize=fontsize)


    axs[4, 0].annotate('selective', xy=(-0.3, 0.5), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    axs[3, 0].set_ylabel('no compl.', fontsize=fontsize)
    axs[4, 0].set_ylabel('compl.', fontsize=fontsize)
    axs[5, 0].set_ylabel('mask', fontsize=fontsize)

    axs[7, 0].annotate('random', xy=(-0.3, 0.5), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold', rotation=90)
    axs[6, 0].set_ylabel('no compl.', fontsize=fontsize)
    axs[7, 0].set_ylabel('compl.', fontsize=fontsize)
    axs[8, 0].set_ylabel('mask', fontsize=fontsize)

    #
    axs[0, 5].annotate('masking ratio in %', xy=(0.5, 1.3), xycoords='axes fraction',
                       fontsize=fontsize, ha='center', va='center', fontweight='bold')

    plt.show()
    fig.savefig(FIG_DIR + 'recons_imgnet.pdf', format='pdf', bbox_inches="tight")


if __name__ == "__main__":

    util.set_seed(0)

    # plot_recons(transformer='rnd', img_id=6)
    # plot_recons(transformer='rnd', img_id=9)
    # plot_recons(transformer='rnd', img_id=14)
    # plot_recons(transformer='rnd', img_id=15)
    # plot_recons(transformer='rnd', img_id=17)


    # plot_token_val()
    plot_masking_ratios()

    pass
