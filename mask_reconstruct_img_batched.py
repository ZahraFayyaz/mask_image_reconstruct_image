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

from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim

HOSTNAME = socket.gethostname()
#
# if HOSTNAME in {'lux47', 'gpu01', 'gpu02', 'gpu03'}:
#     DATASETS_DIR = Path('/local/rathjjgf/datasets/')
# else:
#     DATASETS_DIR = Path.home() / 'datasets'

DATA_DIR = Path(__file__).parent / 'data'
TMP_DIR = DATA_DIR / 'tmp'
RUNS_DIR = DATA_DIR / 'runs'

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.cuda.set_device(2)
    torch.cuda.empty_cache()
else:
    DEVICE = 'cpu'

if HOSTNAME == 'add here':
    QUANTIZED_EPOCH_PATH = '/home/abghamtm/work/masking_comparison/checkpoint/vqvae/quantized_epoch80_flat_vqvae80x80_144x456codebook.npy'
    INDICES_PATH = '/home/abghamtm/work/masking_comparison/checkpoint/vqvae/indices_epoch80_flat_vqvae80x80_144x456codebook.npy'
    LABELS_PATH = '/home/abghamtm/work/masking_comparison/checkpoint/vqvae/labels_epoch80_flat_vqvae80x80_144x456codebook.npy'
    EIGHTY_EIGHTY_PATH = '/home/abghamtm/work/masking_comparison/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100.pt'
    VQVAE_PATH = '/home/abghamtm/work/masking_comparison/checkpoint/vqvae/model_epoch80_flat_vqvae80x80_144x456codebook.pth'
else:
    DATA_DIR = Path(__file__).parent / 'data'
    TMP_DIR = DATA_DIR / 'tmp'
    RUNS_DIR = DATA_DIR / 'runs'
    QUANTIZED_EPOCH_PATH = DATA_DIR / 'vqvae' / 'quantized_epoch80_flat_vqvae80x80_144x456codebook.npy'
    INDICES_PATH = DATA_DIR / 'vqvae' / 'indices_epoch80_flat_vqvae80x80_144x456codebook.npy'
    LABELS_PATH = DATA_DIR / 'vqvae' / 'labels_epoch80_flat_vqvae80x80_144x456codebook.npy'
    EIGHTY_EIGHTY_PATH = DATA_DIR / 'vqvae' / '80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100.pt'
    VQVAE_PATH = DATA_DIR / 'vqvae' / 'model_epoch80_flat_vqvae80x80_144x456codebook.pth'


def full_mask(q, indices):
    mask_pattern = torch.ones_like(indices, dtype=torch.bool)
    masked_q = q.clone()  # shallow copy
    masked_q[mask_pattern] = 0  # Assuming 0 is the mask token
    mask_indices = indices.clone()
    mask_indices[~mask_pattern] = -100  # Assuming -100 is the mask label token
    return masked_q, mask_indices, mask_pattern


def hide_extras(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def hide_all_extras(axs):
    for row_id in range(axs.shape[0]):
        for column_id in range(axs.shape[1]):
            hide_extras(axs[row_id][column_id])


def shift_rows_and_columns(axs, rows_to_shift=(), cols_to_shift=(), vertical_shift=0.01, horizontal_shit=0.01):
    cumulative_vertical_shift = 0
    if len(axs.shape) == 1:
        for j in range(axs.shape[0]):
            cumulative_horizontal_shift = 0
            if j in rows_to_shift:
                cumulative_vertical_shift += vertical_shift
            # if axs.shape
            for i in range(axs.shape[0]):
                ax = axs[i]
                pos = ax.get_position()
                if i in cols_to_shift:
                    cumulative_horizontal_shift += horizontal_shit
                ax.set_position(
                    [pos.x0 + cumulative_horizontal_shift, pos.y0 - cumulative_vertical_shift, pos.width, pos.height])
    else:
        cumulative_vertical_shift = 0
        for j in range(axs.shape[0]):
            cumulative_horizontal_shift = 0
            if j in rows_to_shift:
                cumulative_vertical_shift += vertical_shift
            # if axs.shape
            for i in range(axs.shape[1]):
                ax = axs[j, i]
                pos = ax.get_position()
                if i in cols_to_shift:
                    cumulative_horizontal_shift += horizontal_shit
                ax.set_position(
                    [pos.x0 + cumulative_horizontal_shift, pos.y0 - cumulative_vertical_shift, pos.width, pos.height])



def conc_unique_elements(x: torch.Tensor, y: torch.Tensor, n_elements=1) -> torch.Tensor:
    # Expand dimensions for broadcasting
    x_exp = x.unsqueeze(2)  # (a, n, 1)
    y_exp = y.unsqueeze(1)  # (a, 1, m)
    # Identify elements in y not present in x
    mask = ~(x_exp == y_exp).any(dim=1)  # (a, m)
    idx = torch.argsort(-mask.float(), stable=True)[:, :n_elements]
    to_conc = y.gather(1, idx)
    # Concatenate result
    return torch.cat((x, to_conc), dim=1)


def load_embedding_space(path=QUANTIZED_EPOCH_PATH):
    quantizes = np.load(path)
    quant_b = quantizes
    n, c, h, w = quantizes.shape
    quantizes = quantizes.transpose(0, 2, 3, 1)
    quantizes = quantizes.reshape(n, h * w, c)
    return quantizes, quant_b


def load_indices(path=INDICES_PATH):
    indices = np.load(path)
    n, h, w = indices.shape
    indices = indices.reshape(n, h * w)
    return indices


def load_labels(path=LABELS_PATH):
    labels = np.load(path)
    # labels = torch.from_numpy(labels)
    return labels


def vqvae_model_setup(ckpt_vqvae):
    model_vqvae = FlatVQVAE().to(DEVICE)
    model_vqvae.load_state_dict(torch.load(ckpt_vqvae, map_location=DEVICE))
    model_vqvae = model_vqvae.to(DEVICE)
    model_vqvae.eval()
    return model_vqvae


def transformer_setup(args, n_token, vocab_size, d_embed_vec):
    cfg = DistilBertConfig(vocab_size=vocab_size, hidden_size=d_embed_vec, sinusoidal_pos_embds=False, n_layers=6,
                           n_heads=4, max_position_embeddings=n_token)
    model_distil = DistilBertForMaskedLM(cfg).to(DEVICE)
    model_state = torch.load(args.ckpt_distil_combined, map_location=DEVICE)
    model_distil.load_state_dict(model_state)
    model_distil = model_distil.to(DEVICE)
    model_distil.eval()
    return model_distil


def vqvae_setup(args):
    model_vqvae = FlatVQVAE().to(DEVICE)
    model_vqvae.load_state_dict(torch.load(args.ckpt_vqvae, map_location=DEVICE))
    model_vqvae = model_vqvae.to(DEVICE)
    model_vqvae.eval()
    return model_vqvae


def load_data():
    quantizes, quant_b = load_embedding_space()
    indices = load_indices()
    labels = load_labels()
    return torch.from_numpy(labels), torch.from_numpy(indices), torch.from_numpy(quantizes), torch.from_numpy(quant_b)
    # return (torch.from_numpy(labels)[:20], torch.from_numpy(indices)[:20],
    #         torch.from_numpy(quantizes)[:20], torch.from_numpy(quant_b)[:20])


def load_models_to_device(n_token, d_embed_vec):
    model_distil = transformer_setup(args, n_token, 456, d_embed_vec)
    model_vqvae = vqvae_setup(args)
    return model_distil, model_vqvae

def batch_to_device(tensors, start_index, batch_size):
    batched_tensors = []
    end_index = min(start_index + batch_size, tensors[0].shape[0])
    for tensor in tensors:
        batched_tensors.append(tensor[start_index:end_index].to(DEVICE))
    return tuple(batched_tensors)

def get_imgs_from_ids(tensors, imgs_ids):
    batched_tensors = []
    for tensor in tensors:
        batched_tensors.append(tensor[[imgs_ids]].to(DEVICE))
    return tuple(batched_tensors)

def denormalize(img):
    img = img * torch.tensor([0.5, 0.5, 0.5], device=img.device).view(-1, 1, 1)
    img = img + torch.tensor([0.5, 0.5, 0.5], device=img.device).view(-1, 1, 1)
    return img



@torch.no_grad()
def additive_attn_eval(step_size=1, batch_size=2000, file_name='additive_attn_recon_errors.pt'):
    class_labels, index_repr, q, q_2d = load_data()
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    recon_errors_over_batches = []
    for start_index in range(0, n_samples, batch_size):
        q_batch, index_repr_batch, q_2d_batch = batch_to_device((q, index_repr, q_2d), start_index, batch_size)
        q_masked, _, _ = full_mask(q_batch, index_repr_batch)
        vqvae_out = model_vqvae.decode(q_2d_batch)
        pos_to_unmask = torch.empty((q_batch.shape[0], 0), dtype=torch.int64).to(DEVICE)
        recon_errors_over_tokens = []
        rows = torch.arange(q_batch.size(0)).unsqueeze(1)

        for i in range(0, q_batch.shape[1], step_size):
            q_masked[rows, pos_to_unmask] = q_batch[rows, pos_to_unmask]
            logits = model_distil(inputs_embeds=q_masked, output_hidden_states=True).logits
            max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
            max_index_per_pos[rows, pos_to_unmask] = index_repr_batch[rows, pos_to_unmask]
            recons_from_max_indices = model_vqvae.decode_code(max_index_per_pos.reshape(-1, length, length).to(DEVICE))
            sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1)
            pos_to_unmask = conc_unique_elements(pos_to_unmask, sorted_max_conf_per_pos, n_elements=step_size)
            recon_loss = F.mse_loss(recons_from_max_indices, vqvae_out, reduction='none')
            recon_errors_over_tokens.insert(0, torch.mean(recon_loss, dim=(1, 2, 3)).tolist())
        recon_errors_over_batches.append(recon_errors_over_tokens)
    recon_errors = torch.tensor(recon_errors_over_batches).permute(1, 0, 2).flatten(1)
    mean_recon_errors = torch.mean(recon_errors, dim=1)
    torch.save(mean_recon_errors, DATA_DIR / 'recon_errors' / file_name)


@torch.no_grad()
def random_attn_eval(tokens_to_add=1, batch_size=2000, file_name='random_attn_recon_errors.pt'):
    class_labels, index_repr, q, q_2d = load_data()
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    recon_errors_over_batches = []
    for start_index in range(0, n_samples, batch_size):
        q_batch, index_repr_batch, q_2d_batch = batch_to_device((q, index_repr, q_2d), start_index, batch_size)
        q_masked = q_batch.clone()
        index_masked = index_repr_batch.clone()
        vqvae_out = model_vqvae.decode(q_2d_batch)
        rnd_mask = torch.stack([torch.randperm(q_batch.shape[1]) for _ in range(q_batch.shape[0])]).to(DEVICE)
        recon_errors_over_tokens = []
        rows = torch.arange(q_batch.size(0)).unsqueeze(1)
        for i in range(0, q.shape[1] // tokens_to_add + 1):
            pos_to_mask = rnd_mask[:, :i]
            q_masked[rows, pos_to_mask, :] = 0
            logits = model_distil(inputs_embeds=q_masked, output_hidden_states=True).logits
            max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
            index_masked[rows, pos_to_mask] = max_index_per_pos[rows, pos_to_mask]
            recons_from_max_indices = model_vqvae.decode_code(index_masked.reshape(-1, length, length).to(DEVICE))
            recon_loss = F.mse_loss(recons_from_max_indices, vqvae_out, reduction='none')
            recon_errors_over_tokens.append(torch.mean(recon_loss, dim=(1, 2, 3)).tolist())
        recon_errors_over_batches.append(recon_errors_over_tokens)
    recon_errors = torch.tensor(recon_errors_over_batches).permute(1, 0, 2).flatten(1)
    mean_recon_errors = torch.mean(recon_errors, dim=1)
    torch.save(mean_recon_errors, DATA_DIR / 'recon_errors' / file_name)


@torch.no_grad()
def selective_iterative_attn_eval(step_size=1, batch_size=2000, file_name='selective_attn_recon_errors.pt'):
    class_labels, index_repr, q, q_2d = load_data()
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    recon_errors_over_batches = []
    for start_index in range(0, n_samples, batch_size):
        q_b, index_b, q_2d_b = batch_to_device((q, index_repr, q_2d), start_index, batch_size)
        vqvae_out = model_vqvae.decode(q_2d_b)
        masked_pos = torch.empty((q_b.shape[0], 0), dtype=torch.int64).to(DEVICE)
        recon_errors_over_tokens = []
        rows = torch.arange(index_b.size(0)).unsqueeze(1)

        for i in range(0, q_b.shape[1] + 1, step_size):
            # mask q at most confident positions
            q_b[rows, masked_pos] = 0
            # get most likely indices for masked positions
            logits = model_distil(inputs_embeds=q_b, output_hidden_states=True).logits
            max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
            # replace indices at masked positions with most likely indices
            index_b[rows, masked_pos] = max_index_per_pos[rows, masked_pos]
            # compute mse
            recons_from_max_indices = model_vqvae.decode_code(index_b.reshape(-1, length, length).to(DEVICE))  # bx20x20 -> bx3x80x80
            recon_loss = F.mse_loss(recons_from_max_indices, vqvae_out, reduction='none')
            recon_errors_over_tokens.append(torch.mean(recon_loss, dim=(1, 2, 3)).tolist())
            # Get new mask with the new pos the model is most confident about
            sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1, descending=True)
            masked_pos = conc_unique_elements(masked_pos, sorted_max_conf_per_pos, n_elements=step_size)
        recon_errors_over_batches.append(recon_errors_over_tokens)
    recon_errors = torch.tensor(recon_errors_over_batches).permute(1, 0, 2).flatten(1)
    mean_recon_errors = torch.mean(recon_errors, dim=1)
    torch.save(mean_recon_errors, DATA_DIR / 'recon_errors' / file_name)

@torch.no_grad()
def selective_iterative_attn_plot(step_size=1, batch_size=2000,  file_name='recons_sel_iter_attn', img_ids=None):
    class_labels, index_repr, q, q_2d = load_data()
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    if img_ids:
        q_b, index_b, q_2d_b = get_imgs_from_ids((q, index_repr, q_2d), imgs_ids=img_ids)
    else:
        q_b, index_b, q_2d_b = batch_to_device((q, index_repr, q_2d), 0, batch_size)
    mask = torch.ones(size=(q_b.shape[0], q_b.shape[1]))
    rows = torch.arange(index_b.size(0)).unsqueeze(1)

    masked_pos = torch.empty((q_b.shape[0], 0), dtype=torch.int64).to(DEVICE)

    logits_from_unmasked_img = model_distil(inputs_embeds=q_b, output_hidden_states=True).logits
    max_conf_unmasked_img, max_index_unmasked_img = torch.max(logits_from_unmasked_img, dim=2)
    masks, perc, recons, confidences, normed_confidences = [], [], [], [], []

    for i in range(0, q_b.shape[1] + 1, step_size):
        # mask q at most confident positions
        q_b[rows, masked_pos] = 0
        # get most likely indices for masked positions
        logits = model_distil(inputs_embeds=q_b, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
        # replace indices at masked positions with most likely indices
        index_b[rows, masked_pos] = max_index_per_pos[rows, masked_pos]

        recons_from_max_indices = model_vqvae.decode_code(index_b.reshape(-1, length, length).to(DEVICE))  # bx20x20 -> bx3x80x80
        masks.append(mask.clone())
        recons.append(recons_from_max_indices)
        confidences.append(max_conf_per_pos)
        normed_confidences.append(torch.max(torch.softmax(logits, dim=2), dim=2)[0])
        perc.append(i / 400)

        sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1, descending=True)
        masked_pos = conc_unique_elements(masked_pos, sorted_max_conf_per_pos, n_elements=step_size)
    plot_recons(recons, masks, perc, confidences, normed_confidences, file_name)



@torch.no_grad()
def selective_direct_attn_eval(batch_size=2000, file_name='selective_attn_direct_recon_errors.pt', step_size=1, reverse=False):
    class_labels, index_repr, q, q_2d = load_data()
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    recon_errors_over_batches = []
    for start_index in range(0, n_samples, batch_size):
        q_b, index_b, q_2d_b = batch_to_device((q, index_repr, q_2d), start_index, batch_size)
        vqvae_out = model_vqvae.decode(q_2d_b)
        logits_from_unmasked_img = model_distil(inputs_embeds=q_b, output_hidden_states=True).logits
        max_conf_unmasked_img, max_index_unmasked_img = torch.max(logits_from_unmasked_img, dim=2)

        arg_sort_max_conf_unmasked_img = torch.argsort(max_conf_unmasked_img, dim=1, descending=not reverse)
        recon_errors_over_tokens = []
        rows = torch.arange(index_b.size(0)).unsqueeze(1)

        for i in range(0, q_b.shape[1] + 1, step_size):
            masked_pos = arg_sort_max_conf_unmasked_img[:, :i]
            q_b[rows, masked_pos] = 0
            logits_from_masked_img = model_distil(inputs_embeds=q_b, output_hidden_states=True).logits
            max_conf_masked_img, max_index_masked_img = torch.max(logits_from_masked_img, dim=2)
            index_b[rows, masked_pos] = max_index_masked_img[rows, masked_pos]
            recons_from_max_indices = model_vqvae.decode_code(
                index_b.reshape(-1, length, length).to(DEVICE))  # bx20x20 -> bx3x80x80
            # replace indices at masked positions with most likely indices
            recon_loss = F.mse_loss(recons_from_max_indices, vqvae_out, reduction='none')
            recon_errors_over_tokens.append(torch.mean(recon_loss, dim=(1, 2, 3)).tolist())
            # get most likely indices for masked positions
            # replace indices at masked positions with most likely indices
            # Get new mask with the new pos the model is most confident about
            # sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1, descending=True)
        recon_errors_over_batches.append(recon_errors_over_tokens)
    recon_errors = torch.tensor(recon_errors_over_batches).permute(1, 0, 2).flatten(1)
    mean_recon_errors = torch.mean(recon_errors, dim=1)
    if reverse:
        torch.save(mean_recon_errors, DATA_DIR / 'recon_errors' / file_name)
    else:
        torch.save(mean_recon_errors, DATA_DIR / 'recon_errors' / 'selective_attn_direct_inverse_recon_errors.pt')

@torch.no_grad()
def selective_direct_plot(batch_size=10, step_size=1, file_name='recons_sel_dir_attn', img_ids=None):
    class_labels, index_repr, q, q_2d = load_data()
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    if img_ids:
        q_b, index_b, q_2d_b = get_imgs_from_ids((q, index_repr, q_2d), imgs_ids=img_ids)
    else:
        q_b, index_b, q_2d_b = batch_to_device((q, index_repr, q_2d), 0, batch_size)
    mask = torch.ones(size=(q_b.shape[0], q_b.shape[1]))
    rows = torch.arange(index_b.size(0)).unsqueeze(1)

    logits_from_unmasked_img = model_distil(inputs_embeds=q_b, output_hidden_states=True).logits
    max_conf_unmasked_img, max_index_unmasked_img = torch.max(logits_from_unmasked_img, dim=2)
    arg_sort_max_conf_unmasked_img = torch.argsort(max_conf_unmasked_img, dim=1, descending=True)

    masks, perc, recons, confidences, normed_confidences = [], [], [], [], []

    for i in range(0, q_b.shape[1] + 1, step_size):
        masked_pos = arg_sort_max_conf_unmasked_img[:, :i]
        mask[rows, masked_pos] = 0
        q_b[rows, masked_pos] = 0

        logits_from_masked_img = model_distil(inputs_embeds=q_b, output_hidden_states=True).logits
        max_conf_masked_img, max_index_masked_img = torch.max(logits_from_masked_img, dim=2)
        index_b[rows, masked_pos] = max_index_masked_img[rows, masked_pos]
        recons_from_max_indices = model_vqvae.decode_code(index_b.reshape(-1, length, length).to(DEVICE))  # bx20x20 -> bx3x80x80

        masks.append(mask.clone())
        recons.append(recons_from_max_indices)
        confidences.append(max_conf_masked_img)
        normed_confidences.append(torch.max(torch.softmax(logits_from_masked_img, dim=2), dim=2)[0])
        perc.append(i/400)

    plot_recons(recons, masks, perc, confidences, normed_confidences, file_name)


@torch.no_grad()
def additive_plot(batch_size=10, file_name='recons_add_attn', step_size=1, img_ids=None):
    class_labels, index_repr, q, q_2d = load_data()
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    if img_ids:
        q_batch, index_repr_batch, q_2d_batch = get_imgs_from_ids((q, index_repr, q_2d), imgs_ids=img_ids)
    else:
        q_batch, index_repr_batch, q_2d_batch = batch_to_device((q, index_repr, q_2d), 0, batch_size)
    q_masked, _, _ = full_mask(q_batch, index_repr_batch)
    pos_to_unmask = torch.empty((q_batch.shape[0], 0), dtype=torch.int64).to(DEVICE)
    rows = torch.arange(q_batch.size(0)).unsqueeze(1)
    mask = torch.zeros(size=(q_batch.shape[0], q_batch.shape[1]))
    masks, recons, confidences, normed_confidences, mask_perc = [], [], [], [], []

    for i in range(0, q_batch.shape[1] + 1, step_size):
        q_masked[rows, pos_to_unmask] = q_batch[rows, pos_to_unmask]
        logits = model_distil(inputs_embeds=q_masked, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
        max_index_per_pos[rows, pos_to_unmask] = index_repr_batch[rows, pos_to_unmask]
        recons_from_max_indices = model_vqvae.decode_code(max_index_per_pos.reshape(-1, length, length).to(DEVICE))
        sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1)
        pos_to_unmask = conc_unique_elements(pos_to_unmask, sorted_max_conf_per_pos, n_elements=step_size)

        recons.append(recons_from_max_indices)
        mask[rows, pos_to_unmask[:, :i]] = 1
        masks.append(mask.clone())
        mask_perc.append((400 - i)/400)
        confidences.append(max_conf_per_pos)
        normed_confidences.append(torch.max(torch.softmax(logits, dim=2), dim=2)[0])

    plot_recons(recons, masks, mask_perc, confidences, normed_confidences, file_name)


@torch.no_grad()
def selective_attn_weird_implementation_plot(batch_size=10, file_name='recons_selective_attn_weird_implementation', img_ids=None):
    class_labels, index_repr, q, q_2d = load_data()
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    if img_ids:
        q_batch, index_repr_batch, q_2d_batch = get_imgs_from_ids((q, index_repr, q_2d), imgs_ids=img_ids)
    else:
        q_batch, index_repr_batch, q_2d_batch = batch_to_device((q, index_repr, q_2d), 0, batch_size)
    q_masked, _, _ = full_mask(q_batch, index_repr_batch)
    pos_to_unmask = torch.empty((q_batch.shape[0], 0), dtype=torch.int64).to(DEVICE)
    rows = torch.arange(q_batch.size(0)).unsqueeze(1)
    mask = torch.zeros(size=(q_batch.shape[0], q_batch.shape[1]))
    masks, recons, confidences, normed_confidences, mask_perc = [], [], [], [], []

    for i in range(0, q_batch.shape[1] + 1):
        q_masked[rows, pos_to_unmask] = q_batch[rows, pos_to_unmask]
        logits = model_distil(inputs_embeds=q_masked, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
        max_index_per_pos[rows, pos_to_unmask] = index_repr_batch[rows, pos_to_unmask]
        recons_from_max_indices = model_vqvae.decode_code(max_index_per_pos.reshape(-1, length, length).to(DEVICE))
        sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1)
        if i == 0:
            pos_to_unmask = torch.full(size=(batch_size, 1), fill_value=200)
        else:
            pos_to_unmask = conc_unique_elements(pos_to_unmask, sorted_max_conf_per_pos, n_elements=1)
        recons.append(recons_from_max_indices)
        mask[rows, pos_to_unmask[:, :i]] = 1
        masks.append(mask.clone())
        mask_perc.append((400 - i)/400)
        confidences.append(max_conf_per_pos)
        normed_confidences.append(torch.max(torch.softmax(logits, dim=2), dim=2)[0])
        if i == 100:
            break

    plot_recons(recons, masks, mask_perc, confidences, normed_confidences, file_name)


def plot_recons(recons, masks, perc, confidences, normed_confidences, file_name):
    n_rows, n_cols = len(masks) + 1, recons[0].shape[0] * 4
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))
    for j in range(0, recons[0].shape[0]):
        for k in range(len(masks)):
            axs[k][4 * j].set_ylabel(perc[k])
            axs[k][4 * j].imshow(denormalize(recons[k])[j].permute(1, 2, 0))
            axs[k][4 * j + 1].imshow(masks[k].reshape(-1, 20, 20, 1)[j], vmin=0, vmax=1)
            axs[k][4 * j + 2].imshow(confidences[k].reshape(-1, 1, 20, 20)[j].permute(1, 2, 0).numpy())
            axs[k][4 * j + 3].imshow(normed_confidences[k].reshape(-1, 1, 20, 20)[j].permute(1, 2, 0).numpy(),
                                     vmin=0, vmax=1)
    shift_rows_and_columns(axs, cols_to_shift=np.arange(0, n_cols, 4))
    hide_all_extras(axs)
    fig.savefig(file_name + '.pdf', format='pdf')

@torch.no_grad()
def random_plot(batch_size=2000, step_size=1, img_ids=None, file_name='recons_rnd_attn'):
    class_labels, index_repr, q, q_2d = load_data()
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    if img_ids:
        q_batch, index_repr_batch, q_2d_batch = get_imgs_from_ids((q, index_repr, q_2d), imgs_ids=img_ids)
    else:
        q_batch, index_repr_batch, q_2d_batch = batch_to_device((q, index_repr, q_2d), 0, batch_size)
    q_masked = q_batch.clone()

    index_masked = index_repr_batch.clone()
    rnd_mask = torch.stack([torch.randperm(q_batch.shape[1]) for _ in range(q_batch.shape[0])]).to(DEVICE)
    mask = torch.ones(size=(q_batch.shape[0], q_batch.shape[1]))
    masks, recons, confidences, normed_confidences, perc = [], [], [], [], []
    rows = torch.arange(q_batch.size(0)).unsqueeze(1)

    for i in range(0, q_batch.shape[1] + 1, step_size):
        pos_to_mask = rnd_mask[:, :i]
        q_masked[rows, pos_to_mask]  = 0
        logits = model_distil(inputs_embeds=q_masked, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)

        mask[rows, pos_to_mask[:, :i]] = 0


        index_masked[rows, pos_to_mask] = max_index_per_pos[rows, pos_to_mask]

        masks.append(mask.clone()),
        perc.append(i / 400)
        confidences.append(max_conf_per_pos)
        normed_confidences.append(torch.max(torch.softmax(logits, dim=2), dim=2)[0])
        recons.append(model_vqvae.decode_code(index_masked.reshape(-1, length, length)))

    plot_recons(recons, masks, perc, confidences, normed_confidences, file_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14)
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--ckpt_vqvae', type=str, default=VQVAE_PATH)
    parser.add_argument('--ckpt_distil_combined', type=str, default=EIGHTY_EIGHTY_PATH)

    # plt.plot(torch.load(DATA_DIR / 'recon_errors' / 'selective_attn_recon_errors.pt'), label='iter_selective')
    plt.plot(torch.load(DATA_DIR / 'recon_errors' / 'random_attn_recon_errors.pt'), label='random')
    plt.plot(torch.load(DATA_DIR / 'recon_errors' / 'additive_attn_recon_errors.pt'), label='additive')
    plt.plot(torch.load(DATA_DIR / 'recon_errors' / 'selective_attn_direct_recon_errors.pt'), label='direct_selective')
    # plt.plot(torch.load(DATA_DIR / 'recon_errors' / 'selective_attn_direct_inverse_recon_errors.pt'), label='direct_inv_selective')
    plt.legend()
    plt.show()
    # #
    #
    args = parser.parse_args()
    #
    # additive_plot(batch_size=10, step_size=5, img_ids=(462, 1671,  1836, 4970, 5852, 7777, 8513, 8685, 9469, 9644))
    # random_plot(batch_size=10, step_size=10, img_ids=(462, 1671,  1836, 4970, 5852, 7777, 8513, 8685, 9469, 9644))

    #
    # selective_direct_attn_eval(batch_size=500, step_size=1)
    # selective_direct_attn_eval(batch_size=100, step_size=1, reverse=True)
    selective_attn_weird_implementation_plot(batch_size=10, img_ids=(462, 1671,  1836, 4970, 5852, 7777, 8513, 8685, 9469, 9644))


    # selective_direct_plot(batch_size=10, step_size=10, img_ids=(462, 1671,  1836, 4970, 5852, 7777, 8513, 8685, 9469, 9644))
    # selective_iterative_attn_plot(batch_size=10, step_size=10, img_ids=(462, 1671,  1836, 4970, 5852, 7777, 8513, 8685, 9469, 9644))

    # random_direct_attn_eval(batch_size=20)

    # selective_direct_inverse_attn_eval(batch_size=500)
    # random_attn_eval()
    # additive_attn_eval(batch_size=500)
    # selective_iterative_attn_eval(batch_size=500)
    # za
    # main(args)
    # dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
