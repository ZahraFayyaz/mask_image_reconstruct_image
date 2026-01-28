import numpy as np
import argparse, math, sys, os
import torch
from torch import nn
import distributed as dist
from transformers import DistilBertForMaskedLM, DistilBertConfig

from mask_reconstruct_img import DATA_DIR
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

if HOSTNAME == 'add here':
    QUANTIZED_EPOCH_PATH = '/home/abghamtm/work/masking_comparison/checkpoint/vqvae/quantized_epoch80_flat_vqvae80x80_144x456codebook.npy'
    INDICES_PATH = '/home/abghamtm/work/masking_comparison/checkpoint/vqvae/indices_epoch80_flat_vqvae80x80_144x456codebook.npy'
    LABELS_PATH = '/home/abghamtm/work/masking_comparison/checkpoint/vqvae/labels_epoch80_flat_vqvae80x80_144x456codebook.npy'
    EIGHTY_EIGHTY_PATH = '/home/abghamtm/work/masking_comparison/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100.pt'
    VQVAE_PATH = '/home/abghamtm/work/masking_comparison/checkpoint/vqvae/model_epoch80_flat_vqvae80x80_144x456codebook.pth'
    CLASSIFIER_WEIGHTS = '/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/weights_epoch30.pth'
else:
    QUANTIZED_EPOCH_PATH = DATA_DIR / 'vqvae' / 'quantized_epoch80_flat_vqvae80x80_144x456codebook.npy'
    INDICES_PATH = DATA_DIR / 'vqvae' / 'indices_epoch80_flat_vqvae80x80_144x456codebook.npy'
    LABELS_PATH = DATA_DIR / 'vqvae' / 'labels_epoch80_flat_vqvae80x80_144x456codebook.npy'
    EIGHTY_EIGHTY_PATH = DATA_DIR / 'vqvae' / '80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100.pt'
    VQVAE_PATH = DATA_DIR / 'vqvae' / 'model_epoch80_flat_vqvae80x80_144x456codebook.pth'
    CLASSIFIER_WEIGHTS = DATA_DIR / 'classifier' / 'weights_epoch30.pth'


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


def accuracy(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    e = pred.eq(target.view_as(pred)).sum() / target.shape[0]
    return e


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


def classifier_setup():
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    classifier = resnet50(pretrained=False)
    classifier.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=torch.device(DEVICE)))
    classifier.to(DEVICE)
    classifier.eval()
    return classifier, preprocess


def load_data(train_data=False):
    quantizes, quant_b = load_embedding_space()
    indices = load_indices()
    labels = load_labels()
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    set_indices = train_indices if train_data else test_indices
    return torch.from_numpy(labels)[set_indices], torch.from_numpy(indices)[set_indices], torch.from_numpy(quantizes)[set_indices], torch.from_numpy(quant_b)[set_indices]
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
def additive_attn_eval(step_size=1, batch_size=2000, file_name_ce='additive_classifier_ce.pt',
                       file_name_acc='additive_classifier_acc.pt', init_unmask_pos=None, train_data=False):
    class_labels, index_repr, q, q_2d = load_data(train_data)
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    classifier, preprocess = classifier_setup()
    ce_over_batches, acc_over_batches = [], []
    n_inputs = 0
    for start_index in range(0, n_samples, batch_size):
        q_batch, index_repr_batch, q_2d_batch, class_labels_batch = batch_to_device((q, index_repr, q_2d, class_labels),
                                                                                    start_index, batch_size)
        q_masked, _, _ = full_mask(q_batch, index_repr_batch)
        pos_to_unmask = torch.empty((q_batch.shape[0], 0), dtype=torch.int64).to(DEVICE)
        ce_over_tokens, acc_over_tokens = [], []
        rows = torch.arange(q_batch.size(0)).unsqueeze(1)
        n_inputs += q_batch.shape[0]
        for i in range(0, q_batch.shape[1], step_size):
            q_masked[rows, pos_to_unmask] = q_batch[rows, pos_to_unmask]
            logits = model_distil(inputs_embeds=q_masked, output_hidden_states=True).logits
            max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
            max_index_per_pos[rows, pos_to_unmask] = index_repr_batch[rows, pos_to_unmask]
            recons_from_max_indices = model_vqvae.decode_code(max_index_per_pos.reshape(-1, length, length).to(DEVICE))
            sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1)
            if i == 0 and init_unmask_pos is not None:
                pos_to_unmask = torch.full(size=(batch_size, 1), fill_value=init_unmask_pos).to(DEVICE)
            else:
                pos_to_unmask = conc_unique_elements(pos_to_unmask, sorted_max_conf_per_pos, n_elements=step_size)
            preprocessed_image = preprocess(recons_from_max_indices)
            logits = classifier(preprocessed_image)
            ce_over_tokens.append(F.cross_entropy(input=logits, target=class_labels_batch).item())
            acc_over_tokens.append(accuracy(logits=logits, target=class_labels_batch).item())

            # recon_loss = F.mse_loss(recons_from_max_indices, vqvae_out, reduction='none')
            # recon_errors_over_tokens.insert(0, torch.mean(recon_loss, dim=(1, 2, 3)).tolist())
        ce_over_batches.append((torch.tensor(ce_over_tokens) * q_batch.shape[0]).tolist())
        acc_over_batches.append((torch.tensor(acc_over_tokens) * q_batch.shape[0]).tolist())
    ces = torch.sum(torch.tensor(ce_over_batches), dim=0) / n_inputs
    accs = torch.sum(torch.tensor(acc_over_batches), dim=0) / n_inputs
    torch.save(accs, DATA_DIR / 'class_errors' / file_name_acc)
    torch.save(ces, DATA_DIR / 'class_errors' / file_name_ce)


@torch.no_grad()
def random_attn_eval(tokens_to_add=1, batch_size=2000, file_name_ce='random_classifier_ce.pt',
                     file_name_acc='random_classifier_acc.pt', train_data=False):
    class_labels, index_repr, q, q_2d = load_data(train_data)
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    classifier, preprocess = classifier_setup()
    ce_over_batches, acc_over_batches = [], []
    n_inputs = 0
    for start_index in range(0, n_samples, batch_size):
        q_batch, index_repr_batch, q_2d_batch, class_labels_batch = batch_to_device((q, index_repr, q_2d, class_labels),
                                                                                    start_index, batch_size)
        q_masked = q_batch.clone()
        index_masked = index_repr_batch.clone()
        rnd_mask = torch.stack([torch.randperm(q_batch.shape[1]) for _ in range(q_batch.shape[0])]).to(DEVICE)
        ce_over_tokens, acc_over_tokens = [], []
        rows = torch.arange(q_batch.size(0)).unsqueeze(1)
        n_inputs += q_batch.shape[0]
        for i in range(0, q.shape[1] // tokens_to_add + 1):
            pos_to_mask = rnd_mask[:, :i]
            q_masked[rows, pos_to_mask, :] = 0
            logits = model_distil(inputs_embeds=q_masked, output_hidden_states=True).logits
            max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
            index_masked[rows, pos_to_mask] = max_index_per_pos[rows, pos_to_mask]
            recons_from_max_indices = model_vqvae.decode_code(index_masked.reshape(-1, length, length).to(DEVICE))
            preprocessed_image = preprocess(recons_from_max_indices)
            logits = classifier(preprocessed_image)
            ce_over_tokens.append(F.cross_entropy(input=logits, target=class_labels_batch).item())
            acc_over_tokens.append(accuracy(logits=logits, target=class_labels_batch).item())
        ce_over_batches.append((torch.tensor(ce_over_tokens) * q_batch.shape[0]).tolist())
        acc_over_batches.append((torch.tensor(acc_over_tokens) * q_batch.shape[0]).tolist())
    ces = torch.sum(torch.tensor(ce_over_batches), dim=0) / n_inputs
    accs = torch.sum(torch.tensor(acc_over_batches), dim=0) / n_inputs
    torch.save(accs, DATA_DIR / 'class_errors' / file_name_acc)
    torch.save(ces, DATA_DIR / 'class_errors' / file_name_ce)


@torch.no_grad()
def selective_iterative_attn_eval(step_size=1, batch_size=2000, file_name_ce='selective_iterative_classifier_ce.pt',
                                  file_name_acc='selective_iterative_classifier_acc.pt', train_data=False):
    class_labels, index_repr, q, q_2d = load_data(train_data)
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    classifier, preprocess = classifier_setup()
    ce_over_batches, acc_over_batches = [], []
    n_inputs = 0
    for start_index in range(0, n_samples, batch_size):
        q_b, index_b, q_2d_b, class_labels_batch = batch_to_device((q, index_repr, q_2d, class_labels), start_index,
                                                                   batch_size)
        masked_pos = torch.empty((q_b.shape[0], 0), dtype=torch.int64).to(DEVICE)
        ce_over_tokens, acc_over_tokens = [], []
        rows = torch.arange(index_b.size(0)).unsqueeze(1)
        n_inputs += q_b.shape[0]

        for i in range(0, q_b.shape[1] + 1, step_size):
            # mask q at most confident positions
            q_b[rows, masked_pos] = 0
            # get most likely indices for masked positions
            logits = model_distil(inputs_embeds=q_b, output_hidden_states=True).logits
            max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
            # replace indices at masked positions with most likely indices
            index_b[rows, masked_pos] = max_index_per_pos[rows, masked_pos]
            # compute mse
            recons_from_max_indices = model_vqvae.decode_code(
                index_b.reshape(-1, length, length).to(DEVICE))  # bx20x20 -> bx3x80x80
            preprocessed_image = preprocess(recons_from_max_indices)
            logits = classifier(preprocessed_image)
            ce_over_tokens.append(F.cross_entropy(input=logits, target=class_labels_batch).item())
            acc_over_tokens.append(accuracy(logits=logits, target=class_labels_batch).item())
            # Get new mask with the new pos the model is most confident about
            sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1, descending=True)
            masked_pos = conc_unique_elements(masked_pos, sorted_max_conf_per_pos, n_elements=step_size)
        ce_over_batches.append((torch.tensor(ce_over_tokens) * q_b.shape[0]).tolist())
        acc_over_batches.append((torch.tensor(acc_over_tokens) * q_b.shape[0]).tolist())
    ces = torch.sum(torch.tensor(ce_over_batches), dim=0) / n_inputs
    accs = torch.sum(torch.tensor(acc_over_batches), dim=0) / n_inputs
    torch.save(accs, DATA_DIR / 'class_errors' / file_name_acc)
    torch.save(ces, DATA_DIR / 'class_errors' / file_name_ce)


@torch.no_grad()
def selective_direct_attn_eval(batch_size=2000, file_name_ce='selective_direct_classifier_ce.pt',
                               file_name_acc='selective_direct_classifier_acc.pt', step_size=1, reverse=False, train_data=False):
    class_labels, index_repr, q, q_2d = load_data(train_data)
    n_samples, n_token, d_embed_vec, length = q.shape[0], index_repr.shape[1], q.shape[-1], q_2d.shape[-1]
    model_distil, model_vqvae = load_models_to_device(n_token=n_token, d_embed_vec=d_embed_vec)
    classifier, preprocess = classifier_setup()
    ce_over_batches, acc_over_batches = [], []
    n_inputs = 0
    for start_index in range(0, n_samples, batch_size):
        q_b, index_b, q_2d_b, class_labels_batch = batch_to_device((q, index_repr, q_2d, class_labels), start_index,
                                                                   batch_size)
        logits_from_unmasked_img = model_distil(inputs_embeds=q_b, output_hidden_states=True).logits
        max_conf_unmasked_img, max_index_unmasked_img = torch.max(logits_from_unmasked_img, dim=2)

        arg_sort_max_conf_unmasked_img = torch.argsort(max_conf_unmasked_img, dim=1, descending=not reverse)
        rows = torch.arange(index_b.size(0)).unsqueeze(1)
        n_inputs += q_b.shape[0]
        ce_over_tokens, acc_over_tokens = [], []
        for i in range(0, q_b.shape[1] + 1, step_size):
            masked_pos = arg_sort_max_conf_unmasked_img[:, :i]
            q_b[rows, masked_pos] = 0
            logits_from_masked_img = model_distil(inputs_embeds=q_b, output_hidden_states=True).logits
            max_conf_masked_img, max_index_masked_img = torch.max(logits_from_masked_img, dim=2)
            index_b[rows, masked_pos] = max_index_masked_img[rows, masked_pos]
            recons_from_max_indices = model_vqvae.decode_code(
                index_b.reshape(-1, length, length).to(DEVICE))  # bx20x20 -> bx3x80x80
            # replace indices at masked positions with most likely indices
            preprocessed_image = preprocess(recons_from_max_indices)
            logits = classifier(preprocessed_image)
            ce_over_tokens.append(F.cross_entropy(input=logits, target=class_labels_batch).item())
            acc_over_tokens.append(accuracy(logits=logits, target=class_labels_batch).item())

        ce_over_batches.append((torch.tensor(ce_over_tokens) * q_b.shape[0]).tolist())
        acc_over_batches.append((torch.tensor(acc_over_tokens) * q_b.shape[0]).tolist())
    ces = torch.sum(torch.tensor(ce_over_batches), dim=0) / n_inputs
    accs = torch.sum(torch.tensor(acc_over_batches), dim=0) / n_inputs
    torch.save(accs, DATA_DIR / 'class_errors' / file_name_acc)
    torch.save(ces, DATA_DIR / 'class_errors' / file_name_ce)


@torch.no_grad()
def random_plot(batch_size=2000, step_size=1, img_ids=None, file_name='recons_rnd_attn', plot_every=40):
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
        q_masked[rows, pos_to_mask] = 0
        logits = model_distil(inputs_embeds=q_masked, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)

        mask[rows, pos_to_mask[:, :i]] = 0
        index_masked[rows, pos_to_mask] = max_index_per_pos[rows, pos_to_mask]
        if i % plot_every == 0:
            recons.append(model_vqvae.decode_code(index_masked.reshape(-1, length, length)))
            masks.append(mask.clone())
            perc.append(i / 400)
            confidences.append(max_conf_per_pos)
            normed_confidences.append(torch.max(torch.softmax(logits, dim=2), dim=2)[0])

    plot_recons(recons, masks, perc, confidences, normed_confidences, file_name)


@torch.no_grad()
def selective_iterative_attn_plot(step_size=1, batch_size=2000, file_name='recons_sel_iter_attn', img_ids=None):
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

        recons_from_max_indices = model_vqvae.decode_code(
            index_b.reshape(-1, length, length).to(DEVICE))  # bx20x20 -> bx3x80x80
        masks.append(mask.clone())
        recons.append(recons_from_max_indices)
        confidences.append(max_conf_per_pos)
        normed_confidences.append(torch.max(torch.softmax(logits, dim=2), dim=2)[0])
        perc.append(i / 400)

        sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1, descending=True)
        masked_pos = conc_unique_elements(masked_pos, sorted_max_conf_per_pos, n_elements=step_size)
    plot_recons(recons, masks, perc, confidences, normed_confidences, file_name)


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
        recons_from_max_indices = model_vqvae.decode_code(
            index_b.reshape(-1, length, length).to(DEVICE))  # bx20x20 -> bx3x80x80

        masks.append(mask.clone())
        recons.append(recons_from_max_indices)
        confidences.append(max_conf_masked_img)
        normed_confidences.append(torch.max(torch.softmax(logits_from_masked_img, dim=2), dim=2)[0])
        perc.append(i / 400)

    plot_recons(recons, masks, perc, confidences, normed_confidences, file_name)


@torch.no_grad()
def additive_plot(batch_size=10, file_name='recons_add_attn', step_size=1, img_ids=None, init_unmask_pos=None,
                  plot_every=40):
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
        if i == 0 and init_unmask_pos is not None:
            pos_to_unmask = torch.full(size=(batch_size, 1), fill_value=init_unmask_pos).to(DEVICE)
        else:
            pos_to_unmask = conc_unique_elements(pos_to_unmask, sorted_max_conf_per_pos, n_elements=1)
        if i % plot_every == 0:
            recons.append(recons_from_max_indices)
            mask[rows, pos_to_unmask[:, :i]] = 1
            masks.append(mask.clone())
            mask_perc.append((400 - i) / 400)
            confidences.append(max_conf_per_pos)
            normed_confidences.append(torch.max(torch.softmax(logits, dim=2), dim=2)[0])

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
    fig.savefig(DATA_DIR / 'plots' / (file_name + '.pdf'), format='pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)
    port = (2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14)
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--ckpt_vqvae', type=str, default=VQVAE_PATH)
    parser.add_argument('--cuda', '-c', type=int, default=0)

    parser.add_argument('--ckpt_distil_combined', type=str, default=EIGHTY_EIGHTY_PATH)
    args = parser.parse_args()

    if torch.cuda.is_available():
        DEVICE = 'cuda'
        torch.cuda.set_device(args.cuda)
        torch.cuda.empty_cache()
    else:
        DEVICE = 'cpu'
    pass

    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'additive_attn_acc.pt'), label='additive_attn')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'selective_direct_classifier_acc.pt').flip(0),
    #          label='selective_direct')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'selective_iterative_classifier_acc.pt').flip(0),
    #          label='selective_iterative')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'random_classifier_acc.pt').flip(0), label='random_attn')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'selective_strange_attn_acc.pt').numpy().repeat(5), label='selective_strange')
    #
    #
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.show()
    # #
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'additive_attn_ce.pt'), label='additive_attn')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'selective_direct_classifier_ce.pt').flip(0),
    #          label='selective_direct')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'selective_iterative_classifier_ce.pt').flip(0),
    #          label='selective_iterative')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'random_classifier_ce.pt').flip(0), label='random_attn')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'selective_strange_attn_ce.pt').numpy().repeat(5), label='selective_strange')
    #
    # plt.ylim(0, None)
    # plt.legend()
    # plt.show()

    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'additive_attn_acc_train.pt'), label='additive_attn')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'selective_direct_classifier_acc_train.pt').flip(0),
    #          label='selective_direct')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'selective_iterative_classifier_acc_train.pt').flip(0),
    #          label='selective_iterative')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'random_classifier_acc_train.pt').flip(0), label='random_attn')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'selective_strange_attn_acc_train.pt').numpy().repeat(5), label='selective_strange')
    #
    #
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.show()
    # # #
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'additive_attn_ce_train.pt'), label='additive_attn')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'selective_direct_classifier_ce_train.pt').flip(0),
    #          label='selective_direct')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'selective_iterative_classifier_ce_train.pt').flip(0),
    #          label='selective_iterative')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'random_classifier_ce_train.pt').flip(0), label='random_attn')
    # plt.plot(torch.load(DATA_DIR / 'class_errors' / 'selective_strange_attn_ce_train.pt').numpy().repeat(5), label='selective_strange')
    # #
    # plt.ylim(0, None)
    # plt.legend()
    # plt.show()


    random_attn_eval(batch_size=10, tokens_to_add=1, file_name_ce='random_classifier_ce_train.pt', file_name_acc='random_classifier_acc_train.pt', train_data=True)
    # selective_direct_attn_eval(batch_size=2000, step_size=1, file_name_ce='selective_direct_classifier_ce_train.pt', file_name_acc='selective_direct_classifier_acc_train.pt', train_data=True)
    # selective_iterative_attn_eval(batch_size=2000, step_size=1, file_name_ce='selective_iterative_classifier_ce_train.pt', file_name_acc='selective_iterative_classifier_acc_train.pt', train_data=True)
    # additive_attn_eval(batch_size=2000, step_size=5, init_unmask_pos=200, file_name_ce='selective_strange_attn_ce_train.pt', file_name_acc='selective_strange_attn_acc_train.pt', train_data=True)
    # additive_attn_eval(batch_size=2000, step_size=1, init_unmask_pos=None, file_name_ce='additive_attn_ce_train.pt', file_name_acc='additive_attn_acc_train.pt', train_data=True)
    #
    # random_attn_eval(batch_size=2000, tokens_to_add=1, file_name_ce='random_classifier_ce.pt',
    #                  file_name_acc='random_classifier_acc.pt')
    # selective_direct_attn_eval(batch_size=2000, step_size=1, file_name_ce='selective_direct_classifier_ce.pt',
    #                            file_name_acc='selective_direct_classifier_acc.pt')
    # selective_iterative_attn_eval(batch_size=2000, step_size=1,
    #                               file_name_ce='selective_iterative_classifier_ce.pt',
    #                               file_name_acc='selective_iterative_classifier_acc.pt')
    # additive_attn_eval(batch_size=2000, step_size=5, init_unmask_pos=200,
    #                    file_name_ce='selective_strange_attn_ce.pt',
    #                    file_name_acc='selective_strange_attn_acc.pt')
    # additive_attn_eval(batch_size=2000, step_size=1, init_unmask_pos=None, file_name_ce='additive_attn_ce.pt',
    #                    file_name_acc='additive_attn_acc.pt')



    #

    #



    # selective_direct_inverse_attn_eval(batch_size=500)
    # additive_attn_eval(batch_size=500)
    # selective_iterative_attn_eval(batch_size=500)
    # za
    # main(args)
    # dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
