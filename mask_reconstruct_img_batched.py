import numpy as np
import pandas as pd
import argparse, math, sys, os, random
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
else:
    DATA_DIR = Path(__file__).parent / 'data'
    TMP_DIR = DATA_DIR / 'tmp'
    RUNS_DIR = DATA_DIR / 'runs'
    QUANTIZED_EPOCH_PATH = DATA_DIR / 'vqvae' / 'quantized_epoch80_flat_vqvae80x80_144x456codebook.npy'
    INDICES_PATH = DATA_DIR / 'vqvae' / 'indices_epoch80_flat_vqvae80x80_144x456codebook.npy'
    LABELS_PATH = DATA_DIR / 'vqvae' / 'labels_epoch80_flat_vqvae80x80_144x456codebook.npy'
    EIGHTY_EIGHTY_PATH = DATA_DIR / 'vqvae' / '80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100.pt'
    VQVAE_PATH = DATA_DIR / 'vqvae' / 'model_epoch80_flat_vqvae80x80_144x456codebook.pth'


def full_mask(quantizes, indices):
    mask_pattern = torch.ones_like(indices,dtype=torch.bool)
    mask_quantizes = quantizes.clone()  # shallow copy
    mask_quantizes[mask_pattern] = 0  # Assuming 0 is the mask token
    mask_indices = indices.clone()
    mask_indices[~mask_pattern] = -100  # Assuming -100 is the mask label token
    return mask_quantizes, mask_indices, mask_pattern


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


def vqvae_model_setup(ckpt_vqvae, device):
    model_vqvae = FlatVQVAE().to(device)
    model_vqvae.load_state_dict(torch.load(ckpt_vqvae, map_location=device))
    model_vqvae = model_vqvae.to(device)
    model_vqvae.eval()
    return model_vqvae


def setup_resources():
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(3)
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
    return device


def transformer_setup(device, args, n_token, vocab_size, d_embed_vec):
    cfg = DistilBertConfig(vocab_size=vocab_size, hidden_size=d_embed_vec, sinusoidal_pos_embds=False, n_layers=6,
                           n_heads=4, max_position_embeddings=n_token)
    model_distil = DistilBertForMaskedLM(cfg).to(device)
    model_state = torch.load(args.ckpt_distil_combined, map_location=torch.device(device))
    model_distil.load_state_dict(model_state)
    model_distil = model_distil.to(device)
    model_distil.eval()
    return model_distil


def vqvae_setup(device, args):
    model_vqvae = FlatVQVAE().to(device)
    model_vqvae.load_state_dict(torch.load(args.ckpt_vqvae, map_location=device))
    model_vqvae = model_vqvae.to(device)
    model_vqvae.eval()
    return model_vqvae


def mask_iter(min, max, step, n_token, arbitrary_percentage: list = [.85, .95]):
    mask_percentages = np.arange(min, max, step)
    mask_percentages = np.append(mask_percentages, arbitrary_percentage) if arbitrary_percentage else mask_percentages
    mask_percentages = np.sort(mask_percentages)
    reverse_mask_percentages = mask_percentages[::-1]  # reverse the sorted array
    mask_perc_map_indices_length = n_token * reverse_mask_percentages
    return mask_percentages, reverse_mask_percentages, mask_perc_map_indices_length


def load_data(n_samples=10):
    quantizes, quant_b = load_embedding_space()
    indices = load_indices()
    labels = load_labels()
    return labels[0:n_samples], indices[0:n_samples], quantizes[0:n_samples], quant_b[0:n_samples]


def setup_attn_eval(n_samples=10):
    device = setup_resources()
    labels, indices, quantizes, quant_b = load_data(n_samples=n_samples)
    n_sample, d_embed_vec, n_token = quantizes.shape[0], quantizes.shape[2], quantizes.shape[1]
    model_distil = transformer_setup(device, args, n_token, 456, d_embed_vec)
    model_vqvae = vqvae_setup(device, args)
    q = torch.from_numpy(quantizes).to(device)
    index = torch.from_numpy(indices).to(device)
    q_masked, index_masked, mask_pattern = full_mask(q, index)
    q_masked = q_masked.to(device)
    vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b).to(device))
    length = int(math.sqrt(q.shape[1]))
    return model_distil, model_vqvae, q_masked, q, index, vqvae_out, device, length


@torch.no_grad()
def additive_attn_eval(tokens_to_add=1):
    model_distil, model_vqvae, q_masked, q, index, vqvae_out, device, length = setup_attn_eval()
    positions_to_unmask = torch.empty((q.shape[0], 0), dtype=torch.int64)
    for _ in range(0, q.shape[1] // tokens_to_add + 1):
        logits = model_distil(inputs_embeds=q_masked, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
        sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1)
        positions_to_unmask = conc_unique_elements(positions_to_unmask, sorted_max_conf_per_pos,
                                                   n_elements=tokens_to_add)

        max_index_per_pos.scatter_(1, positions_to_unmask, index.gather(1, positions_to_unmask))
        expanded = positions_to_unmask[:, -tokens_to_add:].unsqueeze(-1).expand(-1, -1, q.size(-1))
        q_masked.scatter_(1, expanded, q.gather(1, expanded))
        recons_from_max_indices = model_vqvae.decode_code(
            max_index_per_pos.reshape(-1, length, length).to(device))  # bx20x20 -> bx3x80x80
        recon_loss = F.mse_loss(recons_from_max_indices, vqvae_out)
        print(recon_loss.item())


@torch.no_grad()
def random_attn_eval(tokens_to_add=1):
    model_distil, model_vqvae, q_masked, q, index, vqvae_out, device, length = setup_attn_eval()
    rnd_mask = torch.stack([torch.randperm(q.shape[1]) for _ in range(q.shape[0])])
    for i in range(0, q.shape[1] // tokens_to_add + 1):
        logits = model_distil(inputs_embeds=q_masked, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
        max_index_per_pos.scatter_(1, rnd_mask[:, :i], index.gather(1, rnd_mask[:, :i]))
        expanded = rnd_mask[:, :i].unsqueeze(-1).expand(-1, -1, q.size(-1))
        q_masked.scatter_(1, expanded, q.gather(1, expanded))
        recons_from_max_indices = model_vqvae.decode_code(
            max_index_per_pos.reshape(-1, length, length).to(device))
        recon_loss = F.mse_loss(input=recons_from_max_indices, target=vqvae_out)
        print(recon_loss.item())


@torch.no_grad()
def selective_attn_eval(tokens_to_add=1):
    model_distil, model_vqvae, q_masked, q, index, vqvae_out, device, length = setup_attn_eval()
    q_unmasked = q.clone()
    positions_to_unmask = torch.empty((q.shape[0], 0), dtype=torch.int64)  # bx400
    recon_losses = []
    for _ in range(0, q.shape[1] // tokens_to_add + 1):
        logits = model_distil(inputs_embeds=q_unmasked, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
        sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1)
        positions_to_unmask = conc_unique_elements(positions_to_unmask, sorted_max_conf_per_pos, n_elements=tokens_to_add)

        logits = model_distil(inputs_embeds=q_masked, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)

        max_index_per_pos.scatter_(1, positions_to_unmask, index.gather(1, positions_to_unmask))
        expanded = positions_to_unmask[:, -tokens_to_add:].unsqueeze(-1).expand(-1, -1, q.size(-1))

        q_unmasked.scatter_(1, expanded, 0)
        q_masked.scatter_(1, expanded, q.gather(1, expanded))

        recons_from_max_indices = model_vqvae.decode_code(max_index_per_pos.reshape(-1, length, length).to(device))  # bx20x20 -> bx3x80x80
        recon_loss = F.mse_loss(recons_from_max_indices, vqvae_out)
        print(recon_loss.item())
        recon_losses.append(recon_loss.item())




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (2 ** 15 + 2 ** 14+ hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14)
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--ckpt_vqvae', type=str,default=VQVAE_PATH)
    parser.add_argument('--ckpt_distil_combined', type=str, default=EIGHTY_EIGHTY_PATH)
    #
    #

    args = parser.parse_args()

    #
    # selective_attn_eval()
    # additive_attn_eval()
    random_attn_eval()

    # main(args)
    # dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
