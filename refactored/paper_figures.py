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

HOSTNAME = socket.gethostname()

def set_seed(seed: int):
    import random
    import os
    import numpy as np
    import torch

    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN / CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed (important for dict/set order)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Enforce deterministic algorithms (PyTorch >= 1.8)
    torch.use_deterministic_algorithms(True)

    print(f"Seed set to {seed}")

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
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
else:
    DEVICE = 'cpu'




if HOSTNAME == 'gpu03':
    IMG_NET_TRAIN = '/local/reyhasjb/datasets/Imagenet-100class/train'
    IMG_NET_VAL = '/local/reyhasjb/datasets/Imagenet-100class/val'
    QUANTIZED_EPOCH_PATH = DATA_DIR / 'vqvae' / 'quantized_epoch80_flat_vqvae80x80_144x456codebook.npy'
    INDICES_PATH = DATA_DIR / 'vqvae' / 'indices_epoch80_flat_vqvae80x80_144x456codebook.npy'
    LABELS_PATH = DATA_DIR / 'vqvae' / 'labels_epoch80_flat_vqvae80x80_144x456codebook.npy'
    EIGHTY_EIGHTY_PATH = DATA_DIR / 'vqvae' / '80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100.pt'
    VQVAE_PATH = DATA_DIR / 'vqvae' / 'model_epoch80_flat_vqvae80x80_144x456codebook.pth'
    CUSTOM_CLASSIFIER_WEIGHTS = DATA_DIR / RUNS_DIR / 'VQVAET-482' / 'model_st.pt'

    TRANSFORMER_SEL_M_WEIGHTS = DATA_DIR / RUNS_DIR / 'VQVAET-491' / 'model_st.pt'
    TRANSFORMER_SEL_WEIGHTS = DATA_DIR / RUNS_DIR / 'VQVAET-492' / 'model_st.pt'
    TRANSFORMER_RND_M_WEIGHTS = DATA_DIR / RUNS_DIR / 'VQVAET-489' / 'model_st.pt'
    TRANSFORMER_RND_WEIGHTS = DATA_DIR / RUNS_DIR / 'VQVAET-488' / 'model_st.pt'

else:
    DATA_DIR = Path(__file__).parent / 'data'
    TMP_DIR = DATA_DIR / 'tmp'
    RUNS_DIR = DATA_DIR / 'runs'
    IMG_NET_TRAIN = '/Users/rathjjgf/datasets/Imagenet-100class/val'  # for debugging only
    IMG_NET_VAL = '/Users/rathjjgf/datasets/Imagenet-100class/val'  # for debugging only
    QUANTIZED_EPOCH_PATH = DATA_DIR / 'vqvae' / 'quantized_epoch80_flat_vqvae80x80_144x456codebook.npy'
    INDICES_PATH = DATA_DIR / 'vqvae' / 'indices_epoch80_flat_vqvae80x80_144x456codebook.npy'
    LABELS_PATH = DATA_DIR / 'vqvae' / 'labels_epoch80_flat_vqvae80x80_144x456codebook.npy'
    EIGHTY_EIGHTY_PATH = DATA_DIR / 'vqvae' / '80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100.pt'
    VQVAE_PATH = DATA_DIR / 'vqvae' / 'model_epoch80_flat_vqvae80x80_144x456codebook.pth'
    CUSTOM_CLASSIFIER_WEIGHTS = DATA_DIR / RUNS_DIR / 'VQVAET-482' / 'model_st.pt'
    TRANSFORMER_SEL_M_WEIGHTS = DATA_DIR / RUNS_DIR / 'VQVAET-491' / 'model_st.pt'
    TRANSFORMER_SEL_WEIGHTS = DATA_DIR / RUNS_DIR / 'VQVAET-492' / 'model_st.pt'
    TRANSFORMER_RND_M_WEIGHTS = DATA_DIR / RUNS_DIR / 'VQVAET-489' / 'model_st.pt'
    TRANSFORMER_RND_WEIGHTS = DATA_DIR / RUNS_DIR / 'VQVAET-488' / 'model_st.pt'


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


def full_mask(q, indices):
    mask_pattern = torch.ones_like(indices, dtype=torch.bool)
    masked_q = q.clone()  # shallow copy
    masked_q[mask_pattern] = 0  # Assuming 0 is the mask token
    mask_indices = indices.clone()
    mask_indices[~mask_pattern] = -100  # Assuming -100 is the mask label token
    return masked_q, mask_indices, mask_pattern


def vqvae_setup():
    model_vqvae = FlatVQVAE()
    model_vqvae.load_state_dict(torch.load(VQVAE_PATH, map_location=DEVICE))
    model_vqvae = model_vqvae.to(DEVICE).eval()
    return model_vqvae

def transformer_setup(weight_path):
    cfg = DistilBertConfig(vocab_size=456, hidden_size=144, sinusoidal_pos_embds=False, n_layers=6, n_heads=4, max_position_embeddings=400)
    model_distil = DistilBertForMaskedLM(cfg).to(DEVICE)
    model_distil.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model_distil = model_distil.to(DEVICE).eval()
    model_distil.eval()
    return model_distil


def classifier_setup():
    classifier = resnet50(pretrained=False)
    classifier.fc = nn.Linear(2048, 100)
    classifier.load_state_dict(torch.load(CUSTOM_CLASSIFIER_WEIGHTS, map_location=torch.device(DEVICE),weights_only=False)())
    classifier.to(DEVICE).eval()
    return classifier

def model_setup(transformer_path):
    vqvae = vqvae_setup()
    transformer = transformer_setup(transformer_path)
    classifier = classifier_setup()
    return vqvae, transformer, classifier



def load_data(bs, ids_only=False):
    transform_val = transforms.Compose([transforms.Resize((80, 80)), transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset_val = datasets.ImageFolder(IMG_NET_VAL, transform=transform_val)
    if ids_only:
        img_ids = (462, 1671, 1836, 4970, 5852, 7777, 8513, 8685, 9469, 9644)



    val_dataloader = DataLoader(dataset_val, batch_size=bs, shuffle=True)
    return val_dataloader


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


def set_border(ax, correctly_classified, no_color=False):
    for spine in ax.spines.values():
        color = 'green' if correctly_classified else 'red'
        if no_color:
            color = 'black'
        spine.set_edgecolor(color)
        spine.set_linewidth(2)

    ax.set_xticks([])
    ax.set_yticks([])





def accuracy(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    e = pred.eq(target.view_as(pred))
    return e


@torch.no_grad()
def random_plot(img, labels, vqvae, transformer, classifier, step_size=1, img_ids=None, file_name='recons_rnd_attn', plot_every=40):
    q_batch, _, id_b, _, _ = vqvae.encode(img)
    q_batch, id_b = torch.flatten(q_batch, start_dim=2).permute(0, 2, 1), torch.flatten(id_b, start_dim=1)
    q_masked, index_masked = q_batch.clone(), id_b.clone()
    rnd_mask = torch.stack([torch.randperm(q_batch.shape[1]) for _ in range(q_batch.shape[0])]).to(DEVICE)
    mask = torch.ones(size=(q_batch.shape[0], q_batch.shape [1]))
    masks, recons, confidences, normed_confidences, perc, correctly_classified, recon_without, correctly_classified_without = [], [], [], [], [], [], [], []
    rows = torch.arange(q_batch.size(0)).unsqueeze(1)

    for i in range(0, q_batch.shape[1] + 1, step_size):
        pos_to_mask = rnd_mask[:, :i]
        q_masked[rows, pos_to_mask]  = 0
        logits = transformer(inputs_embeds=q_masked, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)

        mask[rows, pos_to_mask[:, :i]] = 0
        index_masked[rows, pos_to_mask] = max_index_per_pos[rows, pos_to_mask]
        if i % plot_every == 0:
            recons.append(vqvae.decode_code(index_masked.reshape(-1, 20, 20)))
            class_logits = classifier(preprocess(denormalize(recons[-1])))
            correctly_classified.append(accuracy(class_logits, labels))
            masks.append(mask.clone())
            perc.append( i / 400)
            confidences.append(max_conf_per_pos)
            normed_confidences.append(torch.max(torch.softmax(logits, dim=2), dim=2)[0])
            recon_without.append(vqvae.decode(q_masked.permute(0, 2, 1).reshape(-1, 144, 20, 20)))
            class_logits = classifier(preprocess(denormalize(recon_without[-1])))
            correctly_classified_without.append(accuracy(class_logits, labels))
    save_pkl(file_name, recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without)

    # plot_recons(recons, masks, perc, confidences, normed_confidences, file_name, correctly_classified)
@torch.no_grad()
def selective_direct_plot(img, labels, vqvae, transformer, classifier, step_size=1, img_ids=None, file_name='recons_sel_attn', plot_every=40):
    q_batch, _, id_b, _, _ = vqvae.encode(img)
    q_b, index_b = torch.flatten(q_batch, start_dim=2).permute(0, 2, 1), torch.flatten(id_b, start_dim=1)
    mask = torch.ones(size=(q_b.shape[0], q_b.shape[1]))
    rows = torch.arange(index_b.size(0)).unsqueeze(1)

    logits_from_unmasked_img = transformer(inputs_embeds=q_b, output_hidden_states=True).logits
    max_conf_unmasked_img, max_index_unmasked_img = torch.max(logits_from_unmasked_img, dim=2)
    arg_sort_max_conf_unmasked_img = torch.argsort(max_conf_unmasked_img, dim=1, descending=True)

    masks, recons, confidences, normed_confidences, perc, correctly_classified, recon_without, correctly_classified_without = [], [], [], [], [], [], [], []

    for i in range(0, q_b.shape[1] + 1, step_size):
        masked_pos = arg_sort_max_conf_unmasked_img[:, :i]
        mask[rows, masked_pos] = 0
        q_b[rows, masked_pos] = 0

        logits_from_masked_img = transformer(inputs_embeds=q_b, output_hidden_states=True).logits
        max_conf_masked_img, max_index_masked_img = torch.max(logits_from_masked_img, dim=2)
        index_b[rows, masked_pos] = max_index_masked_img[rows, masked_pos]
        if i % plot_every == 0:
            recons.append(vqvae.decode_code(index_b.reshape(-1, 20, 20).to(DEVICE)))
            class_logits = classifier(preprocess(denormalize(recons[-1])))
            correctly_classified.append(accuracy(class_logits, labels))
            masks.append(mask.clone())
            perc.append(i / 400)
            confidences.append(max_conf_masked_img)
            normed_confidences.append(torch.max(torch.softmax(logits_from_masked_img, dim=2), dim=2)[0])
            recon_without.append(vqvae.decode(q_b.permute(0, 2, 1).reshape(-1, 144, 20, 20)))
            class_logits = classifier(preprocess(denormalize(recon_without[-1])))
            correctly_classified_without.append(accuracy(class_logits, labels))


    save_pkl(file_name, recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without)


@torch.no_grad()
def additive_plot(img, labels, vqvae, transformer, classifier, step_size=1, img_ids=None, file_name='recons_sel_attn', plot_every=40, init_unmask_pos=None):
    q_batch, _, id_b, _, _ = vqvae.encode(img)
    q_batch, index_repr_batch = torch.flatten(q_batch, start_dim=2).permute(0, 2, 1), torch.flatten(id_b, start_dim=1)
    q_masked, _, _ = full_mask(q_batch, index_repr_batch)
    pos_to_unmask = torch.empty((q_batch.shape[0], 0), dtype=torch.int64).to(DEVICE)
    rows = torch.arange(q_batch.size(0)).unsqueeze(1)
    mask = torch.zeros(size=(q_batch.shape[0], q_batch.shape[1]))
    masks, recons, confidences, normed_confidences, perc, correctly_classified, recon_without, correctly_classified_without = [], [], [], [], [], [], [], []

    for i in range(0, q_batch.shape[1] + 1, step_size):
        q_masked[rows, pos_to_unmask] = q_batch[rows, pos_to_unmask]
        logits = transformer(inputs_embeds=q_masked, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
        max_index_per_pos[rows, pos_to_unmask] = index_repr_batch[rows, pos_to_unmask]
        recons_from_max_indices = vqvae.decode_code(max_index_per_pos.reshape(-1, 20, 20).to(DEVICE))
        sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1)
        pos_to_unmask = conc_unique_elements(pos_to_unmask, sorted_max_conf_per_pos, n_elements=1)
        if i % plot_every == 0:
            recons.append(recons_from_max_indices)
            mask[rows, pos_to_unmask[:, :i]] = 1
            class_logits = classifier(preprocess(denormalize(recons[-1])))
            correctly_classified.append(accuracy(class_logits, labels))
            masks.append(mask.clone())
            perc.append((400 - i) / 400)
            confidences.append(max_conf_per_pos)
            normed_confidences.append(torch.max(torch.softmax(logits, dim=2), dim=2)[0])
            recon_without.append(vqvae.decode(q_masked.permute(0, 2, 1).reshape(-1, 144, 20, 20)))
            class_logits = classifier(preprocess(denormalize(recon_without[-1])))
            correctly_classified_without.append(accuracy(class_logits, labels))

    save_pkl(file_name, recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without)


def save_pkl(file_name, recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without):
    with open(DATA_DIR / 'plots' / (file_name + '.pkl'), "wb") as f:
        pickle.dump((recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without), f)

def load_pkl(file_name):
    with open(DATA_DIR / 'plots' / (file_name + '.pkl'), "rb") as f:
        recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without = pickle.load(f)
    return recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without

def plot_x(data, axs, num_images, reverse=False):
    recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without = data
    if reverse:
        recons.reverse(), masks.reverse(), perc.reverse(), confidences.reverse(), normed_confidences.reverse(), correctly_classified.reverse(),  recon_without.reverse(), correctly_classified_without.reverse()
    for j in range(num_images):
        for k in range(len(masks)):
            # axs[k][].set_ylabel(perc[k])

            axs[3 * j + 2][k].imshow(masks[k].reshape(-1, 20, 20, 1)[j], vmin=0, vmax=1)
            set_border(axs[j * 3 + 2][k], correctly_classified[k][j], no_color=True)

            axs[j * 3 + 1][k].imshow(denormalize(recons[k])[j].permute(1, 2, 0))
            set_border(axs[j * 3 + 1][k], correctly_classified[k][j])

            axs[j * 3 + 0][k].imshow(denormalize(recon_without[k])[j].permute(1, 2, 0))
            set_border(axs[j * 3 + 0][k], correctly_classified_without[k][j])




            # axs[k][4 * j + 1].imshow(masks[k].reshape(-1, 20, 20, 1)[j], vmin=0, vmax=1)
            # axs[k][4 * j + 2].imshow(confidences[k].reshape(-1, 1, 20, 20)[j].permute(1, 2, 0).numpy())
            # axs[k][4 * j + 3].imshow(normed_confidences[k].reshape(-1, 1, 20, 20)[j].permute(1, 2, 0).numpy(),
            #                          vmin=0, vmax=1)

def get_series(key_prefix, key_suffix, run_id, update=False, load=True):
    path = RUNS_DIR / run_id / (key_suffix + '.npy')
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
    run_ids = [('VQVAET-491', 'selective'), ('VQVAET-489', 'random')]
    for run_id in run_ids:
        ser = get_series(key_prefix='val', key_suffix='rnd_acc_masked_only', run_id=run_id[0], load=True)
        ax[0][0].plot(ser, label=run_id[1])
    for run_id in run_ids:
        ser = get_series(key_prefix='val', key_suffix='rnd_class_acc', run_id=run_id[0], load=True)
        ax[0][1].plot(ser,  label=run_id[1])

    for run_id in run_ids:
        ser = get_series(key_prefix='val', key_suffix='selective_desc_acc_masked_only', run_id=run_id[0], load=True)
        ax[1][0].plot(ser, label=run_id[1])
    for run_id in run_ids:
        ser = get_series(key_prefix='val', key_suffix='selective_desc_class_acc', run_id=run_id[0], load=True)
        ax[1][1].plot(ser, label=run_id[1])

    ax[1][1].legend()
    plt.show()


def plot_masking_ratios():
    n_rows, n_cols = 1, 3
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    axs[0].plot(torch.load(DATA_DIR / 'mask_ratio_eval' / 'add_mse.pt'), label='additive')
    axs[0].plot(torch.load(DATA_DIR / 'mask_ratio_eval' / 'sel_mse.pt'), label='selective')
    axs[0].plot(torch.load(DATA_DIR / 'mask_ratio_eval' / 'rnd_mse.pt'), label='random')

    axs[1].plot(torch.load(DATA_DIR / 'mask_ratio_eval' / 'add_acc.pt'), label='additive')
    axs[1].plot(torch.load(DATA_DIR / 'mask_ratio_eval' / 'sel_acc.pt'), label='selective')
    axs[1].plot(torch.load(DATA_DIR / 'mask_ratio_eval' / 'rnd_acc.pt'), label='random')

    axs[2].plot(torch.load(DATA_DIR / 'mask_ratio_eval' / 'add_ce.pt'), label='additive')
    axs[2].plot(torch.load(DATA_DIR / 'mask_ratio_eval' / 'sel_ce.pt'), label='selective')
    axs[2].plot(torch.load(DATA_DIR / 'mask_ratio_eval' / 'rnd_ce.pt'), label='random')

    plt.ylim(0, None)

    plt.legend()
    plt.show()





def plot_recons(num_images, transformer='rnd'):
    n_rows, n_cols = num_images * 3 * 3,  11
    fontsize = 22
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))
    add_data = load_pkl(f'{transformer}_add_m')
    plot_x(data=add_data, axs=axs[:n_rows // 3, :], num_images=num_images, reverse=True)

    sel_data = load_pkl(f'{transformer}_sel_m')
    plot_x(data=sel_data, axs=axs[(n_rows // 3):(n_rows // 3) * 2 :], num_images=num_images)

    rnd_data = load_pkl(f'{transformer}_rnd_m')
    plot_x(data=rnd_data, axs=axs[(n_rows // 3) * 2 :, :], num_images=num_images)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    mask_ratios = np.linspace(0, 100, 11, dtype=int)
    for i in range(n_cols):
        axs[0, i].set_title(mask_ratios[i], fontsize=fontsize)

    shift_rows_and_columns(axs, rows_to_shift=np.arange(0, n_rows, 3))
    plt.show()
    # fig.savefig(DATA_DIR / 'plots' / (file_name + '.pdf'), format='pdf')
    # pass
    # mult_columns = 1
    # n_rows, n_cols = len(masks), recons[0].shape[0] * mult_columns
    # fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))
    # for j in range(0, recons[0].shape[0]):
    #     for k in range(len(masks)):
    #         axs[k][mult_columns * j].set_ylabel(perc[k])
    #         axs[k][mult_columns * j].imshow(denormalize(recons[k])[j].permute(1, 2, 0))
    #         set_border(axs[k][mult_columns * j],  correctly_classified[k][j])
    #         # axs[k][4 * j + 1].imshow(masks[k].reshape(-1, 20, 20, 1)[j], vmin=0, vmax=1)
    #         # axs[k][4 * j + 2].imshow(confidences[k].reshape(-1, 1, 20, 20)[j].permute(1, 2, 0).numpy())
    #         # axs[k][4 * j + 3].imshow(normed_confidences[k].reshape(-1, 1, 20, 20)[j].permute(1, 2, 0).numpy(),
    #         #                          vmin=0, vmax=1)
    # # hide_all_extras(axs)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)
    port = (2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14)
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--ckpt_vqvae', type=str, default=VQVAE_PATH)
    parser.add_argument('--ckpt_distil_combined', type=str, default=EIGHTY_EIGHTY_PATH)

    args = parser.parse_args()

    set_seed(0)

    dataloader = load_data(10, ids_only=False)
    img, labels = next(iter(dataloader))
    vqvae, transformer, classifier = model_setup(TRANSFORMER_RND_M_WEIGHTS)
    preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
    # #
    random_plot(img=img, labels=labels, vqvae=vqvae, transformer=transformer, classifier=classifier, step_size=40, file_name='rnd_rnd_m')  # transfomer - masking technique
    selective_direct_plot(img=img, labels=labels, vqvae=vqvae, transformer=transformer, classifier=classifier, step_size=40, file_name='rnd_sel_m')
    additive_plot(img=img, labels=labels, vqvae=vqvae, transformer=transformer, classifier=classifier, step_size=1, file_name='rnd_add_m')
    #
    # transformer = transformer_setup(TRANSFORMER_SEL_M_WEIGHTS)
    # random_plot(img=img, labels=labels, vqvae=vqvae, transformer=transformer, classifier=classifier, step_size=40, file_name='sel_rnd_m')
    # selective_direct_plot(img=img, labels=labels, vqvae=vqvae, transformer=transformer, classifier=classifier, step_size=40, file_name='sel_sel_m')
    # additive_plot(img=img, labels=labels, vqvae=vqvae, transformer=transformer, classifier=classifier, step_size=1, file_name='sel_add_m')
    #
    #
    #
    # #
    # plot_recons(num_images=1, transformer='rnd')
    # plot_recons(num_images=1, transformer='sel')

    plot_token_val()

    pass
    # additive_attn_eval(batch_size=500, step_size=1, init_unmask_pos=200, file_name='additive_attn_recon_errors_init_200.pt')
    # additive_attn_eval(batch_size=500, step_size=1, init_unmask_pos=190, file_name='additive_attn_recon_errors_init_190.pt')
    # additive_attn_eval(batch_size=500, step_size=5, file_name='additive_attn_recon_errors_stepsize_5.pt')
    # additive_plot(batch_size=10, step_size=1, img_ids=(462, 1671,  1836, 4970, 5852, 7777, 8513, 8685, 9469, 9644), init_unmask_pos=190, file_name='recons_add_attn_190_stepsize_1')


    #
    # additive_plot(batch_size=10, step_size=5, img_ids=(462, 1671,  1836, 4970, 5852, 7777, 8513, 8685, 9469, 9644))
    # random_plot(batch_size=10, step_size=10, img_ids=(462, 1671,  1836, 4970, 5852, 7777, 8513, 8685, 9469, 9644), plot_every=40)

    #
    # selective_direct_attn_eval(batch_size=500, step_size=1)
    # selective_direct_attn_eval(batch_size=100, step_size=1, reverse=True)
    # additive_plot(batch_size=10, img_ids=(462, 1671,  1836, 4970, 5852, 7777, 8513, 8685, 9469, 9644), plot_every=40)


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

