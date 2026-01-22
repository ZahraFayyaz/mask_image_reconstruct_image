import argparse
from sched import scheduler
import sys
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import distributed as dist
from transformers import DistilBertForMaskedLM, DistilBertConfig
import neptune.new as neptune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from pathlib import Path
import socket
from vqvae import FlatVQVAE
from torch.functional import F

DATA_DIR = Path(__file__).parent / 'data'
TMP_DIR = DATA_DIR / 'tmp'
RUNS_DIR = DATA_DIR / 'runs'
HOSTNAME = socket.gethostname()

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.cuda.set_device(1)
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

else:
    DATA_DIR = Path(__file__).parent / 'data'
    TMP_DIR = DATA_DIR / 'tmp'
    RUNS_DIR = DATA_DIR / 'runs'
    IMG_NET_TRAIN = '/Users/rathjjgf/datasets/small_ecoset/val'  # for debugging only
    IMG_NET_VAL = '/Users/rathjjgf/datasets/small_ecoset/val'  # for debugging only
    QUANTIZED_EPOCH_PATH = DATA_DIR / 'vqvae' / 'quantized_epoch80_flat_vqvae80x80_144x456codebook.npy'
    INDICES_PATH = DATA_DIR / 'vqvae' / 'indices_epoch80_flat_vqvae80x80_144x456codebook.npy'
    LABELS_PATH = DATA_DIR / 'vqvae' / 'labels_epoch80_flat_vqvae80x80_144x456codebook.npy'
    EIGHTY_EIGHTY_PATH = DATA_DIR / 'vqvae' / '80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100.pt'
    VQVAE_PATH = DATA_DIR / 'vqvae' / 'model_epoch80_flat_vqvae80x80_144x456codebook.pth'


def mask_q(quant_b, model_distil, masking_strategy, mask_ratio=0.75, descending=True):
    rows = torch.arange(quant_b.size(0)).unsqueeze(1)
    if masking_strategy == 'random':
        rnd_mask = torch.stack([torch.randperm(quant_b.shape[1]) for _ in range(quant_b.shape[0])]).to(DEVICE)
        pos_to_mask = rnd_mask[:, :int(mask_ratio * quant_b.shape[1])]
    elif masking_strategy == 'selective':
        logits_from_unmasked_img = model_distil(inputs_embeds=quant_b, output_hidden_states=False).logits
        max_conf_unmasked_img, max_index_unmasked_img = torch.max(logits_from_unmasked_img, dim=2)
        arg_sort_max_conf_unmasked_img = torch.argsort(max_conf_unmasked_img, dim=1, descending=descending)
        pos_to_mask = arg_sort_max_conf_unmasked_img[:,  :int(mask_ratio * quant_b.shape[1])]
    else:
        raise NameError(f'Unknown masking strategy {masking_strategy}')
    quant_b[rows, pos_to_mask] = 0
    return quant_b


@torch.no_grad()
def eval_model(model_distil, model_vqvae, dataloader, masking_ratio=0.75, run=None):
    sum_acc_rnd, sum_ce_rnd, sum_ce_select, sum_acc_select, sum_ce_select_desc, sum_acc_select_desc, n_inputs = 0, 0, 0, 0, 0, 0, 0

    model_distil.eval(), model_vqvae.eval()
    for batch_id, (img, _) in enumerate(dataloader):
        img = img.to(DEVICE, non_blocking=True)
        quant_b, _, id_b, _, _ = model_vqvae.encode(img)
        quant_b, id_b = torch.flatten(quant_b, start_dim=2).permute(0, 2, 1), torch.flatten(id_b, start_dim=1)
        quant_b_2, quant_b_3 = quant_b.clone(), quant_b.clone()

        quant_b = mask_q(quant_b, model_distil, 'selective', masking_ratio, descending=True)
        logits = model_distil(inputs_embeds=quant_b, output_hidden_states=False).logits
        sum_ce_select_desc += F.cross_entropy(input=logits.permute(0, 2, 1), target=id_b) * img.shape[0]
        preds = logits.argmax(dim=-1)  # (64, 400)
        acc = (preds == id_b).float().mean()
        sum_acc_select_desc += acc * img.shape[0]

        quant_b = mask_q(quant_b_3, model_distil, 'selective', masking_ratio, descending=False)
        logits = model_distil(inputs_embeds=quant_b, output_hidden_states=False).logits
        sum_ce_select += F.cross_entropy(input=logits.permute(0, 2, 1), target=id_b) * img.shape[0]
        preds = logits.argmax(dim=-1)  # (64, 400)
        acc = (preds == id_b).float().mean()
        sum_acc_select += acc * img.shape[0]

        quant_b = mask_q(quant_b_2, model_distil, 'random', masking_ratio,)
        logits = model_distil(inputs_embeds=quant_b, output_hidden_states=False).logits
        sum_ce_rnd += F.cross_entropy(input=logits.permute(0, 2, 1), target=id_b) * img.shape[0]
        preds = logits.argmax(dim=-1)  # (64, 400)
        acc = (preds == id_b).float().mean()
        sum_acc_rnd += acc * img.shape[0]
        n_inputs += img.shape[0]

    sum_acc_rnd, sum_ce_rnd = (sum_acc_rnd / n_inputs).item(), (sum_ce_rnd / n_inputs).item()
    sum_acc_select, sum_ce_select = (sum_acc_select / n_inputs).item(), (sum_ce_select / n_inputs).item()
    sum_acc_select_desc, sum_ce_select_desc = (sum_acc_select_desc / n_inputs).item(), (sum_ce_select_desc / n_inputs).item()

    if run:
        run["series/val/rnd_ce_loss"].append(sum_ce_rnd)
        run["series/val/rnd_acc"].append(sum_acc_rnd)

        run["series/val/select_ce_loss"].append(sum_ce_select)
        run["series/val/select_acc"].append(sum_acc_select)

        run["series/val/select_ce_loss_desc"].append(sum_ce_select_desc)
        run["series/val/select_acc_desc"].append(sum_acc_select_desc)

    return sum_ce_rnd


if __name__ == '__main__':
    os.nice(19)
    bs = 2048
    epochs = 10000
    masking_ratio = 0.75
    masking_strategy = 'random'
    lr = 0.0003
    descending = False

    train_cfg = {'batch_size': bs, 'masking_ratio': masking_ratio, 'masking_strategy': masking_strategy, 'lr': lr, 'descending': descending}
    run = neptune.init_run(project="tns/Vqvae-transformer", monitoring_namespace='monitoring',
                           capture_stdout=False, capture_stderr=False, capture_hardware_metrics=False)
    run_id = run["sys/id"].fetch()
    run['cfg'] = train_cfg

    (RUNS_DIR / run_id).mkdir(exist_ok=True)
    transform = transforms.Compose([transforms.Resize((80, 80)), transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataset_train = datasets.ImageFolder(IMG_NET_TRAIN, transform=transform)
    dataset_val = datasets.ImageFolder(IMG_NET_VAL, transform=transform)

    train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=True)

    model_vqvae = FlatVQVAE()
    model_vqvae.load_state_dict(torch.load(VQVAE_PATH, map_location=DEVICE))
    model_vqvae = model_vqvae.to(DEVICE).eval()

    cfg = DistilBertConfig(vocab_size=456, hidden_size=144, sinusoidal_pos_embds=False, n_layers=6,
                           n_heads=4, max_position_embeddings=400)
    model_distil = DistilBertForMaskedLM(cfg)
    model_distil = model_distil.to(DEVICE).eval()

    optimizer = torch.optim.AdamW(model_distil.parameters(), lr=lr)

    best_eval_loss = eval_model(model_distil=model_distil, model_vqvae=model_vqvae, dataloader=val_dataloader, run=run)
    for epoch in range(epochs):
        for _, (img, _) in enumerate(train_dataloader):
            img = img.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                quant_b, _, id_b, _, _ = model_vqvae.encode(img)
                quant_b, id_b = torch.flatten(quant_b, start_dim=2).permute(0, 2, 1), torch.flatten(id_b, start_dim=1)
                quant_b = mask_q(quant_b, model_distil, masking_strategy, masking_ratio, descending)
            logits = model_distil(inputs_embeds=quant_b, output_hidden_states=False).logits
            loss = F.cross_entropy(input=logits.permute(0, 2, 1), target=id_b)
            optimizer.zero_grad()  # 1. Clear old gradients
            loss.backward()  # 2. Compute gradients (∂loss/∂params)
            optimizer.step()
            run['train/loss'].append(loss)
        eval_loss = eval_model(model_distil=model_distil, model_vqvae=model_vqvae, dataloader=val_dataloader, run=run)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model_distil.state_dict, f=(RUNS_DIR / run_id / 'model_st.pt'))
            run['files/model_st.pt'].upload(str(RUNS_DIR / run_id / 'model_st.pt'), wait=True)
