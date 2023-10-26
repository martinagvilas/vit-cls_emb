import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from src.identifiability import get_class_embed
from src.linear_probing.prober import LAYERS


def plot_top1_acc(res_path, dataset_path):
    """
    Plot top-1 accuracies of linear probing and cls projection.
    """
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(ncols=2, figsize=(13, 5))

    # Get linear probing accuracies
    accs = []
    for layer in LAYERS:
        file = res_path / 'linear_probing' / layer / 'acc.npy'
        acc = np.load(file)[1:]
        df = pd.DataFrame(acc, columns=['top-1 acc'])
        df['layer'] = layer.split('-')[0]
        df['block'] = int(layer.split('-')[1]) + 1
        accs.append(df)
    accs = pd.concat(accs)
    ## Plot
    sns.lineplot(accs, y='top-1 acc', x='block', hue='layer', ax=axes[0])
    axes[0].set_title('linear probing')
    axes[0].set_xticks(np.arange(1, 13))

    # Get cls projection accuracies
    accs = []
    for layer in LAYERS:
        block = layer.split('-')[1]
        layer_type = layer.split('-')[0]
        if 'attn' in layer:
            layer = f'hs-attn_{block}'
        elif 'mlp' in layer:
            layer = f'hs-mlp_{block}'
        else:
            layer = f'hs_{block}'
        dec = get_class_embed(
            res_path, dataset_path, 'vit_b_32', layer, 'pos', normalize=False
        )
        dec = dec[:, 1:]
        acc = torch.sum((dec == 0), axis=0) / dec.shape[0]
        df = pd.DataFrame(acc.detach().numpy(), columns=['top-1 acc'])
        df['layer'] = layer_type
        df['block'] = int(block) + 1
        accs.append(df)
    accs = pd.concat(accs)
    ## Plot
    sns.lineplot(accs, y='top-1 acc', x='block', hue='layer', ax=axes[1])
    axes[1].set_title('cls projection')
    axes[1].set_xticks(np.arange(1, 13))

    plt.tight_layout()
    plt.show()
    return


def plot_perturbation(res_path, model_name='vit_b_32'):
    """
    Plot and compare perturbation experiments.
    """
    plt.rcParams.update({'font.size': 12})

    labels = torch.arange(48) / 49 * 100

    fig, ax = plt.subplots(figsize=(6,4))

    # Plot our method
    f = res_path / 'perturbation' / model_name / 'negative_grads.pt'
    neg_perturb = torch.load(f, map_location='cpu')
    neg_perturb = torch.flip(neg_perturb, dims=(0,)).detach().numpy()
    sns.lineplot(x=labels, y=neg_perturb, ax=ax, label='NEG cls-based removal')
    print(f'negative AUC emb: {np.sum(neg_perturb) / (np.max(neg_perturb) * 49)}')

    # Plot linear probing
    f = res_path / 'perturbation' / model_name / 'negative_linear-probe.pt'
    linear_perturb = torch.load(f, map_location='cpu')
    linear_perturb = torch.flip(linear_perturb, dims=(0,)).detach().numpy()
    sns.lineplot(x=labels, y=linear_perturb, ax=ax, label='NEG probe removal')
    print(f'negative AUC linear: {np.sum(linear_perturb) / (np.max(linear_perturb) * 49)}')

    # Plot our method
    f = res_path / 'perturbation' / model_name / 'positive_grads.pt'
    neg_perturb = torch.load(f, map_location='cpu')
    neg_perturb = torch.flip(neg_perturb, dims=(0,)).detach().numpy()
    sns.lineplot(x=labels, y=neg_perturb, ax=ax, label='POS cls-based removal')
    print(f'positive AUC emb: {np.sum(neg_perturb) / (np.max(neg_perturb) * 49)}')

    # Plot linear probing
    f = res_path / 'perturbation' / model_name / 'positive_linear_probe.pt'
    linear_perturb = torch.load(f, map_location='cpu')
    linear_perturb = torch.flip(linear_perturb, dims=(0,)).detach().numpy()
    sns.lineplot(x=labels, y=linear_perturb, ax=ax, label='POS probe removal')
    print(f'positive AUC linear: {np.sum(linear_perturb) / (np.max(linear_perturb) * 49)}')

    # Plot random perturbation
    f = res_path / 'perturbation' / model_name / 'perturb_random.pt'
    rand_perturb = torch.mean(torch.load(f, map_location='cpu'), dim=0)
    rand_perturb = torch.flip(rand_perturb, dims=(0,)).detach().numpy()
    sns.lineplot(x=labels, y=rand_perturb, ax=ax, label='random removal')
    print(f'random AUC: {np.sum(rand_perturb) / (np.max(rand_perturb) * 49)}')


    ax.hlines(
        xmin=-0.5, xmax=100.5, y=neg_perturb[0], colors='dimgray', linestyles='--', lw=2, 
        label='baseline accuracy'
    )
    ax.set_xticks(np.arange(0, 100, 10))
    ax.set_xlim(-0.5, 95)
    ax.set_ylim(0,0.9)
    ax.set_xlabel('percentage of tokens removed')
    ax.set_ylabel('accuracy')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()
    return


def plot_all_linear_probing(res_path, dataset_path):
    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(ncols=3, figsize=(13, 4))

    ## ACCURACY
    accs = []
    for layer in LAYERS:
        file = res_path / 'linear_probing' / layer / 'acc.npy'
        acc = np.load(file)[1:]
        df = pd.DataFrame(acc, columns=['top-1 acc'])
        df['layer'] = layer.split('-')[0]
        df['block'] = int(layer.split('-')[1]) + 1
        accs.append(df)
    accs = pd.concat(accs)
    sns.lineplot(accs, y='top-1 acc', x='block', hue='layer', ax=axes[0])
    axes[0].set_title('linear probing')
    axes[0].set_xticks(np.arange(1, 13))

    accs = []
    for layer in LAYERS:
        block = layer.split('-')[1]
        layer_type = layer.split('-')[0]
        if 'attn' in layer:
            layer = f'hs-attn_{block}'
        elif 'mlp' in layer:
            layer = f'hs-mlp_{block}'
        else:
            layer = f'hs_{block}'
        dec = get_class_embed(
            res_path, dataset_path, 'vit_b_32', layer, 'pos', normalize=False
        )
        dec = dec[:, 1:]
        acc = torch.sum((dec == 0), axis=0) / dec.shape[0]
        df = pd.DataFrame(acc.detach().numpy(), columns=['top-1 acc'])
        df['layer'] = layer_type
        df['block'] = int(block) + 1
        accs.append(df)
    accs = pd.concat(accs)
    sns.lineplot(accs, y='top-1 acc', x='block', hue='layer', ax=axes[1])
    axes[1].set_title('cls projection')
    axes[1].set_xticks(np.arange(1, 13))

    ## PERTURBATION
    model_name = 'vit_b_32'

    labels = torch.arange(48) / 49 * 100

    # Plot our method
    f = res_path / 'perturbation' / model_name / 'negative_grads.pt'
    neg_perturb = torch.load(f, map_location='cpu')
    neg_perturb = torch.flip(neg_perturb, dims=(0,)).detach().numpy()
    sns.lineplot(x=labels, y=neg_perturb, ax=axes[2], label='NEG cls-based removal')

    # Plot linear probing
    f = res_path / 'perturbation' / model_name / 'negative_linear-probe.pt'
    linear_perturb = torch.load(f, map_location='cpu')
    linear_perturb = torch.flip(linear_perturb, dims=(0,)).detach().numpy()
    sns.lineplot(x=labels, y=linear_perturb, ax=axes[2], label='NEG probe removal')

    # Plot our method
    f = res_path / 'perturbation' / model_name / 'positive_grads.pt'
    neg_perturb = torch.load(f, map_location='cpu')
    neg_perturb = torch.flip(neg_perturb, dims=(0,)).detach().numpy()
    sns.lineplot(x=labels, y=neg_perturb, ax=axes[2], label='POS cls-based removal')

    # Plot linear probing
    f = res_path / 'perturbation' / model_name / 'positive_linear_probe.pt'
    linear_perturb = torch.load(f, map_location='cpu')
    linear_perturb = torch.flip(linear_perturb, dims=(0,)).detach().numpy()
    sns.lineplot(x=labels, y=linear_perturb, ax=axes[2], label='POS probe removal')

    # Plot random perturbation
    f = res_path / 'perturbation' / model_name / 'perturb_random.pt'
    rand_perturb = torch.mean(torch.load(f, map_location='cpu'), dim=0)
    rand_perturb = torch.flip(rand_perturb, dims=(0,)).detach().numpy()
    sns.lineplot(x=labels, y=rand_perturb, ax=axes[2], label='random removal')

    ax = axes[2]
    ax.hlines(
        xmin=-0.5, xmax=100.5, y=neg_perturb[0], colors='dimgray', linestyles='--', lw=2, 
        label='baseline accuracy'
    )
    ax.set_xticks(np.arange(0, 100, 10))
    ax.set_xlim(-0.5, 95)
    ax.set_ylim(0,0.9)
    ax.set_xlabel('percentage of tokens removed')
    ax.set_ylabel('accuracy')
    ax.legend(prop={'size': 10})

    plt.tight_layout()
    f = res_path / 'figures' / f'compare_linear.png'
    plt.savefig(f, dpi=300)
    plt.show()
    return