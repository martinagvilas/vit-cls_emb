import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns
import torch

from src.identifiability import MODEL_MAP
from src.identifiability import (
    get_class_embed, get_ident_change_rate, get_ident_mean_evolution,
    get_ident_segmented, compute_context_diff
)
from src.vis import Vis


def print_identifiability(model_name, res_path, dataset_path, layer=11):
    """
    Compute and print class identifiability measures.
    """
    # Get class identifiability
    dec = get_class_embed(res_path, dataset_path, model_name, f'hs_{layer}', 'pos')[:, 1:]
    
    # Compute class identifiability rate
    avg_percentage = (torch.sum(dec == 0, dim=1) / dec.shape[1]).float().mean() * 100
    print(f'- Average percentage of class identifiable image tokens per image: {avg_percentage}')
    
    # Compute percentage of images with at least one identifiable token
    image_percentage = torch.sum((torch.sum((dec == 0), dim=1) > 0)) / dec.shape[0] * 100
    print(
        f'- Percentage of images with at least one class identifiable token: {image_percentage}'
    )
    
    return


def label_token(row):
    """
    Assign to token its correct label.
    """
    if row['token'] == 0:
        return '[CLS]'
    else:
        return 'image'


def plot_identifiability_evolution(res_path, dataset_path):
    """
    Plot identifiability evolution over blocks.
    """
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(nrows=7, figsize=(13, 30))
    for model_name, ax in zip(MODEL_MAP.keys(), axes.flat):
        if 'large' in model_name:
            n_layers = 24
        else:
            n_layers = 12
        
        dfs = []
        pvals = []
        for b in range(n_layers):
            # Get class identifiability of block
            dec = get_class_embed(
                res_path, dataset_path, model_name, f'hs_{b}', 'pos', normalize=True
            )
            
            # Compare to random model
            if 'miil' in model_name:
                random_model = f'vit_b_16_random'
            else:
                random_model = f'{model_name}_random'
            rand_dec = get_class_embed(
                res_path, dataset_path, random_model, f'hs_{b}', 'pos', normalize=True
            )
            wx = wilcoxon(dec.flatten(), rand_dec.flatten(), alternative='greater')
            pvals.append(np.round(wx.pvalue, 3))
            
            # Add to df
            df = (
                pd.DataFrame(pd.DataFrame(dec).stack())
                .reset_index(names=['image', 'token'])
                .rename(columns={0:'class identifiability'})
            )
            if 'gap' not in model_name:
                df['token label'] = df.apply(lambda row: label_token(row), axis=1)
            df['block'] = b + 1
            dfs.append(df)
        dfs = pd.concat(dfs)
        
        # Plot
        if 'gap' not in model_name:
            sns.boxplot(
                dfs, x='block', y='class identifiability', hue='token label', 
                fliersize=0, palette=sns.color_palette("Set2"), ax=ax
            )
            sns.move_legend(ax, 'lower right')
        else:
            sns.boxplot(
                dfs, x='block', y='class identifiability', fliersize=0, 
                palette=sns.color_palette(['#fc8d62']), ax=ax
            )
        ax.hlines(xmin=-1, xmax=n_layers, y=0.5, colors='dimgray', linestyles='--', lw=2)
    
        # Add significance stars
        sig = np.where(np.array(pvals) < 0.05)[0]
        ax.scatter(sig, [1.1] * len(sig), marker='*', c='grey', s=45)
    
        ax.set_xlim(-0.5, int(n_layers))
        ax.set_title(f'{MODEL_MAP[model_name]}')
        
    plt.tight_layout()
    f = res_path / 'figures' / f'class_identifiability_evolution.png'
    plt.savefig(f, dpi=300)
    plt.show()
    
    return


def compare_identifiability_evolution(res_path, dataset_path, cifar_path):
    # Get evolution
    df = []
    for model in MODEL_MAP.keys():
        if 'cifar' in model:
            ident = get_ident_mean_evolution(model, cifar_path, res_path)
        else:
            ident = get_ident_mean_evolution(model, dataset_path, res_path)
        df.append(ident)
    df = pd.concat(df)
    df['class identifiability'] = df['class identifiability'].astype('float')

    # Plot
    plt.rcParams.update({'font.size': 17})
    fig, axes = plt.subplots(ncols=2, figsize=(13, 3.5))

    cls_df = df.loc[df['token type'] == '[CLS]']
    sns.lineplot(
        cls_df, x='block', y='class identifiability', hue='model', 
        ax=axes[0], marker='*', markersize=10
    )
    axes[0].set_title('[CLS] token')

    img_df = df.loc[df['token type'] == 'image']
    sns.lineplot(
        img_df, x='block', y='class identifiability', hue='model', 
        ax=axes[1], marker='*', markersize=10
    )
    axes[1].set_title('image tokens', fontsize=20)

    for ax in axes.flat:
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel('class identifiability', fontsize=20)
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(np.arange(1, 13))
        ax.set_xticklabels([])
        ax.set_xlim((0.8, 12.2))
        ax.set_xlabel('normalized block', fontsize=20)
        ax.hlines(
            xmin=0.8, xmax=12.2, y=0.5, colors='dimgray', linestyles='--', lw=2, 
            label='Chance Level'
        )

    # Unify legend
    lines, labels = axes[1].get_legend_handles_labels()
    lgd = fig.legend(
        lines, labels, loc='center right', #nrow=7,
        bbox_to_anchor=(1.15, 0.5),
    )
    for ax in axes.flat:
        ax.get_legend().remove()

    f = res_path / 'figures' / f'mean_evolution.png'
    plt.savefig(f, dpi=300, bbox_inches='tight')
    plt.show()
    return


def plot_logits_increment(res_path, dataset_path):
    """
    Plot percentage of image tokens that increment the logits of the correct class.
    """
    
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(13, 13))
    for model_name, ax in zip(MODEL_MAP.keys(), axes.flat):
        
        if 'large' in model_name:
            n_layers = 24
        else:
            n_layers = 12
        
        # Get change rate 
        pers = get_ident_change_rate(model_name, dataset_path, res_path, n_layers)

        # Plot
        sns.lineplot(pers, x='block', y='change rate', marker='o', markersize=9, ax=ax)
        ax.set_xlim(1.8, int(n_layers) + 0.2)
        ax.set_xticks(np.arange(2, int(n_layers) + 1))
        ax.xaxis.set_tick_params(labelsize=10)
        ax.set_ylim(0.5, 1)
        ax.set_yticks(np.arange(0.6, 1, 0.1))
        ax.yaxis.set_tick_params(labelsize=12)
        ax.set_title(MODEL_MAP[model_name])
    
    fig.delaxes(axes[3,1])
        
    plt.tight_layout()
    f = res_path / 'figures' / f'prob_increment.png'
    plt.savefig(f, dpi=300)
    plt.show()

    return


def print_attn_perturb(model_name, perturb_type, res_path, dataset_path):
    """
    Print class identifiability of attention perturbation studies.
    """
    # Get perturbed identifiability
    f = res_path / 'perturbation' / model_name / f'attn-{perturb_type}_dec.pt'
    attn_dec = torch.load(f, map_location='cpu')
    attn_dec = 1 - (attn_dec / 1000)
    attn_dec_acc = torch.sum(attn_dec == 1) / attn_dec.flatten().shape[0] * 100
    print(f'Class identifiability in {perturb_type}: {attn_dec_acc}')

    # Get unperturbed identifiability
    dec = get_class_embed(
        res_path, dataset_path, model_name, 'hs_11', decoding_type='pos', normalize=True
    )[:, 1:]
    dec_acc = torch.sum(dec == 1) / dec.flatten().shape[0] * 100
    print(f'Class identifiability in unperturbed model: {dec_acc}')
    
    return


def plot_context_diff(model_name, proj_path, dataset_path):
    """
    Plot identifiability evolution separately for class- and context-labeled tokens.
    """
    dfs = get_ident_segmented(model_name, proj_path, dataset_path)

    if 'large' in model_name:
        nrows = 4
        n_layers = 24
        height = 10
    else:
        nrows = 2
        n_layers = 12
        height = 5

    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(nrows=nrows, ncols=6, figsize=(13, height))
    for ax, b in zip(axes.flat, np.arange(n_layers)):
        df = dfs.loc[dfs['block'] == b]
        _, pval = compute_context_diff(df)
        sns.boxplot(
            df, y='class identifiability', x='token location',
            fliersize=0, ax=ax
        )
        if pval < 0.05:
            ax.set_title(f'block {b+1} *')
        else:
            ax.set_title(f'block {b+1}')
        ax.set_xlabel('')
    for ax in axes.flat[1:]:
        ax.set_ylabel('')
    fig.suptitle(MODEL_MAP[model_name])
    
    plt.tight_layout()
    f = proj_path / 'results/figures' / f'context_{model_name}.png'
    plt.savefig(f, dpi=300)
    plt.show()
    
    return


def print_context_perturb(model_name, mask_type, proj_path, dataset_path):
    """
    Print class identifiability of context perturbation studies.
    """
    # Get identifiability of perturbed model
    res_path = proj_path / 'results'
    f = res_path / 'perturbation' / model_name / f'no_{mask_type}_tokens_dec.pt'
    context_dec = torch.load(f, map_location='cpu')['hs_11']
    context_dec_acc = []
    for i in context_dec:
        i = 1 - (i / 1000)
        context_dec_acc.append(torch.sum(i == 1) / i.shape[0])
    context_dec_acc = torch.mean(torch.stack(context_dec_acc)) * 100
    print(f'Class identifiability with no {mask_type} tokens: {context_dec_acc}')

    # Get identifiability of unperturbed model
    im = Vis(proj_path, dataset_path, model_name, device='cpu')
    stim_info = im.stim_info
    concepts = stim_info['imagenet_id'].unique().tolist()
    sgts = []
    for c in concepts:
        for i in range(5):
            gt = im.get_segmentation(c, i).flatten()
            sgts.append(gt)
    sgts = np.hstack(sgts)
            
    dec = get_class_embed(
        res_path, dataset_path, model_name, 'hs_11', decoding_type='pos', normalize=True
    )[:, 1:]
    dec = dec.flatten()
    if mask_type == 'context':
        dec = dec[(sgts == 1).nonzero()]
    elif mask_type == 'class_label':
        dec = dec[(sgts == 0).nonzero()]
    dec_acc = torch.sum(dec == 1) / dec.shape[0] * 100
    print(f'Class identifiability in unperturbed model: {dec_acc}')
    
    return