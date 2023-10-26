import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

from src.identifiability import MODEL_MAP
from src.identifiability import (
    compute_class_similarity_change, compute_residual_match
)
from src.memories import Memory, compute_key_value_agr_rate
from src.vis import Vis


def plot_class_building(model_name, res_path, dataset_path):
    """
    Plot categorical building over blocks and layers.
    """
    plt.rcParams.update({'font.size': 16})

    if 'large' in model_name:
        n_layers = 24
    else:
        n_layers = 12

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,5))
    
    # Plot class similarity change rate
    pers = []
    for layer_type in ['attn', 'mlp']:
        for b in range(1, n_layers):
            
            diff = compute_class_similarity_change(
                model_name, b, layer_type, dataset_path, res_path
            )

            if 'gap' not in model_name:
                diff_cls = diff[:, 0]
                cls_change_rate = torch.sum(diff_cls > 0) / diff_cls.flatten().shape[0]
                pers.append([f'{layer_type} - [CLS] ', b+1, cls_change_rate.detach().numpy()])
            
                diff_img = diff[:, 1:]
                diff_change_rate = torch.sum(diff_img > 0) / diff_img.flatten().shape[0]
                pers.append([f'{layer_type} - Image', b+1, diff_change_rate.detach().numpy()])
            
            else:
                diff_img = diff
                diff_change_rate = torch.sum(diff_img > 0) / diff_img.flatten().shape[0]
                pers.append([f'{layer_type} - Image', b+1, diff_change_rate.detach().numpy()])
    
    pers = pd.DataFrame(pers, columns=['Layer Type', 'block', 'change rate'])
    pers['change rate'] = pers['change rate'].astype('float')

    if model_name == 'vit_gap_16':
        colors = ["#1f78b4", "#33a02c"]
        sns.lineplot(
            pers, x='block', y='change rate', hue='Layer Type', marker='*', ax=axes[0],
            palette=sns.color_palette(colors)
        )
    else:
        sns.lineplot(
            pers, x='block', y='change rate', hue='Layer Type', marker='*', ax=axes[0],
            palette=sns.color_palette("Paired", 4)
        )
    axes[0].hlines(xmin=-1, xmax=n_layers, y=0.5, colors='dimgray', linestyles='--',  lw=2)
    
    axes[0].set_xlim(1.9, int(n_layers) + 0.1)
    axes[0].set_xticks(np.arange(2, n_layers + 1))
    axes[0].xaxis.set_tick_params(labelsize=10)
    axes[0].set_ylim(0, 1)
    
    axes[0].legend(loc='lower left').set_title('')
    axes[0].set_title('class similarity change rate')
    
    # Plot residual composition
    match = compute_residual_match(model_name, dataset_path, res_path)
    match = match.pivot(index='block', columns=['Source'])
    match.columns = match.columns.droplevel()
    match = match[['prev', 'attn', 'mlp', 'comp']]
    match.plot(kind='bar', stacked=True, rot=0, ax=axes[1], cmap="Set2")
    
    axes[1].set_ylabel('% of tokens')
    axes[1].xaxis.set_tick_params(labelsize=10)
    axes[1].legend(
        ['previous', 'attn','mlp', 'composition'], loc='upper left', ncol=2
    )
    axes[1].set_title('residual stream composition')

    fig.suptitle(MODEL_MAP[model_name])
    plt.tight_layout()
    f = res_path / 'figures' / f'compositionality_{model_name}.png'
    plt.savefig(f, dpi=300)
    plt.show()

    return


def get_segmented_increases(model_name, layer_type, proj_path, dataset_path):
    """
    Compute class similarity change separately for class- and context-labeled tokens.
    """
    # Get segmentations
    im = Vis(proj_path, dataset_path, model_name, device='cpu')
    stim_info = im.stim_info
    concepts = stim_info['imagenet_id'].unique().tolist()
    sgts = []
    for c in concepts:
        for i in range(5):
            gt = im.get_segmentation(c, i).flatten()
            sgts.append(gt)
    sgts = np.hstack(sgts)

    # Compute increases
    if 'large' in model_name:
        n_layers = 24
    else:
        n_layers = 12
        
    dfs = []
    for b in range(1, n_layers):
        res_path = proj_path / 'results'
        dec = compute_class_similarity_change(model_name, b, layer_type, dataset_path, res_path)
        if 'gap' not in model_name:
            dec = dec[:, 1:]
        df = pd.DataFrame({'decoding': dec.flatten().detach().numpy(), 'token location': sgts})
        df['block'] = b
        dfs.append(df)
    dfs = pd.concat(dfs)
    dfs['token location'] = dfs['token location'].replace({0: 'context', 1: 'category'})
    return dfs


def compute_context_diff(model_name, layer_type, proj_path, dataset_path):
    """
    Compare class similarity change between class- and context-labeled tokens.
    """
    if 'large' in model_name:
        n_layers = 24
    else:
        n_layers = 12
        
    dfs = get_segmented_increases(model_name, layer_type, proj_path, dataset_path)
    diffs = []
    pvals = []
    for b in range(1, n_layers):
        df = dfs.loc[dfs['block'] == b]
        context = df.loc[df['token location'] == 'context']['decoding'].to_numpy()
        category = df.loc[df['token location'] == 'category']['decoding'].to_numpy()

        # Compute true difference
        true_diff = np.mean(category) - np.mean(context)

        # Compare to random model
        context_len = context.shape[0]
        all_data = np.concatenate((context, category))
        random_diffs = []
        n_perm = 300
        for p in range(300):
            np.random.shuffle(all_data)
            context, category = all_data[:context_len], all_data[context_len:]
            random_diffs.append(np.mean(category) - np.mean(context))
        pval = np.sum(true_diff < random_diffs) / n_perm
            
        diffs.append(true_diff)
        pvals.append(pval)
    
    df = pd.DataFrame({'diffs': diffs, 'pvals': pvals})
    df['model'] = model_name
    df['layer'] = layer_type
    df['block'] = np.arange(len(diffs))

    return df


def plot_categorical_updates(proj_path, dataset_path):
    """
    Plot categorical updates for all models.
    """
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(10, 20))
    for m_idx, model in enumerate(MODEL_MAP.keys()):
        mem = Memory(proj_path, dataset_path, model)
        
        for l_idx, layer_type in enumerate(['attn', 'mlp']):
            df, pvals_l, _ = mem.compute_class_value_agr(layer_type)
        
            # Plot pvalues
            sig = np.where(np.array(pvals_l) < 0.05)[0]
            axes[m_idx, l_idx].scatter(sig, [0.8] * len(sig), marker='*', c='grey', s=50)
        
            # Plot agreement values
            sns.boxplot(df, x='block', y='top-1 logits', ax=axes[m_idx, l_idx])
            axes[m_idx, l_idx].set_ylim((0,0.85))
            axes[m_idx, l_idx].xaxis.set_tick_params(labelsize=9)
            axes[m_idx, l_idx].yaxis.set_tick_params(labelsize=12)
            
            axes[m_idx, l_idx].set_title(f'{MODEL_MAP[model]} - {layer_type}')

    plt.tight_layout()
    f = proj_path / 'results/figures' / f'match_score.png'
    plt.savefig(f, dpi=300)
    plt.show()
    return


def plot_key_value_agreement_rate(res_path):
    """
    Plot key-value agreement rate.
    """
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(10, 20))
    for m_idx, model_name in enumerate(MODEL_MAP.keys()):
        for l_idx, layer_type in enumerate(['attn', 'mlp']):
            df = compute_key_value_agr_rate(model_name, layer_type, res_path)
            sns.stripplot(
                df, x='block', y='agreement rate', alpha=0.3, zorder=1,
                ax=axes[m_idx, l_idx]
            )
            sns.pointplot(
                df, x='block', y='agreement rate', color='red',
                markers="d", scale=.75, errorbar=None, ax=axes[m_idx, l_idx]
            )
            axes[m_idx, l_idx].set_title(f'{MODEL_MAP[model_name]} - {layer_type}')
            axes[m_idx, l_idx].set_ylim((0,100))
            
            axes[m_idx, l_idx].xaxis.set_tick_params(labelsize=9)
            axes[m_idx, l_idx].yaxis.set_tick_params(labelsize=12)

    plt.tight_layout()
    f = res_path / 'figures' / f'key_val_agr.png'
    plt.savefig(f, dpi=300)
    plt.show()
    return


def compare_memory_pairs(proj_path, dataset_path):
    """
    Compare memory results across ViT variants.
    """
    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(13, 3))

    # Plot class-value agreement
    for layer_type, ax in zip(['attn', 'mlp'], axes.flat[:2]):
        dfs = []
        random_k = []
        for model_name in MODEL_MAP.keys():
            mem = Memory(proj_path, dataset_path, model_name)
            df, _, rk = mem.compute_class_value_agr(layer_type)
            random_k.append(rk)
        
            if model_name == 'vit_large_16':
                new_df = []
                for idx, b in enumerate(np.arange(1, 25, 2)):
                    mean = df.loc[df['block'].isin([b, b+1])]['top-1 logits'].mean()
                    new_df.append([mean, idx+1, MODEL_MAP['vit_large_16']])
                new_df = pd.DataFrame(
                    new_df, columns=['top-1 logits', 'block', 'model']
                )
                df = new_df
            else:
                df['model'] = MODEL_MAP[model_name]
            
            dfs.append(df)

        dfs = pd.concat(dfs)
        sns.lineplot(
            dfs, x='block', y='top-1 logits', hue='model', ax=ax,
            marker='*', markersize=10
        )
        ax.set_title(layer_type)
        ax.set_ylim((0.08, 0.45))
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels([])
        ax.set_xlim((0.8, 12.2))
        ax.set_xlabel('normalized layer')

    axes[0].set_ylabel('class-value agreement')
    axes[1].set_ylabel('')

    # Plot key-value agreement rate
    for layer_type, ax in zip(['attn', 'mlp'], axes[2:].flat):
        dfs = []
        for model_name in MODEL_MAP.keys():
            res_path = proj_path / 'results'
            df = compute_key_value_agr_rate(model_name, layer_type, res_path)
        
            if model_name == 'vit_large_16':
                new_df = []
                for idx, b in enumerate(np.arange(1, 25, 2)):
                    mean = df.loc[df['block'].isin([b, b+1])]['agreement rate'].mean()
                    new_df.append([mean, idx+1, MODEL_MAP['vit_large_16']])
                new_df = pd.DataFrame(
                    new_df, columns=['agreement rate', 'block', 'model']
                )
                df = new_df
            else:
                df['model'] = MODEL_MAP[model_name]
            
            dfs.append(df)

        dfs = pd.concat(dfs)
        sns.lineplot(
            dfs, x='block', y='agreement rate', hue='model', ax=ax,
            marker='*', markersize=10
        )
        ax.set_title(layer_type)
        ax.set_ylim((0,85))
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels([])
        ax.set_xlim((0.8, 12.2))
        ax.set_xlabel('normalized layer')

    axes[2].set_ylabel('key-value agreement rate')
    axes[3].set_ylabel('')

    lines, labels = axes[1].get_legend_handles_labels()
    lgd = fig.legend(
        lines, labels, loc='upper center', ncol=4,
        bbox_to_anchor=(0.5, 1.25),
    )

    for ax in axes.flat:
        ax.get_legend().remove()

    plt.tight_layout()
    f = proj_path / 'results/figures' / f'compare_all.png'
    plt.savefig(f, dpi=300, bbox_inches='tight')
    plt.show()
    return


def plot_agr_rate_diff(res_path, device='cpu'):
    """
    Plot difference in key-value agreement rate between accurate and non-accurate samples.
    """
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(10,20))
    for m_idx, model_name in enumerate(MODEL_MAP.keys()):
        for l_idx, layer_type in enumerate(['attn', 'mlp']):
            if 'large' in model_name:
                n_layers = 24
            else:
                n_layers = 12
            
            # Get accuracy
            acc_file = res_path / 'class_embed'/ model_name / 'acc.pt'
            acc = torch.load(acc_file, map_location=device)
            
            diffs = []
            pvals = []
            for b in range(n_layers):
                # Compute difference in accuracy
                f = res_path / 'memories/' / model_name / f'{layer_type}_{b}_value-class_agreement.pt'
                agr = torch.load(f, map_location='cpu')[:,:, 0].flatten()
            
                acc_1 = agr[acc==1].float()
                agr_acc = torch.sum(acc_1) / acc_1.shape[0] * 100
                acc_2 = agr[acc==0].float()
                agr_inacc = torch.sum(acc_2) / acc_2.shape[0] * 100
            
                true_diff = agr_acc - agr_inacc
                diffs.append(true_diff.detach().numpy())
                
                # Compare to random model
                random_diffs = []
                for p in range(300):
                    rand_idxs = torch.randperm(len(acc))
                    r_acc_1 = agr[rand_idxs[:len(acc_1)]].float()
                    r_agr_acc = torch.sum(r_acc_1) / r_acc_1.shape[0] * 100
                    r_acc_2 = agr[rand_idxs[len(acc_1):]].float()
                    r_agr_inacc = torch.sum(r_acc_2) / r_acc_2.shape[0] * 100
                    
                    random_diffs.append(r_agr_acc - r_agr_inacc)
            
                pval = torch.sum(true_diff < torch.stack(random_diffs)) / 300
                pvals.append(pval.detach().numpy())
            
            df = pd.DataFrame({'diff': diffs, 'block': np.arange(1, n_layers+1)})
            df['model'] = model_name
            df['layer'] = layer_type
            df['diff'] = df['diff'].astype('float')
        
            # Plot pvalues
            sig = np.where(np.array(pvals) < 0.05)[0] + 1
            axes[m_idx, l_idx].scatter(sig, [69] * len(sig), marker='*', c='grey', s=50)
            
            sns.lineplot(df, x='block', y='diff', marker='o', ax=axes[m_idx, l_idx])
            axes[m_idx, l_idx].set_ylim(-5, 72)
            axes[m_idx, l_idx].set_xticks(np.arange(1, n_layers+1))
            axes[m_idx, l_idx].set_xlim((0.8, n_layers + 0.2))
        
            axes[m_idx, l_idx].hlines(
                xmin=1, xmax=n_layers+1, y=0, colors='dimgray', linestyles='--',  lw=2
            )

            axes[m_idx, l_idx].set_ylabel('difference (%)')
            axes[m_idx, l_idx].xaxis.set_tick_params(labelsize=9)
            axes[m_idx, l_idx].yaxis.set_tick_params(labelsize=12)

            axes[m_idx, l_idx].set_title(f'{MODEL_MAP[model_name]} - {layer_type}')

    plt.tight_layout()
    f = res_path / 'figures' / f'acc_agr_rate.png'
    plt.savefig(f, dpi=300)
    plt.show()
    
    return


def plot_memory_compositionality(res_path):
    """
    Plot compositionality of memory pairs.
    """
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(13, 15))
    for model_name, ax in zip(MODEL_MAP.keys(), axes.flat):
        if 'large' in model_name:
            n_layers = 24
        else:
            n_layers = 12
        
        # Get compositionality
        comps = []
        for layer_type in ['attn', 'mlp']:
            for b in range(n_layers):
                f = res_path / 'memories' / model_name / f'{layer_type}_{b}_top5_pred-match.pt'
                comp = torch.load(f, map_location='cpu')#[:, :, 0]
                preds = torch.any(comp, dim=-1)
                per = torch.sum(torch.any(comp, dim=-1)) / preds.flatten().shape[0] * 100
                comps.append([(b+1), layer_type, per.detach().numpy()])
        
        comps = pd.DataFrame(comps, columns=['block', 'layer type', 'percentage'])
        comps['percentage'] = comps['percentage'].astype('float')
        
        # Plot
        sns.lineplot(comps, x='block', y='percentage', hue='layer type', marker='o', ax=ax)
        ax.set_xticks(np.arange(1, n_layers+1))
        ax.xaxis.set_tick_params(labelsize=10)
        ax.set_xlim(1, n_layers)
        ax.set_yticks(np.arange(0, 75, 10))
        ax.yaxis.set_tick_params(labelsize=12)
        ax.set_ylim((0, 75))
        ax.set_ylabel('%')
        
        ax.legend(loc='upper left')
        ax.set_title(MODEL_MAP[model_name])

    fig.delaxes(axes[3,1])

    plt.tight_layout()
    f = res_path / 'figures' / f'composition.png'
    plt.savefig(f, dpi=300)
    plt.show()
    return