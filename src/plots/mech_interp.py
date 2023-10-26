import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

from src.identifiability import MODEL_MAP
from src.identifiability import (
    compute_class_similarity_change, compute_residual_match
)
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