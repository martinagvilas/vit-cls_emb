import torch

import numpy as np
import pandas as pd

from src.datasets.cifar import MyCIFAR100
from src.datasets.imagenet import ImagenetDatasetS
from src.vis import Vis


MODEL_MAP = {
    'vit_b_32': 'ViT-B/32',
    'vit_b_16': 'ViT-B/16',
    'vit_large_16': 'ViT-L/16',
    'vit_miil_16': 'ViT-B/16-MIIL',
    'vit_cifar_16': 'ViT-B/16-CIFAR',
    'deit_ensemble_16': 'ViT-B/16-Refinement',
    'vit_gap_16': 'ViT-B/16-GAP',
}


def get_class_embed(
        res_path, img_path, model, layer, decoding_type='pos', normalize=False
    ):
    """Get class projection data.

    Parameters
    ----------
    res_path : pathlib.Path
        Path to results.
    img_path : pathlib.Path
        Path to dataset.
    model : str
        Model name. Can be one of the following options: vit_b_16, vit_b_32, 
        vit_large_16, vit_miil_16, vit_cifar_16, deit_ensemble_16, vit_gap_16. 
    layer : str
        Layer index.
    decoding_type : str, optional
        Type of projection, by default 'pos'. Can be one of the following: 
        'pos', 'probs'.
    normalize : bool, optional
        Whether to normalize over number of classes, by default False.

    Returns
    -------
    torch.Tensor
        Class projection data.
    """
    net_path = res_path / 'class_embed'
    
    # Get dataset info
    if 'cifar' in model:
        dataset = MyCIFAR100(img_path)
        stim_info = dataset.stim_info
        concepts = list(stim_info.keys())
    else:
        dataset = ImagenetDatasetS(img_path)
        stim_info = dataset.stim_info
        concepts = stim_info['imagenet_id'].unique()
    
    # Stack decoding info across categories
    dec = []
    for c in concepts:
        f = net_path / model / c / f'{decoding_type}_{layer}.pt'
        dec.append(torch.load(f, map_location='cpu'))
    dec = torch.vstack(dec)
    if normalize == True:
        if 'cifar' in model:
            dec = 1 - (dec / 100)
        else:
            dec = 1 - (dec / 1000)
        return dec
    else:
        return dec


def get_ident_mean_evolution(model_name, dataset_path, res_path):
    """Get mean identifiability evolution.

    Parameters
    ----------
    model_name : str
        Model name. Can be one of the following options: vit_b_16, vit_b_32, 
        vit_large_16, vit_miil_16, vit_cifar_16, deit_ensemble_16, vit_gap_16. 
    dataset_path : pathlib.Path
        Path to dataset.
    res_path : pathlib.Path
        Path to results.

    Returns
    -------
    pandas.DataFrame
        Evolution of mean identifiability across layers.
    """
    if 'large' in model_name:
        n_layers = 24
    else:
        n_layers = 12
    
    df = []
    for b in range(n_layers):
        dec = get_class_embed(
            res_path, dataset_path, model_name, f'hs_{b}', 'pos', normalize=True
        )
        if 'gap' in model_name:
            dec = dec.mean()
            df.append([b+1, 'image', dec])
        else:
            dec = dec.mean(dim=0)
            cls_dec = dec[0].numpy()
            df.append([b+1, '[CLS]', cls_dec])
            token_dec = torch.mean(dec[1:]).numpy()
            df.append([b+1, 'image', token_dec])
    
    df = pd.DataFrame(df, columns=['block', 'token type', 'class identifiability'])
    df['model'] = MODEL_MAP[model_name]

    if model_name == 'vit_large_16':
        new_df = []
        for idx, b in enumerate(np.arange(1, 25, 2)):
            for tt in ['[CLS]', 'image']:
                t_df = df.loc[df['token type'] == tt]
                mean = t_df.loc[t_df['block'].isin([b, b+1])]['class identifiability'].mean()
                new_df.append([idx+1, tt, mean, MODEL_MAP['vit_large_16']])
        new_df = pd.DataFrame(
            new_df, columns=['block', 'token type', 'class identifiability', 'model']
        )
        df = new_df
    
    return df


def get_ident_change_rate(model_name, dataset_path, res_path, n_layers=12):
    """Get change rate of class identifiability across layers.

    Parameters
    ----------
    model_name : str
        Model name. Can be one of the following options: vit_b_16, vit_b_32, 
        vit_large_16, vit_miil_16, vit_cifar_16, deit_ensemble_16, vit_gap_16. 
    dataset_path : pathlib.Path
        Path to dataset.
    res_path : pathlib.Path
        Path to results.
    n_layers : int, optional
        Number of layers in model, by default 12.

    Returns
    -------
    pandas.DataFrame
        Containing the change rate across layers.
    """
    pers = []
    for b in range(1, n_layers):
        i_dec = get_class_embed(res_path, dataset_path, model_name, f'hs_{b-1}', 'probs')[:, 1:]
        j_dec = get_class_embed(res_path, dataset_path, model_name, f'hs_{b}', 'probs')[:, 1:]
        per = (torch.sum((j_dec - i_dec) > 0) / j_dec.flatten().shape[0]).detach().numpy()
        pers.append([b+1, per])
    pers = pd.DataFrame(pers, columns=['block', 'change rate'])
    pers['change rate'] = pers['change rate'].astype('float')
    return pers


def get_ident_segmented(model_name, proj_path, dataset_path):
    """Get class identifiability separately for class- and context-labeled tokens.

    Parameters
    ----------
    model_name : str
        Model name. Can be one of the following options: vit_b_16, vit_b_32, 
        vit_large_16, vit_miil_16, vit_cifar_16, deit_ensemble_16, vit_gap_16. 
    proj_path : pathlib.Path
        Path to source code.
    dataset_path : pathlib.Path
        Path to dataset.

    Returns
    -------
    pandas DataFrame
        Containing class identifiability of class- and context-labeled tokens.
    """
    # Get segmentation annotations
    im = Vis(proj_path, dataset_path, model_name, device='cpu')
    stim_info = im.stim_info
    concepts = stim_info['imagenet_id'].unique().tolist()
    sgts = []
    for c in concepts:
        for i in range(5):
            gt = im.get_segmentation(c, i).flatten()
            sgts.append(gt)
    sgts = np.hstack(sgts)

    # Get identifiability
    if 'large' in model_name:
        n_layers = 24
    else:
        n_layers = 12
    
    # Save in dataframe
    dfs = []
    for b in range(n_layers):
        dec = get_class_embed(
            proj_path / 'results', dataset_path, model_name, f'hs_{b}', 
            decoding_type='pos', normalize=True
        )
        if 'gap' not in model_name:
            dec = dec[:, 1:] # remove cls token
        df = pd.DataFrame(
            {'class identifiability': dec.flatten().detach().numpy(), 'token location': sgts}
        )
        df['block'] = b
        dfs.append(df)
    dfs = pd.concat(dfs)
    dfs['token location'] = dfs['token location'].replace({0: 'context', 1: 'class'})
    
    return dfs


def compute_context_diff(df):
    """Compute significant difference in identifiability between class- and context-labeled tokens

    Parameters
    ----------
    df : pandas DataFrame
        Containing class identifiability of class- and context-labeled tokens.

    Returns
    -------
    tuple
        Containing difference value and pvalue,
    """
    context = df.loc[df['token location'] == 'context']['class identifiability'].to_numpy()
    category = df.loc[df['token location'] == 'class']['class identifiability'].to_numpy()

    # Compute true difference
    true_diff = np.mean(category) - np.mean(context)

    # Compute difference with shuffled labeles
    context_len = context.shape[0]
    all_data = np.concatenate((context, category))
    random_diffs = []
    n_perm = 300
    for _ in range(300):
        np.random.shuffle(all_data)
        context, category = all_data[:context_len], all_data[context_len:]
        random_diffs.append(np.mean(category) - np.mean(context))
    pval = np.sum(true_diff < random_diffs) / n_perm
    
    return true_diff, pval


def compute_class_similarity_change(model_name, block, layer_type, dataset_path, res_path):
    """Compute class similarity change rate of layer.

    Parameters
    ----------
    model_name : str
        Model name. Can be one of the following options: vit_b_16, vit_b_32, 
        vit_large_16, vit_miil_16, vit_cifar_16, deit_ensemble_16, vit_gap_16.
    block : int
        Index of block.
    layer_type : str
        Layer type. Can be one of 'attn' or 'mlp'.
    dataset_path : pathlib.Path
        Path to dataset.
    res_path : pathlib.Path
        Path to results.

    Returns
    -------
    torch.Tensor
        Class similarity change rate.
    """
    dec_layer = get_class_embed(
        res_path, dataset_path, model_name, f'hs-{layer_type}_{block}', 'probs'
    )
    dec = get_class_embed(
        res_path, dataset_path, model_name, f'hs_{block-1}', 'probs'
    )
    return (dec_layer - dec)


def compute_residual_match(model_name, dataset_path, res_path, token_type='all'):
    """Compute match with the predictions of the residual.

    Parameters
    ----------
    model_name : str
        Model name. Can be one of the following options: vit_b_16, vit_b_32, 
        vit_large_16, vit_miil_16, vit_cifar_16, deit_ensemble_16, vit_gap_16.
    dataset_path : pathlib.Path
        Path to dataset.
    res_path : pathlib.Path
        Path to results.
    token_type : str, optional
        Compute match with all tokens, or with cls only, by default 'all'.

    Returns
    -------
    pd.DataFrame
        Match with residual
    """

    if 'large' in model_name:
        n_layers = 24
    else:
        n_layers = 12

    if 'gap' in model_name:
        idxs = torch.arange(196)
    elif token_type == 'cls':
        idxs = 0
    elif (token_type == 'all') & ('32' in model_name):
        idxs = torch.arange(50)
    elif (token_type == 'all') & ('16' in model_name):
        idxs = torch.arange(197)
            
    data = []
    for block in range(1, n_layers):
        # Get residual stream prediction
        topk_b = get_class_embed(
            res_path, dataset_path, model_name, f'hs_{block}', decoding_type='topk'
        )[:, idxs, 0]

        # Get attention layer prediction and compute match
        topk_attn = get_class_embed(
           res_path, dataset_path, model_name, f'hs-attn_{block}', decoding_type='topk'
        )[:, idxs, 0]
        attn_match = torch.sum(topk_b == topk_attn) / topk_b.flatten().shape[0] * 100
        data.append(['attn', block+1, attn_match.detach().numpy()])
        
        # Get MLP layer prediction and compute match
        topk_mlp = get_class_embed(
            res_path, dataset_path, model_name, f'hs-mlp_{block}', decoding_type='topk'
        )[:, idxs, 0]
        mlp_match = torch.sum(topk_b == topk_mlp) / topk_b.flatten().shape[0] * 100
        data.append(['mlp', block+1, mlp_match.detach().numpy()])

        # Get previous block residual stream prediction and compute match
        topk_prev = get_class_embed(
            res_path, dataset_path, model_name, f'hs_{block-1}', decoding_type='topk'
        )[:, idxs, 0]
        prev_match = torch.sum(topk_b == topk_prev) / topk_b.flatten().shape[0] * 100
        data.append(['prev', block+1, prev_match.detach().numpy()])
        
        # Compute rate of tokens that do not match any prediction of the above
        comp_tokens = (topk_b != topk_prev) & (topk_b != topk_attn) & (topk_b != topk_attn)
        comp = torch.sum(comp_tokens) / topk_b.flatten().shape[0] * 100
        data.append(['comp', block+1, comp.detach().numpy()])

    data = pd.DataFrame(data, columns=['Source', 'block', 'match'])
    data['match'] = data['match'].astype('float')
    return data