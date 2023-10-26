import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.gradients import ViTGradients
from src.vis import Vis


def plot_specific_ft_importance(
    model_name, block, head, imgs_info, stim_info, proj_path, dataset_path
):
    """
    Plot block- and head-specific feature importance results.
    """
    # Get gradients
    vis = Vis(proj_path, dataset_path, model_name, 'cpu')
    g = ViTGradients(model_name, proj_path, dataset_path)

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    for i, ax in zip(imgs_info, axes.flat):
        # Compute gradients
        grads, _ = g.compute(i[0], i[1], i[2])
        mask = - grads[block][head, 0, 1:]
        mask_vis = vis.mask(i[0], i[1], mask)
        
        ax.imshow(mask_vis)
        ax.set_xticks([])
        ax.set_yticks([])
        cat = stim_info[stim_info['index'] == i[2]]['cat'].unique()[0].split(',')[0].lower()
        ax.set_title(cat)
    
    f = proj_path / 'results/figures' / f'explainability_{model_name}_head_1.png'
    plt.tight_layout()
    plt.savefig(f, dpi=300)
    plt.show()
    return


def plot_sum_ft_importance(model_name, imgs_info, stim_info, proj_path, dataset_path):
    """
    Plot sum of gradients feature importance results.
    """
    # Visualize feature importance
    if len(imgs_info) <= 6:
        fig, axes = plt.subplots(nrows=1, ncols=len(imgs_info), figsize=(10, 3))
    else:
        fig, axes = plt.subplots(nrows=2, ncols=int(len(imgs_info)/2), figsize=(12, 5))
    
    for i, ax in zip(imgs_info, axes.flat):
        
        # Compute importance by gradients
        grads, _ = ViTGradients(model_name, proj_path, dataset_path).compute(i[0], i[1], i[2])
        importance = []
        for b in range(12):
            importance.append(torch.sum(grads[b], dim=0)[0])
        mask = torch.sum(torch.stack(importance), dim=0)[1:]
        mask = - mask

        # Plot heatmap over image
        vis = Vis(proj_path, dataset_path, model_name, 'cpu')
        mask_vis = vis.mask(i[0], i[1], mask)
        ax.imshow(mask_vis)
        ax.set_xticks([])
        ax.set_yticks([])
        if i[2] == 582:
            cat = 'grocery store'
        else:
            cat = stim_info[stim_info['index'] == i[2]]['cat'].unique()[0].split(',')[0].lower()
        ax.set_title(cat)

    f = proj_path / 'results/figures' / f'explainability_{model_name}.png'
    plt.tight_layout()
    plt.savefig(f, dpi=300)
    plt.show()
    
    return


def compare_explainability(proj_path, dataset_path, stim_info):
    model_name = 'vit_b_32'

    # Select different images, classes, blocks and head
    img_info = [
        ['n02422699', 0, 352, 7, 0], ['n02422699', 0, 350, 7, 0],
        ['n04404412', 1, 851, 11, 6], ['n04404412', 1, 831, 11, 6],
    ] # [imagenet_id, image_id, class_id, block, attention head]

    # Compute gradients
    vis = Vis(proj_path, dataset_path, model_name, 'cpu')
    g = ViTGradients(model_name, proj_path, dataset_path)

    # Plot
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 6))
    #fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
    for i, ax in zip(img_info, axes.flat[:4]):
        b = i[3]
        h = i[4]
        grads, _ = g.compute(i[0], i[1], i[2])
        mask = - grads[b][h, 0, 1:]
        mask_vis = vis.mask(i[0], i[1], mask)
        ax.imshow(mask_vis)
        ax.set_xticks([])
        ax.set_yticks([])
        cat = stim_info[stim_info['index'] == i[2]]['cat'].unique()[0].split(',')[0].lower()
        ax.set_title(f'{cat} - bl. {b}, h. {h}', fontsize=10)
        
    for i, ax in zip(img_info, axes.flat[4:]):    
        # Compute importance by gradients
        grads, _ = g.compute(i[0], i[1], i[2])
        importance = []
        for b in range(12):
            importance.append(torch.sum(grads[b], dim=0)[0])
        mask = torch.sum(torch.stack(importance), dim=0)[1:]
        mask = - mask

        # Plot heatmap over image
        mask_vis = vis.mask(i[0], i[1], mask)
        ax.imshow(mask_vis)
        ax.set_xticks([])
        ax.set_yticks([])
        if i[2] == 582:
            cat = 'grocery store'
        else:
            cat = stim_info[stim_info['index'] == i[2]]['cat'].unique()[0].split(',')[0].lower()
        ax.set_title(cat, fontsize=10)

    fig.text(x=0.1, y=0.92, s='(a) Block- and head-specific visualization', weight='bold', fontsize=12)
    fig.text(x=0.1, y=0.5, s='(b) Sum of gradients visualization', weight='bold', fontsize=12)
    fig.text(x=0.11, y=0.86, s='(1)', weight='bold', fontsize=11)
    fig.text(x=0.51, y=0.86, s='(2)', weight='bold', fontsize=11)
    fig.text(x=0.11, y=0.44, s='(1)', weight='bold', fontsize=11)
    fig.text(x=0.51, y=0.44, s='(2)', weight='bold', fontsize=11)

    f = proj_path / 'results/figures' / f'explainability_{model_name}_v2.png'
    plt.savefig(f, dpi=200)
    plt.show()
    return


def plot_perturbation(model_name, perturb_type, res_path, random=False):
    """ 
    Plot perturbation experiment results.
    """
    labels = torch.arange(48) / 49 * 100
    
    fig, ax = plt.subplots(figsize=(6,4))
    
    f = res_path / 'perturbation' / model_name / f'{perturb_type}_grads.pt'
    emb_perturb = torch.load(f, map_location='cpu')
    emb_perturb = torch.flip(emb_perturb, dims=(0,)).detach().numpy()
    sns.lineplot(x=labels, y=emb_perturb, ax=ax, label='cls-emb removal')
    print(f'AUC emb: {np.sum(emb_perturb) / (np.max(emb_perturb) * 49)}')
    
    f = res_path / 'perturbation' / model_name / f'{perturb_type}_grads_chefer.pt'
    chefer_perturb = torch.load(f, map_location='cpu')
    chefer_perturb = torch.flip(chefer_perturb, dims=(0,)).detach().numpy()
    sns.lineplot(x=labels, y=chefer_perturb, ax=ax, label='chefer removal')
    print(f'AUC chefer: {np.sum(chefer_perturb) / (np.max(chefer_perturb) * 49)}')
    
    if random == True:
        f = res_path / 'perturbation' / model_name / 'perturb_random.pt'
        rand_perturb = torch.mean(torch.load(f, map_location='cpu'), dim=0)
        rand_perturb = torch.flip(rand_perturb, dims=(0,)).detach().numpy()
        sns.lineplot(x=labels, y=rand_perturb, ax=ax, label='random removal')
        print(f'AUC random: {np.sum(rand_perturb) / (np.max(rand_perturb) * 49)}')
    
    ax.hlines(
        xmin=-0.5, xmax=100.5, y=emb_perturb[0], colors='dimgray', linestyles='--', lw=2, 
        label='baseline accuracy'
    )
    ax.set_xticks(np.arange(0, 100, 10))
    ax.set_xlim(-0.5, 95)
    ax.set_ylim(0,0.9)
    ax.set_xlabel('percentage of tokens removed')
    ax.set_ylabel('accuracy')
    ax.legend()
    
    plt.tight_layout()
    f = res_path / 'figures' / f'neg_perturb_{model_name}.png'
    plt.savefig(f, dpi=300)
    plt.show()

    return