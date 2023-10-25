from pathlib import Path

import torch

from src.datasets.imagenet import ImagenetDatasetS


def compute_accuracy(
    proj_path, dataset_path, model, device='cpu', by_concept=False,
    concept=None, percentage=True
):
    # Load accuracy
    acc_file = Path(proj_path) / 'results/class_embed' / model / 'acc.pt'
    acc = torch.load(acc_file, map_location=device)
    
    # Compute accuracy by concept
    if concept:
        stim_info = ImagenetDatasetS(Path(dataset_path)).stim_info
        c_idx = torch.Tensor(
            stim_info.loc[stim_info['imagenet_id'] == concept].index
        ).long()
        acc = acc[c_idx]
        if percentage:
            return torch.sum(acc) / len(acc) * 100
        else:
            return acc
    elif by_concept == True:
        return acc
    else:
        return torch.sum(acc) / len(acc) * 100