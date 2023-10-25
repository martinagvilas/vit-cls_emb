import timm
from timm import create_model
import torch
import torchvision.transforms as T
from transformers import AutoImageProcessor

from src.models.deit_ensemble import base_patch16_224_hierarchical


def load_vit(
        model_name, device, proj_path=None, return_transform=False, 
        pretrained=True
    ):
    """_summary_

    Parameters
    ----------
    model_name : str
        Can be one of the following options: vit_b_16, vit_b_32, vit_large_16, 
        vit_miil_16, vit_cifar_16, deit_ensemble_16, vit_gap_16. 
    device : str
        'cpu' or 'cuda'
    proj_path : pathlib Path, optional
        Path to the folder containing the source code, by default None
    return_transform : bool, optional
        Return image transform, by default False

    Returns
    -------
    list
       Containing model, number of tokens, dimension of hidden states
       and image transform.
    """
    # Get model source and info
    if model_name == 'vit_b_32':
        msource = 'vit_base_patch32_224'
        psource = 'google/vit-base-patch32-224-in21k'
        n_tokens = 50
        hs_dim = 768
    elif model_name == 'vit_b_16':
        msource = 'vit_base_patch16_224'
        psource = 'google/vit-base-patch16-224-in21k'
        n_tokens = 197
        hs_dim = 768
    elif model_name == 'vit_large_16':
        msource = 'vit_large_patch16_224'
        psource = 'google/vit-large-patch16-224-in21k'
        n_tokens = 197
        hs_dim = 1024
    elif model_name == 'vit_miil_16':
        msource = 'vit_base_patch16_224_miil'
        n_tokens = 197
        hs_dim = 768
    elif model_name == 'vit_cifar_16':
        n_tokens = 197
        hs_dim = 768
    elif model_name == 'deit_ensemble_16':
        psource = 'facebook/deit-base-distilled-patch16-224'
        n_tokens = 197
        hs_dim = 768
    elif model_name == 'vit_gap_16':
        msource = 'vit_base_patch16_rpn_224'
        psource = 'google/vit-base-patch16-224-in21k'
        n_tokens = 196
        hs_dim = 768
    
    # Load model
    if model_name.startswith('deit'):
        model = base_patch16_224_hierarchical(pretrained=pretrained).to(device)
    elif 'cifar' in model_name:
        model = create_model('vit_base_patch16_224', num_classes=100).to(device)
        if pretrained == True:
            state_dict = torch.load(proj_path / 'model_cktp' / 'vit_base_patch16_224-CIFAR100.pt')
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
    else:
        model = create_model(msource, pretrained=pretrained).to(device)

    # Get image transform
    if return_transform == True:
        if 'miil' in model_name:
            data_config = timm.data.resolve_data_config({}, model=model)
            img_transform = timm.data.create_transform(**data_config, is_training=False)
        elif 'cifar' in model_name:
            img_transform = T.Compose([
                T.Resize((224, 224),),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
        else:
            img_transform = AutoImageProcessor.from_pretrained(psource)
    else:
        img_transform = None

    return model, n_tokens, hs_dim, img_transform