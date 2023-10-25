import argparse
from itertools import product
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from src.datasets.imagenet import ImagenetDatasetS
from src.datasets.cifar import MyCIFAR100
from src.models.load import load_vit


class ViTExtractor():
    """Project hidden states to class embedding space and save key coefficients.
    """
    def __init__(
            self, model_name, project_path, imgs_path, device='cpu', 
            pretrained=True
        ):
        self.pretrained = pretrained        
        if self.pretrained == True:
            self.model_name = model_name
        elif self.pretrained == False:
            self.model_name = f'{model_name}_random'
        self.device = device

        self.project_path = project_path
        self.imgs_path = imgs_path
        self.hs_path = project_path / 'results' / 'class_embed' / model_name
        self.hs_path.mkdir(parents=True, exist_ok=True)
        
        # super().__init__(self.model_name, project_path, imgs_path, device)

        self._get_model_layers()
        self._load_model()

    def _get_model_layers(self):
        prefix = 'blocks.'
        join = '.'
        blocks_layer_types = [
            'add_1', 'attn.getitem_4', 'attn.getitem_5', 'attn.softmax', 
            'attn.proj', 'mlp.fc1', 'mlp.fc2'
        ]

        if '_large_' in self.model_name:
            n_blocks = np.arange(24)
        else:
            n_blocks = np.arange(12)

        layers = ['head']
        for b, l in product(n_blocks, blocks_layer_types):
            if l == None:
                layers.append(f'{prefix}{b}')
            else:
                layers.append(f'{prefix}{b}{join}{l}')

        self.layers = layers
        return
    
    def _get_layer_name(self, layer_model):
        """Create layer name from layer id."""
        if layer_model == 'head':
            layer_name = 'cls-out'
        else:
            block = re.findall(r'\d+', layer_model)[0]
            if any(w in layer_model for w in ['.k_', 'getitem_4']):
                suffix = f'key'
            elif any(w in layer_model for w in ['.q_', 'getitem_3']):
                suffix = f'query'
            elif any(w in layer_model for w in ['.v_', 'getitem_5']):
                suffix = f'value'
            elif any(w in layer_model for w in ['attn.softmax']):
                suffix = f'attn-w'
            elif any(w in layer_model for w in ['attn.proj']):
                suffix = f'hs-attn'
            elif any(w in layer_model for w in ['mlp.fc1']):
                suffix = 'key-mlp'
            elif any(w in layer_model for w in ['.mlp', 'mlp.fc2']):
                suffix = f'hs-mlp'
            else:
                suffix = 'hs'
            layer_name = f'{suffix}_{block}'
        return layer_name

    def _load_model(self):
        # Get model
        self.model, self.n_tokens, _, self.img_transform = load_vit(
            self.model_name, self.device, self.project_path, return_transform=True
        )
        self.model.eval()

        # Add feature extractor
        layers_map = {l: self._get_layer_name(l) for l in self.layers}
        self.extractor = create_feature_extractor(self.model, layers_map).to(self.device)

        # Get normalization    
        if self.model_name == 'vit_gap_16':
            self.ln = self.model.fc_norm
        else:
            self.ln = self.model.norm
        
        # Get projection matrix
        self.cls_proj = self.model.head
    
        return

    def _get_dataset(self):
        self.dataset = ImagenetDatasetS(self.imgs_path)
        self.dataloader = DataLoader(
            self.dataset, batch_size=5, collate_fn=self._imagenet_collate_batch
        )
        return

    def _imagenet_collate_batch(self, batch):
        assert all(i['imagenet_id']==batch[0]['imagenet_id'] for i in batch)
        data = {}
        data['imagenet_id'] = batch[0]['imagenet_id']
        data['index'] = batch[0]['index']
        data['cat'] = batch[0]['cat']
        data['imgs'] = [i['img'] for i in batch]
        return data

    def extract_hidden_states(self):
        self._get_dataset()
        acc = []
        for _, data in tqdm(
            enumerate(self.dataloader), total=len(self.dataloader)
        ):
            # Prepare concept path
            cls_emb_path = self.hs_path / data['imagenet_id']
            cls_emb_path.mkdir(parents=True, exist_ok=True)
            if self.pretrained == True:
                net_ft_path = self.project_path / 'results' / 'net_ft' / self.model_name / data['imagenet_id']
                net_ft_path.mkdir(parents=True, exist_ok=True)

            # Get image features
            try:
                img_ft = self.img_transform(data['imgs'], return_tensors="pt")
                img_ft = img_ft['pixel_values'].to(self.device)
            except:
                img_ft = [self.img_transform(i) for i in data['imgs']]
                img_ft = torch.stack(img_ft).to(self.device)

            # Compute hidden states
            with torch.no_grad():
                out = self.extractor(img_ft)
            
            # Compute and save projections
            for l_name, l_repr in out.items():
                if l_name == 'cls-out':
                    pred = l_repr.topk(1)[1]
                    cat_acc = torch.squeeze((pred == data['index']).long())
                    acc.append(cat_acc)
                
                elif 'hs' in l_name:
                    # Project to class embedding space
                    if 'gap' in self.model_name:  # add normalization
                        block = int(l_name.split('_')[-1])
                        with torch.no_grad():
                            if 'attn' in l_name:
                                l_repr = self.model.blocks[block].norm1(l_repr)
                            elif 'mlp' in l_name:
                                l_repr = self.model.blocks[block].norm2(l_repr)
                            preds = self.cls_proj(self.ln(l_repr))
                    else:
                        with torch.no_grad():
                            preds = self.cls_proj(self.ln(l_repr))
                    
                    # Get top-5 predictions
                    top_k = preds.topk(5, dim=-1)[1]
                    torch.save(top_k, (cls_emb_path / f'topk_{l_name}.pt'))

                    # Get correct label position
                    ordered_idx = torch.argsort(preds, dim=2, descending=True)
                    label_idx = (ordered_idx == data['index']).nonzero()
                    pos = label_idx[:, 2].reshape(self.dataset.n_imgs, (self.n_tokens))
                    torch.save(pos, (cls_emb_path / f'pos_{l_name}.pt'))

                    # Get correct label probability
                    probs = preds[:, :, data['index']].clone()
                    torch.save(probs, (cls_emb_path / f'probs_{l_name}.pt'))
                
                # Save key coefficients
                elif self.pretrained == True:
                    l_repr = l_repr.clone()
                    torch.save(l_repr, (net_ft_path / f'{l_name}.pt'))

                else:
                    continue

        # Save accuracy
        acc = torch.hstack(acc)
        file = self.hs_path / 'acc.pt'
        torch.save(acc, file)
    
        return


class ExtractorCIFAR100(ViTExtractor):
    """Project hidden states to class embedding space and save key coefficients
    of the CIFAR model.
    """
    def __init__(self, model_name, project_path, imgs_path, device, pretrained=True):
        self.model_name = model_name
        self.pretrained = pretrained
        super().__init__(self.model_name, project_path, imgs_path, device, pretrained=pretrained)
        self._load_model()
        return
    
    def _get_dataset(self):
        self.dataset = CIFAR100(
            self.imgs_path, train=False, download=True, transform=self.img_transform
        )
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        return
    
    def extract_hidden_states(self):
        dataset = MyCIFAR100(self.imgs_path)

        acc = []
        for label, data in tqdm(dataset.stim_info.items(), total=len(dataset)):
            # Prepare concept path
            cls_emb_path = self.hs_path / label
            cls_emb_path.mkdir(parents=True, exist_ok=True)
            
            if self.pretrained == True:
                net_ft_path = self.project_path / 'results' / 'net_ft' / self.model_name / label
                net_ft_path.mkdir(parents=True, exist_ok=True)

            # Get image features
            img_ft = torch.stack([self.img_transform(img) for img in data]).to(self.device)

            # Compute hidden states
            with torch.no_grad():
                out = self.extractor(img_ft)
            
            # Save hidden states
            for l_name, l_repr in out.items():
                if l_name == 'cls-out':
                    pred = l_repr.topk(1)[1]
                    cat_acc = torch.squeeze((pred == int(label)).long())
                    acc.append(cat_acc)
                
                elif 'hs' in l_name:
                    with torch.no_grad():
                        preds = self.cls_proj(self.ln(l_repr))
                        
                    top_k = preds.topk(5, dim=-1)[1]
                    torch.save(top_k, (cls_emb_path / f'topk_{l_name}.pt'))

                    ordered_idx = torch.argsort(preds, dim=2, descending=True)
                    label_idx = (ordered_idx == int(label)).nonzero()
                    pos = label_idx[:, 2].reshape(dataset.n_imgs, (self.n_tokens))
                    torch.save(pos, (cls_emb_path / f'pos_{l_name}.pt'))

                    probs = preds[:, :, int(label)].clone()
                    torch.save(probs, (cls_emb_path / f'probs_{l_name}.pt'))
                
                elif self.pretrained == True:
                    l_repr = l_repr.clone()
                    torch.save(l_repr, (net_ft_path / f'{l_name}.pt'))

                else:
                    continue

        # Save accuracy
        acc = torch.hstack(acc)
        file = self.hs_path / 'acc.pt'
        torch.save(acc, file)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-pp', action='store', required=True, 
        help='Path to the folder containing the source code.'
    )
    parser.add_argument(
        '-dp', action='store', required=True, 
        help='Path to the folder containing the dataset.'
    )
    parser.add_argument(
        '-m', action='store', required=True, 
        help='Select which model to run. Can be one of the following options: \
        vit_b_16, vit_b_32, vit_large_16, vit_miil_16,  vit_cifar_16, \
        deit_ensemble_16, vit_gap_16.'
    )
    parser.add_argument(
        '-pretrained', action='store_true', help='Use pretrained model.'
    )

    args = parser.parse_args()
    project_path = Path(args.pp)
    data_path = Path(args.dp)
    
    model = args.m
    pretrained = args.pretrained

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model == 'vit_cifar_16':
        extractor = ExtractorCIFAR100(
            model, project_path, data_path, device=device, pretrained=pretrained
        )
        extractor.extract_hidden_states()
    else:
        extractor = ViTExtractor(
            model, project_path, data_path, device=device, pretrained=pretrained
        )
        extractor.extract_hidden_states()