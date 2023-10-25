import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch

from src.datasets.cifar import MyCIFAR100
from src.datasets.imagenet import ImagenetDatasetS
from src.models.load import load_vit


class Memory:
    def __init__(self, proj_path, dataset_path, model, device='cpu'):
        self.model_name = model
        self.device = device

        self.proj_path = Path(proj_path)
        self.dataset_path = Path(dataset_path)
        self.net_ft_path = proj_path / 'results' / 'net_ft' / model
        self.dec_path = proj_path / 'results' / 'class_embed' / model

        if 'cifar' in model:
            self.stim_info = MyCIFAR100(self.dataset_path).stim_info
        else:
            self.stim_info = ImagenetDatasetS(self.dataset_path).stim_info

        self._load_model()

    def _load_model(self):
        # Load model
        self.model, self.n_tokens, self.hs_dim, _ = load_vit(
            self.model_name, self.device, self.proj_path
        )
        self.model.eval()
        
        # Load random model
        self.random_model, _, _, _ = load_vit(
            self.model_name, self.device, self.proj_path, pretrained=False
        )
        self.random_model.eval()

        # Define number of layers
        if 'large' in self.model_name:
            self.n_layers = 24
        else:
            self.n_layers = 12

        # Define number of classes
        if 'cifar' in self.model_name:
            self.n_classes = 100
        else:
            self.n_classes = 1000
        
        return

    def compute_class_value_agr(self, layer_type):
        """
        Compute class-value agreement scores.
        """
        # Normalize class embedding        
        with torch.no_grad():
            self.model.head.weight /= self.model.head.weight.norm(dim=-1, keepdim=True)
            self.random_model.head.weight /= self.random_model.head.weight.norm(dim=-1, keepdim=True)

        # Get top-1 logit per class
        max_logits = []
        pvals = []
        random_k = []
        for b in range(self.n_layers):
            if layer_type == 'mlp':
                with torch.no_grad():
                    self.model.blocks[b].mlp.fc2.weight /= self.model.blocks[b].mlp.fc2.weight.norm(dim=0, keepdim=True)
                    val_proj = self.model.head.weight @ self.model.blocks[b].mlp.fc2.weight
            else:
                with torch.no_grad():
                    self.model.blocks[b].attn.proj.weight /= self.model.blocks[b].attn.proj.weight.norm(dim=0, keepdim=True)
                    val_proj = self.model.head.weight @ self.model.blocks[b].attn.proj.weight
            
            logits_top_k = torch.squeeze(val_proj.topk(1, dim=1)[0])
            max_logits.append(logits_top_k)
            
            # Get top-1 logit per class in random model
            if layer_type == 'mlp':
                with torch.no_grad():
                    self.random_model.blocks[b].mlp.fc2.weight /= self.random_model.blocks[b].mlp.fc2.weight.norm(dim=0, keepdim=True)
                    val_proj = self.random_model.head.weight @ self.random_model.blocks[b].mlp.fc2.weight
            else:
                with torch.no_grad():
                    self.random_model.blocks[b].attn.proj.weight /= self.random_model.blocks[b].attn.proj.weight.norm(dim=0, keepdim=True)
                    val_proj = self.random_model.head.weight @ self.random_model.blocks[b].attn.proj.weight
                    
            random_top_k = torch.squeeze(val_proj.topk(1, dim=1)[0]).detach().numpy()
            random_k.append(np.mean(random_top_k))
            
            # Compute statistical difference
            wx = wilcoxon(logits_top_k.detach().numpy(), random_top_k, alternative='greater')
            pvals.append(wx.pvalue)

        # Save results in dataframe
        max_logits = torch.stack(max_logits).flatten().detach().numpy()
        blocks = torch.arange(1, self.n_layers+1).repeat_interleave(self.n_classes)
        df = pd.DataFrame({'top-1 logits': max_logits, 'block': blocks})
        
        return df, pvals, random_k

    def compute_key_value_agreement(self, block, layer_type):
        """Compute key value agreement rates.
        """
        res_path = self.proj_path / 'results' / 'memories' / self.model_name
        res_path.mkdir(parents=True, exist_ok=True) 

        if layer_type == 'mlp':
            with torch.no_grad():
                val_proj = self.model.head.weight @ self.model.blocks[block].mlp.fc2.weight
        elif layer_type == 'attn':
            with torch.no_grad():
                val_proj = self.model.head.weight @ self.model.blocks[block].attn.proj.weight
        key_topk = val_proj.topk(5, dim=1)[1]

        if 'cifar' in self.model_name:
            concepts = list(self.stim_info.keys())
            indexes = concepts
            n_classes = 100
        else:
            concepts = self.stim_info['imagenet_id'].unique()
            indexes = self.stim_info['index'].unique()
            n_classes = 1000
        
        agreement = []
        agreement_random = []
        logits = []
        logits_random = []
        for c, c_idx in zip(concepts, indexes):
            c_idx = int(c_idx)

            # Get top-5 keys of hidden state
            if layer_type == 'mlp':
                file = self.net_ft_path / c / f'key-{layer_type}_{block}.pt'
                key_val = torch.load(file, map_location=self.device)
                
            elif layer_type == 'attn':
                attn_file = self.net_ft_path / c / f'attn-w_{block}.pt'
                attn_data = torch.load(attn_file, map_location=self.device)
                val_file = self.net_ft_path / c / f'value_{block}.pt'
                val_data = torch.load(val_file, map_location=self.device)
                key_val = attn_data @ val_data
                key_val = key_val.transpose(1,2).reshape(5, self.n_tokens, self.hs_dim)
            
            hs_key_topk = key_val.topk(5, dim=-1)[1]
            
            # Compute top-k value agreement
            agreement.append(self._compute_agreement(hs_key_topk, key_topk[c_idx]))
            agreement_random.append(self._compute_agreement(
                hs_key_topk, key_topk[torch.randint(0, n_classes, (1,))])
            )

            # Compute average logits
            logits.append(self._get_logits(hs_key_topk, val_proj, c_idx))
            logits_random.append(
                self._get_logits(hs_key_topk, val_proj, torch.randint(0, n_classes, (1,)))
            )

        f_agreement = res_path / f'{layer_type}_{block}_value-class_agreement.pt'
        torch.save(torch.stack(agreement), f_agreement)
        f_agreement_random = res_path / f'{layer_type}_{block}_value-class_agreement-random.pt'
        torch.save(torch.stack(agreement_random), f_agreement_random)

        f_logits = res_path / f'{layer_type}_{block}_value-class_logits.pt'
        torch.save(torch.stack(logits), f_logits)
        f_logits_random = res_path / f'{layer_type}_{block}_value-class_logits-random.pt'
        torch.save(torch.stack(logits_random), f_logits_random)
    
        return
    
    def _compute_agreement(self, hs_key_topk, c_key_topk):
        agr = torch.isin(hs_key_topk, c_key_topk)
        agr = torch.any(agr, dim=-1)
        return agr

    def _get_logits(self, hs_key_topk, val_proj, c_idx):
        c_logits = []
        for img in range(hs_key_topk.shape[0]):
            for t in range(hs_key_topk.shape[1]):
                ks = hs_key_topk[img, t]
                c_logits.append(val_proj[c_idx, ks])
        c_logits = torch.stack(c_logits).view(hs_key_topk.shape[0], hs_key_topk.shape[1], -1)
        return c_logits

    def compute_composition(self, block, layer_type):
        # Create paths
        res_path = self.proj_path / 'results' / 'memories' / self.model_name
        res_path.mkdir(parents=True, exist_ok=True) 

        # Get concepts
        if 'cifar' in self.model_name:
            concepts = list(self.stim_info.keys())
        else:
            concepts = self.stim_info['imagenet_id'].unique()

        # Get most activating class per memory value
        if layer_type == 'mlp':
            with torch.no_grad():
                val_proj = self.model.head.weight @ self.model.blocks[block].mlp.fc2.weight
        elif layer_type == 'attn':
            with torch.no_grad():
                val_proj = self.model.head.weight @ self.model.blocks[block].attn.proj.weight
        key_topk = torch.squeeze(val_proj.topk(1, dim=0)[1])

        # Get top-5 most activated memories per image and token
        match = []
        for c in concepts:
            if layer_type == 'mlp':
                file = self.net_ft_path / c / f'key-{layer_type}_{block}.pt'
                key_val = torch.load(file, map_location=self.device)
                
            elif layer_type == 'attn':
                attn_file = self.net_ft_path / c / f'attn-w_{block}.pt'
                attn_data = torch.load(attn_file, map_location=self.device)
                val_file = self.net_ft_path / c / f'value_{block}.pt'
                val_data = torch.load(val_file, map_location=self.device)
                key_val = attn_data @ val_data
                key_val = key_val.transpose(1,2).reshape(5, self.n_tokens, self.hs_dim)
            
            hs_key_topk = key_val.topk(5, dim=-1)[1]

            # Get most activating classes for the top-5 activated memories
            for img in range(key_val.shape[0]):
                for t in range(key_val.shape[1]):
                    for k in range(5):
                        k_top = key_topk[hs_key_topk[img, t, k]]
                        hs_key_topk[img, t, k] = k_top

            # Compute agreement with predictions at output
            file = self.dec_path / c / f'topk_hs-{layer_type}_{block}.pt'
            preds = torch.load(file, map_location=self.device)[:, :, 0]
            img_match = []
            for img in range(key_val.shape[0]):
                for t in range(key_val.shape[1]):
                    layer_pred = preds[img, t]
                    img_match.append(torch.isin(hs_key_topk[img, t], layer_pred))
            img_match = torch.stack(img_match)
            img_match = img_match.reshape(key_val.shape[0], key_val.shape[1], -1)
            match.append(img_match)

        f = res_path / f'{layer_type}_{block}_top5_pred-match.pt'
        torch.save(torch.stack(match), f)

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
        vit_16, vit_32'
    )
    parser.add_argument(
        '-lt', action='store', required=True, 
        help='Hidden state type'
    )

    args = parser.parse_args()
    
    project_path = Path(args.pp)
    dataset_path = Path(args.dp)
    
    model = args.m
    layer_type = args.lt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if 'large' in model:
        n_layers = 24
    else:
        n_layers = 12

    for b in range(n_layers):
        memory = Memory(project_path, dataset_path, model, device=device)
        memory.compute_key_value_agreement(b, layer_type)
        memory.compute_composition(b, layer_type)