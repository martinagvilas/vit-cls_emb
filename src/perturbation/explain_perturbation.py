import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTImageProcessor

from src.datasets.imagenet import ImagenetDatasetS
from src.gradients import CheferGradients, ViTGradients
from src.models.vit_edited import vit_base_patch32_224, vit_base_patch16_224


class ViTPerturb():
    def __init__(self, model_name, project_path, imgs_path, device='cpu'):
        self.model_name = model_name
        self.device = device

        self.project_path = project_path
        self.imgs_path = imgs_path
        self.perturb_path = project_path / 'results' / 'perturbation' / model
        self.perturb_path.mkdir(parents=True, exist_ok=True)
        (self.perturb_path / 'masks').mkdir(parents=True, exist_ok=True)

        self._load_model()

    def _load_model(self):
        # Get model
        if self.model_name == 'vit_b_32':
            self.model = vit_base_patch32_224(
                pretrained=True, pretrained_cfg=True
            ).to(self.device)
            source = 'google/vit-base-patch32-224-in21k'
            self.n_tokens  = 49
        elif self.model_name == 'vit_b_16':
            self.model = vit_base_patch16_224(
                pretrained=True, pretrained_cfg=True
            ).to(self.device)
            source = 'google/vit-base-patch16-224-in21k'
            self.n_tokens = 196
        self.model.eval()
        
        # Get image transform
        self.img_transform = ViTImageProcessor.from_pretrained(source)
        return

    def _get_dataset(self):
        self.dataset = ImagenetDatasetS(self.imgs_path)
        self.dataloader = DataLoader(self.dataset, batch_size=1, collate_fn=lambda x: x[0])
        return

    def compute(self, perturb_type='negative', source_type='grads'):
        self.perturb_type = perturb_type
        self.source_type = source_type

        # Get mask
        self._get_dataset()
        if self.source_type == 'grads':
            tokens_mask = self.get_accumulated_grads()
        elif self.source_type == 'grads_chefer':
            tokens_mask = self.get_grads_chefer()
        elif self.source_type == 'linear_probe':
            tokens_mask = self.get_accumulated_linear_probe()

        # Compute accuracy
        accs = []
        for n in range(1, self.n_tokens):
            accs.append(self._compute(n, tokens_mask))
        accs = torch.stack(accs).to(self.device)
        
        # Save accuracy
        f = self.perturb_path / f'{self.perturb_type}_{self.source_type}.pt'
        torch.save(accs, f)
        
        return

    def _compute(self, n_tokens, mask):
        self._get_dataset()
        acc = []
        for id, data in tqdm(
            enumerate(self.dataloader), total=len(self.dataloader)
        ):
            # Get image features
            img_ft = self.img_transform(data['img'], return_tensors="pt")
            img_ft = img_ft['pixel_values'].to(self.device)

            # Get decoded tokens and add CLS
            tokens_mask = mask[id][:n_tokens].to(self.device)
            tokens_mask = torch.cat((torch.tensor([0,]).to(self.device), tokens_mask))

            # Compute hidden states
            with torch.no_grad():
                out = self.model(img_ft, tokens_mask)
            
            # Compute accuracy
            pred = out.topk(1)[1]
            cat_acc = torch.squeeze((pred == data['index']).long())
            acc.append(cat_acc)

        # Save accuracy
        acc = torch.hstack(acc).to(self.device)
        acc = torch.sum(acc) / 4550
        print(f'acc {n_tokens}: {acc}', flush=True)

        return acc

    def get_accumulated_grads(self):
        f = self.perturb_path / 'masks' / f'{self.source_type}.pt'
        if f.is_file():
            tokens_mask = torch.load(f).to(self.device)
        else:
            tokens_mask = []
            for data in tqdm(self.dataloader, total=len(self.dataloader)):
                concept=data['imagenet_id']
                img_idx=data['img_index']
                cat_idx=data['index']
                
                # Compute grads
                g = ViTGradients(self.model_name, self.project_path, self.imgs_path)
                grads, _ = g.compute(concept, img_idx, cat_idx)

                # Accumulate over blocks
                importance = []
                for b in range(12):
                    importance.append(torch.sum(grads[b], dim=0)[0])
                mask = torch.sum(torch.stack(importance), dim=0)[1:]
                mask = - mask
                
                # Orden in terms of importance
                mask = mask.topk(self.n_tokens, dim=0)[1] + 1
                tokens_mask.append(mask)
                
            tokens_mask = torch.stack(tokens_mask).to(self.device)
            torch.save(tokens_mask, f)
        
        # Flip order if positive perturbation
        if self.perturb_type == 'positive':
            tokens_mask = torch.flip(tokens_mask, dims=[1])
                
        return tokens_mask
    
    def get_grads_chefer(self):
        f = self.perturb_path / 'masks' / f'{self.source_type}.pt'
        if f.is_file():
            tokens_mask = torch.load(f).to(self.device)
        else:
            tokens_mask = []
            for data in tqdm(self.dataloader, total=len(self.dataloader)):
                concept=data['imagenet_id']
                img_idx=data['img_index']
                cat_idx=data['index']
        
                # Compute grads
                g = CheferGradients(self.model_name, self.project_path, self.imgs_path)
                R = g.compute(concept, img_idx, cat_idx)
                mask = R.topk(self.n_tokens, dim=0)[1] + 1
                tokens_mask.append(mask)
            
            tokens_mask = torch.stack(tokens_mask).to(self.device)
            torch.save(tokens_mask, f)
        
        # Flip order if positive perturbation
        if self.perturb_type == 'positive':
            tokens_mask = torch.flip(tokens_mask, dims=[1])

        return tokens_mask

    def get_accumulated_linear_probe(self):
        # Accumulate linear probing results over blocks
        decs = []
        for b in range(12):
            path = project_path / 'results/linear_probing' / f'hs-{b}'
            b_decs = []
            for t in range(1, 50):
                f = path / f'pos_t{t}.npy'
                dec = np.load(f)
                b_decs.append(dec)
            decs.append(torch.Tensor(np.vstack(b_decs)).to(self.device).T)
        decs = torch.sum(torch.stack(decs), dim=0)

        tokens_mask = decs.topk(k=decs.shape[1], dim=1, largest=False)[1]
        tokens_mask = tokens_mask + 1
        
        # Flip order if positive perturbation
        if self.perturb_type == 'positive':
            tokens_mask = torch.flip(tokens_mask, dims=[1])
        
        return tokens_mask

    def compute_random(self, perms=10):
        random_accs = []
        for p in range(perms):
            # Get random mask
            random_mask = torch.randperm(49, generator=torch.manual_seed(p)) + 1

            # Compute accuracy
            p_accs = []
            for n in range(1, self.n_tokens):
                mask = random_mask[:n]
                mask = torch.cat((torch.tensor([0,]).to(device), mask.to(device)))
                p_accs.append(self._compute_random(mask))
            random_accs.append(torch.stack(p_accs).to(self.device))
        
        random_accs = torch.stack(random_accs).to(self.device)
        f = self.perturb_path / f'perturb_random.pt'
        torch.save(random_accs, f)
        
        return
    
    def _compute_random(self, tokens_mask):
        acc = []
        for _, data in tqdm(
            enumerate(self.dataloader), total=len(self.dataloader)
        ):
            # Get image features
            img_ft = self.img_transform(data['img'], return_tensors="pt")
            img_ft = img_ft['pixel_values'].to(self.device)

            # Compute hidden states
            with torch.no_grad():
                out = self.model(img_ft, tokens_mask)
            
            # Get sample accuracy
            pred = out.topk(1)[1]
            cat_acc = torch.squeeze((pred == data['index']).long())
            acc.append(cat_acc)

        # Compute accuracy
        acc = torch.hstack(acc).to(self.device)
        acc = torch.sum(acc) / 4550
        print(f'acc {len(tokens_mask)}: {acc}', flush=True)

        return acc


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
        vit_16, vit_32.'
    )
    parser.add_argument(
        '-random', action='store_true',help='Compute random mask'
    )
    parser.add_argument(
        '-pt', action='store', required=True, 
        help='Perturbation type: positive or negative'
    )
    parser.add_argument(
        '-st', action='store', required=True, 
        help='Source for the importance weights: grads, grads_chefer or linear_probe'
    )
    
    args = parser.parse_args()
    model = args.m
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random = args.random
    perturb_type = args.pt
    source_type = args.st
    
    project_path = Path(args.pp)
    data_path = Path(args.dp)

    if random == True:
        p = ViTPerturb(model, project_path, data_path, device)
        p.compute_random(perturb_type=perturb_type, source_type=source_type)
    else:
        p = ViTPerturb(model, project_path, data_path, device)
        p.compute(perturb_type=perturb_type, source_type=source_type)