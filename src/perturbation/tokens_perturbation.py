import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTImageProcessor

from src.datasets.imagenet import ImagenetDatasetS
from src.models.vit_edited import vit_base_patch32_224, vit_base_patch16_224
from src.vis import Vis


class ViTPerturbTokens():
    def __init__(self, version, project_path, imgs_path, device='cpu'):
        self.version = version
        self.model_name = f'vit_b_{version}'
        self.device = device

        self.project_path = Path(project_path)
        self.imgs_path = Path(imgs_path)
        self.perturb_path = project_path / 'results' / 'perturbation' / model
        self.perturb_path.mkdir(parents=True, exist_ok=True)

        self._get_model_layers()
        self._load_model()

    def _get_model_layers(self):
        # Compute identifiability of tokens in last block
        self.layers = ['blocks.11']
        return

    def _load_model(self):
        # Load model
        if self.version == '32':
            self.model = vit_base_patch32_224(
                pretrained=True, pretrained_cfg=True
            ).to(self.device)
            self.n_tokens  = 49
            source = 'google/vit-base-patch32-224-in21k'
        elif self.version == '16':
            self.model = vit_base_patch16_224(
                pretrained=True, pretrained_cfg=True
            ).to(self.device)
            source = 'google/vit-base-patch16-224-in21k'
            self.n_tokens = 196
        self.model.eval()

        # Add feature extractor
        self._add_feature_extractor()

        # Get image transform
        self.img_transform = ViTImageProcessor.from_pretrained(source)
        return
    
    def _add_feature_extractor(self):
        def get_activation(layer_name):
            def hook(_, input, output):
                try:
                    self.model._features[layer_name] = output.detach()
                except AttributeError:
                    # attention layer can output a tuple
                    self.model._features[layer_name] = output[0].detach()
            return hook
        
        for layer_name, layer in self.model.named_modules():
            if layer_name in self.layers:
                layer.register_forward_hook(get_activation(layer_name))
        return


    def _get_dataset(self):
        self.dataset = ImagenetDatasetS(self.imgs_path)
        self.dataloader = DataLoader(
            self.dataset, batch_size=1, collate_fn=lambda x: x[0]
        )
        return

    def compute(self, mask_type='context'):
        self._get_dataset()
        
        vis = Vis(self.project_path, self.imgs_path, self.model_name, self.device)

        acc = []
        dec = defaultdict(list)
        for id, data in tqdm(
            enumerate(self.dataloader), total=len(self.dataloader)
        ):
            # Get image features
            img_ft = self.img_transform(data['img'], return_tensors="pt")
            img_ft = img_ft['pixel_values'].to(self.device)

            # Get segmentations of discarded tokens
            mask = vis.get_segmentation(data['imagenet_id'], idx=data['img_index'])
            mask = mask.flatten()
            if mask_type == 'context':
                mask = torch.tensor((mask == 1).nonzero()[0] + 1).to(self.device)
            elif mask_type == 'class_label':
                mask = torch.tensor((mask == 0).nonzero()[0] + 1).to(self.device)
            tokens = torch.hstack((torch.tensor([0]), mask))

            # Compute hidden states
            self.model._features = {}
            with torch.no_grad():
                out = self.model(img_ft, tokens)
            
            # Turn hidden states into decodability scores
            for l_name, l_repr in self.model._features.items():
                l_split = l_name.split('.')
                try:
                    l_name = f'hs-{l_split[2]}_{l_split[1]}'
                except:
                    l_name = f'hs_{l_split[1]}'
                with torch.no_grad():
                    preds = self.model.head(self.model.norm(l_repr))
                ordered_idx = torch.argsort(preds, dim=2, descending=True)
                label_idx = (ordered_idx == data['index']).nonzero()
                dec[l_name].append(label_idx[:, 2])
                
            # Compute accuracy
            pred = out.topk(1)[1]
            cat_acc = torch.squeeze((pred == data['index']).long())
            acc.append(cat_acc)

        # Save accuracy and hidden states
        acc = torch.hstack(acc).to(self.device)
        torch.save(acc, self.perturb_path / f'no_{mask_type}_tokens_acc.pt')
        torch.save(dec, self.perturb_path / f'no_{mask_type}_tokens_dec.pt')

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
        '-mt', action='store', required=True, 
        help='Mask type. Can be one of [context], [class_label].'
    )
    args = parser.parse_args()
    
    project_path = Path(args.pp)
    data_path = Path(args.dp)

    model = args.m
    version = model.split('_')[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_type = args.mt

    perturb = ViTPerturbTokens(version, project_path, data_path, device)
    if mask_type:
        perturb.compute(mask_type)
    else:
        perturb.compute()