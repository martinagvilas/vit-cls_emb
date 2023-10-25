import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTImageProcessor

from src.datasets.imagenet import ImagenetDatasetS
from src.models.vit_edited import vit_base_patch32_224, vit_base_patch16_224


class ViTPerturbAttn():
    def __init__(
        self, version, project_path, imgs_path, device='cpu', 
        perturb_type='no_cls'
    ):
        self.version = version
        self.model_name = f'vit_b_{version}'
        self.device = device
        self.perturb_type = perturb_type

        self.project_path = project_path
        self.imgs_path = imgs_path
        self.perturb_path = project_path / 'results' / 'perturbation' / model
        self.perturb_path.mkdir(parents=True, exist_ok=True)

        self._get_model_layers()
        self._load_model()

    def _get_model_layers(self):
        self.layers = ['blocks.11']
        return
    
    def _load_model(self):
        if self.version == '32':
            self.model = vit_base_patch32_224(
                pretrained=True, pretrained_cfg=True, 
                perturb_type=self.perturb_type, block_perturb='all'
            ).to(self.device)
            source = 'google/vit-base-patch32-224-in21k'
            self.n_tokens  = 49
        elif self.version == '16':
            self.model = vit_base_patch16_224(
                pretrained=True, pretrained_cfg=True, 
                perturb_type=self.perturb_type, block_perturb='all'
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
        dataset = ImagenetDatasetS(self.imgs_path)
        self.dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
        return

    def compute(self):
        self._get_dataset()
        acc = []
        dec = defaultdict(list)
        for id, data in tqdm(
            enumerate(self.dataloader), total=len(self.dataloader)
        ):
            # Get image features
            img_ft = self.img_transform(data['img'], return_tensors="pt")
            img_ft = img_ft['pixel_values'].to(self.device)

            # Compute hidden states
            self.model._features = {}
            with torch.no_grad():
                out = self.model(img_ft)
            
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
        torch.save(acc, self.perturb_path / f'attn-{self.perturb_type}_acc.pt')

        dec = torch.stack(dec['hs_11']).to(self.device)
        torch.save(dec, self.perturb_path / f'attn-{self.perturb_type}_dec.pt')

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
        vit_16, vit_32.'
    )
    parser.add_argument(
        '-pt', action='store', required=True, 
        help='Perturbation type. Can be one of [no_cls], [self_only].'
    )
    args = parser.parse_args()
    
    project_path = Path(args.pp)
    data_path = Path(args.dp)

    model = args.m
    version = model.split('_')[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perturb_type = args.pt
    
    perturb = ViTPerturbAttn(
        version, project_path, data_path, device, perturb_type=perturb_type
    )
    perturb.compute()