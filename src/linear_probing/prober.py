import argparse
from pathlib import Path
from PIL import Image

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from timm import create_model
from tqdm import tqdm
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import AutoImageProcessor

from src.datasets.imagenet import ImagenetDatasetS

LAYERS = [f'{layer}-{i}' for layer in ['hs', 'attn', 'mlp'] for i in range(12)]

CATS_EXCLUDE = [
    'n04356056', 'n04355933', 'n04493381', 'n02808440', 'n03642806',
    'n03832673', 'n04008634', 'n03773504', 'n03887697', 'n15075141'
]


class LinearProber:
    def __init__(self, model_name, layer, project_path, imgs_path, device):
        self.model_name = model_name
        self.layer = layer
        self.device = device
        self._load_model()

        self.project_path = Path(project_path)
        self.imgs_path = Path(imgs_path)

        return

    def _load_model(self):
        # Load model
        self.model = create_model(
            'vit_base_patch32_224', pretrained=True
        ).to(self.device)

        # Add feature extractor
        block = self.layer.split('-')[-1]
        if 'attn' in self.layer:
            self.node = f'blocks.{block}.attn.proj'
        elif 'mlp' in self.layer:
            self.node = f'blocks.{block}.mlp.fc2'
        elif 'hs' in self.layer:
            self.node = f'blocks.{block}.add_1'
        self.extractor = create_feature_extractor(self.model, [self.node]).to(self.device)

        # Get image processor
        self.img_transform = AutoImageProcessor.from_pretrained(
            'google/vit-base-patch32-224-in21k'
        )

        return

    def _get_training_data(self, token):
        dataset = ImagenetDatasetS(self.imgs_path, partition='test', n=10)

        fts = []
        targets = []
        for _, row in tqdm(
            dataset.stim_info.iterrows(), total=len(dataset.stim_info), 
            desc=f'{self.node}/{token}'
        ):
            # Get image features
            img_path = self.imgs_path / 'ImageNetS919/test' / row['imagenet_id'] / row['img_name']
            img = Image.open(img_path).convert('RGB')

            img_ft = self.img_transform(img, return_tensors="pt")
            img_ft = img_ft['pixel_values'].to(self.device)

            # Run model
            with torch.no_grad():
                out = self.extractor(img_ft)

            # Save features
            out = out[self.node][0, token]
            fts.append(out)

            # Get one hot encoder
            target = torch.zeros(1000).to(self.device) - 1
            target[row['index']] = 1
            targets.append(target)

        fts = torch.stack(fts)
        targets = torch.stack(targets)

        return fts, targets

    def _get_testing_data(self, token):
        dataset = ImagenetDatasetS(self.imgs_path)

        fts = []
        targets = []
        for _, row in tqdm(dataset.stim_info.iterrows(), total=len(dataset.stim_info)):
            # Get image features
            img_path = self.imgs_path / 'ImageNetS919/validation' / row['imagenet_id'] / row['img_name']
            img = Image.open(img_path).convert('RGB')

            img_ft = self.img_transform(img, return_tensors="pt")
            img_ft = img_ft['pixel_values'].to(self.device)

            # Run model
            with torch.no_grad():
                out = self.extractor(img_ft)

            # Save features
            out = out[self.node][0, token]
            fts.append(out)

            # Get index
            targets.append(row['index'])

        fts = torch.stack(fts)

        return fts, targets

    def compute(self):
        path = self.project_path / 'results' / 'linear_probing' / self.layer
        path.mkdir(parents=True, exist_ok=True)

        accs = []
        for token in range(50):
            # Get training data
            fts, targets = self._get_training_data(token)

            # Train model
            lm = Ridge(alpha=1.0, solver='lsqr')
            lm.fit(fts.cpu().numpy(), targets.cpu().numpy())

            # Get testing data
            fts, targets = self._get_testing_data(token)

            # Test accuracy
            preds = lm.predict(fts.cpu().numpy())
            topk = np.argmax(preds, axis=1)
            acc = accuracy_score(targets, topk)
            accs.append(acc)

            # Save position of correct prediction
            pos = 999 - np.where(np.argsort(preds) == np.expand_dims(targets, axis=1))[1]
            np.save(path / f'pos_t{token}.npy', pos)

        # Save accuracy for token and layer
        np.save(path / f'acc.npy', accs)

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
        '-l', action='store', required=True,
        help='Layer index.'
    )

    args = parser.parse_args()
    project_path = Path(args.pp)
    data_path = Path(args.dp)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = LAYERS[int(args.l)]
    LinearProber('vit_b_32', layer, project_path, data_path, device).compute()
