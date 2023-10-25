import cv2
from PIL import Image
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.datasets.imagenet import ImagenetDatasetS
from src.models.load import load_vit


class Vis():
    def __init__(self, project_path, dataset_path, model, device):
        # Get paths
        self.project_path = Path(project_path)
        self.res_path = self.project_path / 'results'

        # Get model image processor
        self.model = model
        self.device = device
        self._get_img_processor()

        # Get stimuli information
        self.dataset_path = Path(dataset_path)
        dataset = ImagenetDatasetS(self.dataset_path)
        self.imgs_path = dataset.imgs_path
        self.segmentations_path = dataset.path / 'validation-segmentation'
        self.stim_info = dataset.stim_info
        self.concepts = self.stim_info['imagenet_id'].unique().tolist()

        # Get segmentation mapping
        f = dataset_path / 'data/categories/ImageNetS_categories_im919.txt'
        seg_map = pd.read_csv(f, header=None)
        self.seg_map = {row[0]:idx+1 for idx, row in seg_map.iterrows()}

        return
    
    def _get_img_processor(self):
        """Load image processor and get image dimensions.
        """
        
        # Load model
        _, _, _, self.processor = load_vit(
            self.model, self.device, self.project_path, return_transform=True
        )

        # Get image dimension
        if '32' in self.model:
            self.img_dim = 7
        elif '16' in self.model:
            self.img_dim = 14
        
        return

    def get_array(self, concept, idx):
        """Get image as array.

        Parameters
        ----------
        concept : str
            Imagenet id.
        idx : int
            Image id of Imagenet-S.

        Returns
        -------
        np.array
            Image of concept and idx.
        """
        file = self.imgs_path / concept / (
            self.stim_info.loc[self.stim_info['imagenet_id'] == concept]
            .iloc[idx]['img_name']
        )
        img = Image.open(file).convert('RGB')
        img = self.processor(img)
        try:
            img = np.transpose(img['pixel_values'][0], (1, 2, 0))
        except:
            img = img
        img = (img - img.min()) / (img.max() - img.min())
        return img
    
    def get_pil(self, concept, idx):
        """Get image as PIL Image.

        Parameters
        ----------
        concept : str
            Imagenet id.
        idx : int
            Image id of Imagenet-S.

        Returns
        -------
        PIL Image
            Image of concept and idx.
        """
        img = self.get_array(concept, idx)
        img = Image.fromarray((img * 255).astype(np.uint8))
        return img
    
    def get_token(self, concept, idx, token_idx, as_pil=True):
        """Return token from image.

        Parameters
        ----------
        concept : str
            Imagenet id.
        idx : int
            Image id of Imagenet-S.
        token_idx : int
            Index of token in image.
        as_pil : bool, optional
            Whether to return the token as image, by default True.

        Returns
        -------
        np.array or PIL Image
            Token of image.
        """
        # Get token position
        row_idx = token_idx // self.img_dim
        col_idx = token_idx % self.img_dim

        # Split image
        img = self.get_array(concept, idx)
        token = np.split(img, self.img_dim, axis=0)[row_idx]
        token = np.split(token, self.img_dim, axis=1)[col_idx]

        # Return token
        if as_pil:
            return Image.fromarray((token * 255).astype(np.uint8)).resize((100,100))
        else:
            return token

    def get_segmentation(self, concept, idx):
        """Get class segmentation.

        Parameters
        ----------
        concept : str
            Imagenet id.
        idx : int
            Image id of Imagenet-S.

        Returns
        -------
        np.array
            Mask of class in the image.
        """

        # Get segmentation information
        f = self.stim_info.loc[self.stim_info['imagenet_id'] == concept]
        f = f"{Path(f.iloc[idx]['img_name']).stem}.png"
        sgt = Image.open(self.segmentations_path / concept / f)
        
        # Get mask class 
        mask = np.array(sgt)
        mask = mask[:, :, 1] * 256 + mask[:, :, 0]
        mask = (mask == self.seg_map[concept]).astype(int)

        # Resize mask
        mask = cv2.resize(
            mask, dsize=(self.img_dim, self.img_dim), 
            interpolation=cv2.INTER_NEAREST
        )
        return mask

    def mask(self, concept, idx, weights, prepro='normalize_minmax', invert=False):
        """Generate heatmap over image from weights.
        Code adapted from https://github.com/hila-chefer/Transformer-MM-Explainability

        Parameters
        ----------
        concept : str
            Imagenet id.
        idx : int
            Image id of Imagenet-S.
        weights : torch Tensor
            Weights to plot the heatmap. Dimension should be same as the number
            of image tokens.
        prepro : str, optional
            Wheter to preprocess the weights, by default 'normalize_minmax'. 
            Can be one of the following: 'normalize_minmax' to scale the
            weights in the range of 0-1, or 'normalize_decoding' to scale the
            weights with respect to the number of classes in Imagenet.
        invert : bool, optional
            Whether to invert the weights, by default False.

        Returns
        -------
        np.array
            Image with heatmap.
        """

        # Preprocess weights if needed 
        if prepro == 'normalize_minmax':
            weights = (weights - weights.min()) / (weights.max() - weights.min())
        elif prepro == 'normalize_decoding':
            weights = weights / 1000
        elif prepro == None:
            pass
        
        if invert == True:
            weights = - weights
            print('inverted')

        # Transform weights into same size as image
        weights = weights.reshape(1, 1, self.img_dim, self.img_dim).float()
        weights = torch.nn.functional.interpolate(weights, size=224, mode='bilinear')
        weights = torch.squeeze(weights).detach().numpy()

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * weights), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        # Get image
        img = self.get_array(concept, idx)
        
        # Apply heatmap to image
        vis = heatmap + np.float32(img)
        vis = vis / np.max(vis)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

        return vis
