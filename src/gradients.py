from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from transformers import ViTImageProcessor

from src.datasets.imagenet import ImagenetDatasetS
from src.models.vit_edited import vit_base_patch32_224, vit_base_patch16_224


class ViTGradients():
    """Compute feature importance of image tokens.
    """
    def __init__(self, model_name, project_path, imgs_path, device='cpu'):
        self.model_name = model_name
        self.device = device

        self.project_path = Path(project_path)
        self.imgs_path = Path(imgs_path)
        self.res_path = project_path / 'results'
        self.grad_path = self.res_path / 'gradients' / self.model_name
        self.grad_path.mkdir(parents=True, exist_ok=True)

        self._load_model()

    def _load_model(self):
        if self.model_name == 'vit_b_32':
            self.model = vit_base_patch32_224(
                pretrained=True, pretrained_cfg=True
            ).to(self.device)
            self.n_tokens = 50
            source = 'google/vit-base-patch32-224-in21k'
        elif self.model_name == 'vit_b_16':
            self.model = vit_base_patch16_224(
                pretrained=True, pretrained_cfg=True
            ).to(self.device)
            source = 'google/vit-base-patch16-224-in21k'
            self.n_tokens = 197
        self.model.eval()
        self.img_transform = ViTImageProcessor.from_pretrained(source)
        return

    def _get_dataset(self):
        self.dataset = ImagenetDatasetS(self.imgs_path)
        return

    def compute(
            self, concept, img_idx, cat_idx=None, grad_type='cross_entropy',
            input_type='attn_probs'
        ):
        # Get data
        self._get_dataset()
        data = self.dataset.get_item_by_concept(concept, img_idx)
        self.cat_idx = cat_idx

        # Get image features
        img_ft = self.img_transform(data['img'], return_tensors="pt")
        img_ft = img_ft['pixel_values'].to(self.device)

        # Compute hidden states
        self.model.zero_grad()
        for b in range(12):
            self.model.blocks[b].attn.attn_probs = None
            self.model.blocks[b].attn.key_proj_vals = None
            self.model.blocks[b].attn_cls = None
        _ = self.model(img_ft)
        
        # Get gradients
        self.input_type = input_type
        if grad_type == 'cross_entropy':
            grads, attns = self._compute_cross_entropy(data)
        elif grad_type == 'cat_prob':
            grads, attns = self._compute_cat_prob(data)
        
        return grads, attns

    def _compute_cross_entropy(self, data):
        loss = CrossEntropyLoss(reduction='none')
        
        grads = {}
        attns = {}
        for b in range(12):
            if self.input_type == 'attn_probs':
                inp = self.model.blocks[b].attn.attn_probs
            elif self.input_type == 'key_proj_vals':
                inp = self.model.blocks[b].attn.key_proj_vals
            out = self.model.blocks[b].attn_cls

            # Collect grads of all tokens
            out = torch.squeeze(out)
            out = softmax(out, dim=1)
            target = torch.zeros(1, 1000).to(self.device)
            if self.cat_idx != None:
                target[0, self.cat_idx] = 1
            else:
                target[0, data['index']] = 1
            target = target.repeat(self.n_tokens, 1)
            l = loss(out, target)
            
            b_grads = []
            for t in range(self.n_tokens):
                grad = torch.autograd.grad(l[t], inp, retain_graph=True)
                if self.input_type == 'attn_probs':
                    b_grads.append(grad[0][0, :, t, :].detach())
                elif self.input_type == 'key_proj_vals':
                    b_grads.append(grad[0][0, t, :].detach())
            
            grads[b] = torch.stack(b_grads).transpose(0,1)
            
            if self.input_type == 'attn_probs':
                attns[b] = torch.squeeze(inp)
            else:
                continue
        
        return grads, attns

    def _compute_cat_prob(self, data):
        grads = {}
        attns = {}
        for b in range(12):
            if self.input_type == 'attn_probs':
                inp = self.model.blocks[b].attn.attn_probs
            elif self.input_type == 'key_proj_vals':
                inp = self.model.blocks[b].attn.key_proj_vals
            out = self.model.blocks[b].attn_cls

            # Collect grads of all tokens
            b_grads = []
            for t in range(self.n_tokens):
                cat_prob = torch.zeros(1, self.n_tokens, 1000).to(self.device)
                cat_prob[0, t, data['index']] = 1
                cat_prob.requires_grad_(True)
                cat_prob = torch.sum(cat_prob * out).to(self.device)

                # Compute gradients
                grad = torch.autograd.grad(cat_prob, inp, retain_graph=True)
                
                if self.input_type == 'attn_probs':
                    b_grads.append(grad[0][0, :, t, :].detach())
                elif self.input_type == 'key_proj_vals':
                    b_grads.append(grad[0][0, t, :].detach())
            
            grads[b] = torch.stack(b_grads).transpose(0,1)
            
            if self.input_type == 'attn_probs':
                attns[b] = torch.squeeze(inp)
            else:
                continue
        
        return grads, attns


class CheferGradients(ViTGradients):
    """Compute feature importance of image tokens using method from
    https://github.com/hila-chefer/Transformer-MM-Explainability
    """
    def __init__(self, model_name, project_path, imgs_path, device='cpu'):
        super().__init__(model_name, project_path, imgs_path, device)
        return

    def compute(self, concept, img_idx, cat_idx=None):

        # Get data
        self._get_dataset()
        data = self.dataset.get_item_by_concept(concept, img_idx)
        self.cat_idx = cat_idx

        # Get image features
        img_ft = self.img_transform(data['img'], return_tensors="pt")
        img_ft = img_ft['pixel_values'].to(self.device)

        # Compute hidden states
        self.model.zero_grad()
        for b in range(12):
            self.model.blocks[b].attn.attn_probs = None
        out = self.model(img_ft)
        del img_ft

        # Get one hot vector solution
        cat_prob = torch.zeros(1, 1000).to(self.device)
        if self.cat_idx != None:
            cat_prob[0, self.cat_idx] = 1
        else:
            cat_prob[0, data['index']] = 1
        cat_prob.requires_grad_(True)
        cat_prob = torch.sum(cat_prob * out).to(self.device)

        # Get gradients
        R = torch.eye(self.n_tokens, self.n_tokens).to(self.device)
        for b in range(12):
            cam = self.model.blocks[b].attn.attn_probs
            grad = torch.autograd.grad(cat_prob, cam, retain_graph=True)[0]  

            # Average over heads
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0).detach()

            # Apply self-attention rules
            r_add = torch.matmul(cam, R)
            R += r_add
        
        R = R[0, 1:]

        return R
