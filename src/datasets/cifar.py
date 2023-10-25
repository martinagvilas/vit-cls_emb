from collections import defaultdict
from PIL import Image

from torchvision.datasets import CIFAR100


class MyCIFAR100(CIFAR100):
    def __init__(self, imgs_path, n=5):
        self.imgs_path = imgs_path
        super().__init__(imgs_path, train=False, download=False, transform=None)
        self.n_imgs = n
        self.stim_info = self._get_stimuli_info()
        return
    
    def _get_stimuli_info(self):
        stim_info = defaultdict(list)
        for img, label in zip(self.data, self.targets):
            label = str(label)
            if len(stim_info[label]) < self.n_imgs:
                stim_info[label].append(Image.fromarray(img))
        return stim_info
