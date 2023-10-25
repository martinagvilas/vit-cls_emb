from pathlib import Path
from PIL import Image

import pandas as pd
from torch.utils.data import Dataset


# Exclude categories of ImageNet that are not present in ImageNetS
CATS_EXCLUDE = [
    'n04356056', 'n04355933', 'n04493381', 'n02808440', 'n03642806', 
    'n03832673', 'n04008634', 'n03773504', 'n03887697', 'n15075141'
]


class ImagenetDatasetS(Dataset):
    def __init__(self, imagenet_path, partition='validation', n=5):
        self.path = Path(imagenet_path) / 'ImageNetS919'
        self.imgs_path = self.path / partition
        self.partition = partition
        self.n_imgs = n
        self.stim_info = self.get_stimuli_info()

    def get_stimuli_info(self):
        cats = self.get_category_info()
        file = self.path / f'{self.partition}_stim_info.csv'
        if file.exists():
            stim_info = pd.read_csv(file)
        else:
            stim_info = []
            cat_dirs = [d for d in self.imgs_path.iterdir() if d.is_dir()]
            for c in cat_dirs:
                imagenet_id = c.name
                if imagenet_id in CATS_EXCLUDE:
                    continue
                else:
                    idx = cats.loc[imagenet_id]['index']
                    cat = cats.loc[imagenet_id]['cat']
                    imgs_paths = [i for i in c.iterdir() if i.suffix == '.JPEG']
                    if len(imgs_paths) < self.n_imgs:
                        continue
                    else:
                        for i_idx, i in enumerate(imgs_paths[:self.n_imgs]):
                            i_name = i.name
                            stim_info.append([imagenet_id, idx, cat, i_name, i_idx])
            stim_info = pd.DataFrame(
                stim_info, 
                columns=['imagenet_id', 'index', 'cat', 'img_name', 'img_index']
            )
            stim_info.to_csv(file, index=None)
        return stim_info

    def get_category_info(self):
        cats_info = pd.read_csv(
            self.path / 'LOC_synset_mapping.txt',  sep='\t', header=None
        )
        cats_info[['imagenet_id', 'cat']] = cats_info[0].str.split(' ', n=1, expand=True)
        cats_info = cats_info.drop(columns=0).reset_index(drop=False)
        cats_info = cats_info.set_index('imagenet_id')
        return cats_info

    def __len__(self):
        return len(self.stim_info)

    def __getitem__(self, idx):
        item_info = self.stim_info.iloc[idx]
        item = {}
        item['imagenet_id'] = item_info['imagenet_id']
        item['index'] = item_info['index']
        item['cat'] = item_info['cat']
        item['img_index'] = item_info['img_index']
        img_path = self.imgs_path / item_info['imagenet_id'] / item_info['img_name']
        item['img'] = Image.open(img_path).convert('RGB')
        return item
    
    def get_item_by_concept(self, concept, img_idx):
        idx = (
            self.stim_info.loc[self.stim_info['imagenet_id'] == concept]
            .iloc[img_idx].name
        )
        return self.__getitem__(idx)
