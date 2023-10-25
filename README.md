# Analyzing Vision Tranformers in Class Embedding Space
This is the code accompanying the work "Analyzing Vision Transformers for Image
Classification in Class Embedding Space" by Martina Vilas, Timothy Schaumlöffel,
and Gemma Roig. Accepted at NeurIPS 2023.


## :wrench: Running the experiments

### Step 1: Get a local working copy of this code
__1.1.__ Clone this repository in your local machine.

__1.2.__ Install the required software using conda, by running:
```
conda create --name vit-cls python=3.9
conda activate vit-cls
pip install -r requirements.txt
pip install .
```

### Step 2: Download the dataset and modell checkpoints
__2.1.__ Download the ImageNet-S dataset from [here](https://github.com/LUSSeg/ImageNet-S).

__2.2.__ Download the stimuli info file from [here](https://drive.google.com/drive/folders/1bkJeOGMxU2Ta0CrtKeY9JBLArwmQM9mu?usp=sharing), and place it inside the `ImageNet-S/ImageNetS919`
folder downloaded in the previous step.

__2.3.__ Download the model checkpoint folder from [here](https://drive.google.com/drive/folders/1bkJeOGMxU2Ta0CrtKeY9JBLArwmQM9mu?usp=sharing), and place it inside the project folder.

### Step 3: Run experiments for extracting code
__3.1.__ Project hidden states to class embedding space and save key coefficients, by running:
```
python extractor.py -pp {PATH TO SOURCE CODE} -dp {PATH TO DATASET} -m {MODEL} -pretrained
```
- The model can be one of `vit_b_32`, `vit_b_16`, `vit_large_16`, `vit_cifar_16`, `vit_miil_16`, `deit_ensemble_16` (_Refinement_ model) and `vit_gap_16`.
- You can reproduce the results of the random model by removing the `-pretrained` flag.


__3.2.__ Run attention perturbation studies, by:
```
python perturbation/attn_perturbation.py -pp {PATH TO SOURCE CODE} -dp {PATH TO DATASET} -m vit_b_32 -pt {PERTURBATION TYPE}
```
- Perturbation type can be one of `self_only` or `no_cls`.

__3.3.__ Run context perturbation studies, by:
```
python perturbation/tokens_perturbation.py -pp {PATH TO SOURCE CODE} -dp {PATH TO DATASET} -m vit_b_32 -mt {MASK TYPE}
```
- Mask type can be one of `context` or `class label`.

__3.4.__ Run memory extractor, by:
```
python memories.py -pp {PATH TO SOURCE CODE} -dp {PATH TO DATASET} -m {MODEL} -lt {LAYER TYPE}
```
- Layer type can be one of `attn` or `mlp`.

__3.5.__ Run comparison with linear probing studies, by:
```
python linear_probing/prober.py -pp {PATH TO SOURCE CODE} -dp {PATH TO DATASET} -l {LAYER INDEX}
```

### Step 4: Reproduce the results
After running the above code, 
head to the notebooks section to reproduce and visualize the reported results.

## :paperclip: Citation
Please cite this work as: