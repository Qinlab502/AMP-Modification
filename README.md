# AMP-Modification
The AMP-Modification pipeline utilizes a fine-tuned protein language model for the modification of AMPs through the integration of a genetic algorithm. It selects active peptides through the property-based model for positive physicochemical properties and the contact-based model combined with two distinct contact map filter approaches for positive structure constraints. In this pipeline, we take LBD (lipopolysaccharide-binding domain), a type of AMP, as an example for the antimicrobial activity improvement based on both the physicochemical properties and the 3D structures.

![image](https://github.com/Qinlab502/AMP-Modification/blob/main/images/Fig.1.png)

## Model Installation
The protein language model we used for training can be downloaded through this link [(ESM2_t33_650M)](https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/main). Because of the large parameters, we have not put this model file in the models directory.

## Environment Requirement
All the environment dependency packages for pipeline running have been concluded in the 'environment.yml' file. You can download the file and created the environment via ```conda env create -f environment.yml```.
You can follow the pipeline both on CPU and GPU, the inclusion of the ```cudatoolkit==11.1.1``` dependency is needed if you have a GPU. You can check the device by ```torch.cuda.is_available()```.

## Notebooks
### Property-based model  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Qinlab502/scripts/property-based_model_with_lora.ipynb).
We have fine-tuned a pretrained protein language model with LoRA for physicochemical property classification. To begin with, we [extract LBDs](./scripts/LBD_extraction_from_ALF.ipynb) from anti-lipopolysaccharide factor(ALF) for data augumentation. [This jupyter notebook](./scripts/property-based_model_with_lora.ipynb) will show you how to establish a fine-tuned model by LoRA with limited data. You can easily use this pipeline for other few shot learning.

### Contact-based model
[This jupyter notebook](./scripts/contact-based_model.ipynb) will show you how to finetune a contact prediction head with only one pbd file.

### Contact map filter
We employ two distinct approaches to filter the pontential active maps. One is maps intersection to filter active sites and the other is maps flattening for linear projection. You can follow the pipeline in [this jupyter notebook](./scripts/contact_map_filter.ipynb).

### Genetic Algorithm
[This jupyter notebook](./scripts/Genetic_Algorithms.ipynb) will help you to modificate AMPs with a genetic algorithm. We use property-based model as the fitness function to guide the production of the next generation. Active AMPs will be filtered through the contact-based model combined with two map filter approaches. The final step in the pipeline involved the calculation of BCELoss to determine the activity order. 
