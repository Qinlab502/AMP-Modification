# AMP-Modification
The AMP-Modification pipeline utilizes a fine-tuned protein language model for the modification of AMPs through the integration of a genetic algorithm. It selects active peptides through the property-based model for positive physicochemical properties and the contact-based model combined with two distinct contact map filter approaches for positive structure constraints. In this pipeline, we take LBD (lipopolysaccharide-binding domain), a type of AMP, as an example for the antimicrobial activity improvement based on both the physicochemical properties and the 3D structures.

![image](https://github.com/Qinlab502/AMP-Modification/blob/main/images/Fig.1.png)

## Model Installation
The protein language model we used for training can be downloaded through this link [(ESM2_t33_650M)](https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/main). Because of the large parameters, we have not put this model file in the models directory.

## Environment Requirement
All the environment dependency packages for pipeline running have been concluded in the 'environment.yml' file. You can download the file and created the environment via ```conda env create -f environment.yml```.
You can follow the pipeline both on CPU and GPU, the inclusion of the ```cudatoolkit==11.1.1``` dependency is needed if you have a GPU. You can check the device by ```torch.cuda.is_available()```.

## Notebooks
### Property-based model
We have fine-tuned a pretrained protein language model with LoRA for physicochemical property classification. To begin with, we [extract LBDs](./scripts/LBD_extraction_from_ALF.ipynb) from anti-lipopolysaccharide factor(ALF) for data augumentation. [This jupyter notebook](./scripts/property-based_model_with_lora.ipynb) will show you how to establish a fine-tuned model by LoRA with limited data. You can easily use this pipeline for other few shot learning.
