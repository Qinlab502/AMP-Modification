# AMP-Modification
<font size=3>The AMP-Modification pipeline utilizes a fine-tuned protein language model for the modification of AMPs through the integration of a genetic algorithm. It selects active peptides through the property-based model for positive physicochemical properties and the contact-based model combined with two distinct contact map filter approaches for positive structure constraints. In this pipeline, we take LBD (lipopolysaccharide-binding domain), a type of AMP, as an example for the antimicrobial activity improvement based on both the physicochemical properties and the 3D structures.

![image](https://github.com/Qinlab502/AMP-Modification/blob/main/images/Fig.1.png)

## Model Installation
The protein language model we used for training can be downloaded through this link [(link)](https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/main). Since the large parameters, we have not put this model file in the models directory.

## Environment Requirement
All the environment dependency packages for pipeline running have been concluded in the 'environment.yml' file. You can download the file and created the environment via ```conda env create -f environment.yml```.
