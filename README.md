# AMP-Modification
The AMP-Modification pipeline utilizes a fine-tuned protein language model for the modification of AMPs through the integration of a genetic algorithm. It selects active peptides through the property-based model for positive physicochemical properties and the contact-based model combined with two distinct contact map filter approaches for positive structure constraints. In this pipeline, we take LBD (lipopolysaccharide-binding domain), a type of AMP, as an example for the antimicrobial activity improvement based on both the physicochemical properties and the 3D structures.

![image](https://github.com/Qinlab502/AMP-Modification/blob/main/images/Workflow.jpeg)

## Model Installation
The protein language model we used for training can be downloaded through this link [(ESM2_t33_650M)](https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/main). Because of the large parameters, we have not put this model file in the models directory.

## Environment Requirement
All the environment dependency packages for pipeline running have been concluded in the 'environment.yml' file. You can download the file and created the environment via ```conda env create -f environment.yml```.
You can follow the pipeline both on CPU and GPU, the inclusion of the ```cudatoolkit==11.1.1``` dependency is needed if you have a GPU. You can check the device by ```torch.cuda.is_available()```.

## Training
### Property-based model
We have fine-tuned a pretrained protein language model with LoRA for physicochemical property classification. To begin with, we [extract LBDs](./scripts/extract_LBD.py) from anti-lipopolysaccharide factor(ALF) for data augumentation. [property.py](./scripts/property.py) will show you how to establish a fine-tuned model by LoRA with limited data. You can easily use this pipeline for other few shot learning.
#### Example
```
python property.py --model_path ../models/esm2_650M -i ../database/LBD_135.fasta
```
- **--model_path:**    path to the pretrained ESM2 model
- **-i:**    Path to the training fasta file

There are also optional arguments available for model training hyperparameters and LoRA parameters. For more detailed information, you can refer to the `parse_arguments` function. To save the trained model, make sure to include the argument *--save_model*.

### Contact-based model
In this section, we first trained a [contact prediction model](./scripts/contact.py) to predict the contact map of input sequences. A contact map is a two-dimensional representation of Cβ-Cβ distances between residue pairs, where a distance of less than 8 angstroms is considered a contact and assigned a value of 1. Following the procedure outlined by [Rao et al.](https://doi.org/10.1101/2020.12.15.422761), we trained the model and utilized a property-based approach to extract attention scores from each layer and head. After applying symmetrization, these scores were processed with average product correction and subsequently passed into a logistic regression model with a sigmoid activation function for training.
#### Example
```
python contact.py -i ../database/lbdb.cif
```
- **-i:**    Path to the input training pdb file

Other parameters are available in [contact.py](./scripts/contact.py). If you want to save the trained model, you have to set the argument *--save_model* and specify the output path using *-o* to indicate where the model should be saved, like 
```
python contact.py -i ../database/lbdb.cif --save_model -o './model/contact-based_model.pt'
```

Then we employ two distinct approaches to filter the pontential active maps. One is maps intersection to filter active sites and the other is maps flattening for linear projection. Positive maps will be filterer out by calculating their similarity to the LBDB map using Binary Cross Entropy Loss (BCELoss).

![image](https://github.com/Qinlab502/AMP-Modification/blob/main/images/contact_map_filter.png)

The script [mapfilter.py](./scripts/mapfilter.py) defines two functions implementing the approaches described above.

## Design
[design.py](./scripts/design.py) will help you to modificate AMPs with a genetic algorithm. We use property-based model as the fitness function to guide the production of the next generation. Active AMPs will be filtered through the contact-based model combined with two map filter approaches. The final step in the pipeline involved the calculation of BCELoss to determine the activity order. 

```
usage: design.py [-h] -i FASTA_FILE -o MUTATION_FASTA_OUTPUT [--num_parents_mating NUM_PARENTS_MATING] [--num_generations NUM_GENERATIONS] [--num_mutations NUM_MUTATIONS]

optional arguments:
  -h, --help                             Show this help message and exit
  -i, -fasta_file
                                         Path to the input fasta file.
  -o, --mutation_fasta_output
                                         Path to save the mutation fasta file.
  --num_parents_mating
                                         Number of parents mating (default=8).
  --num_generations
                                         Number of rounds to generate iteratively (default=10000).
  --num_mutations
                                         Number of mutation sites (default=3)
```
**Tips:** The inference process will automatically detect whether a GPU is available on the device. If no available GPU is detected, it will default to using the CPU.

## Contact
Zhiwei Qin(z.qin@bnu.edu.cn)\
Qiandi Gao(gaoqiandi@mail.bnu.edu.cn)
