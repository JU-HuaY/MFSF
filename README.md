# MFSF
<img src="./image/intro.png" alt="model"  width="70%"/>

## Installation
**Update**: Now the codes are compatible with PyTorch Geometric (PyG) >= 2.0.
### Dependency
The codes have been tested in the following environment:
Package  | Version
--- | ---
Python | 3.8.12
PyTorch | 1.10.1
CUDA | 11.3.1
PyTorch Geometric | **2.0.0**
RDKit | 2022.03
BioPython | 1.79
### Install via conda yaml file (cuda 11.3)

**Our model consists of three parts: Pretraining process, BS prediction and DTI prediction**
<img src="./image/method.png" alt="model"  width="70%"/>

## Pretrained extractor

The source file of the pre-trained model is in "Representation.py", we can run the "main.py" to pretrain the model. 

setting
```python
>>> train_dataset = Protein_pkl_Dataset(root_dir='BS_data/train')
>>> validation_dataset = Protein_pkl_Dataset(root_dir='BS_data/validation')
>>> model = Representation_model(3, 512, 256, 64, 128, batchs, device)
>>> model = model.to(device)
```
Then, the pretrained model that was saved will be used for BS and DTI prediction.

## BS prediction
We can run main.py. This file will also import "merge.py" and "model.py";

example of setting
```python
>>> dataset_name = "DNA"
>>> train_name = "DNA-646"
>>> test_name = "DNA-181"
>>> target_model_state_dict = model.state_dict()
>>> source_model_state_dict = torch.load("output/model/pretrain_new_2")
```
## DTI prediction
We can run main.py. This file will also import "merge.py" and "model.py";

example of setting    
```python
>>> data_select = "D_to_D"
>>> target_model_state_dict = model.state_dict()
>>> source_model_state_dict = torch.load("output/model/pretrain_new_2")
```
The specific data protocols are described in the file "data_merge.py";



