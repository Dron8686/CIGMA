# CIGMA: Causal-Inspired Invariant Graph Matching with Multi-View Distillation for Predicting Herb-Symptom Associations
<img width="2052" height="937" alt="image" src="https://github.com/user-attachments/assets/8fedd16e-c1f3-4e84-b3ed-e3a9853e58da" />

The project files are organized as follows:
* `main_cv.py` and `main_indep.py`: Scripts used for cross-validation and independent test training.
* `finetune_cv.py` and `finetune_indep.py`: Scripts executed for the fine-tuning stage.
* `model_cv.py` and `model_indep.py`: Scripts containing the model architecture definitions.
* `parameters.py`: Configuration file for setting model and training hyperparameters.
* `save_train_val_test_data.py`: Script used for generating and saving the dataset splits.
* `processed data/`: Directory storing the processed data loaders (`.pth`) and edge indices (`.txt`). 
  * This directory contains subfolders for specific evaluation scenarios: `warm_start`, `cold_starts`, `cold_start_herb`, and `cold_start_symptom`. 
  * The data is divided into files for 5-fold cross-validation and independent testing.

## Data Sources
The datasets used in this repository originate from the following studies:
1. **Network medicine framework reveals generic herb-symptom effectiveness of traditional Chinese medicine**, *Science Advances*, 2023. (https://www.science.org/doi/full/10.1126/sciadv.adh0215)
2. **Revealing Herb-Symptom Associations and Mechanisms of Action in Protein Networks Using Subgraph Matching Learning**, *IEEE Journal of Biomedical and Health Informatics*, 2025. (https://ieeexplore.ieee.org/abstract/document/10938560)
