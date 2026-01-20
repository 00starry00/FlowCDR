# Optimal Preference Transport for Cross-Domain Recommendation via Flow Matching (FlowCDR)

## Introduction

This repository provides the implementation of **FlowCDR** from the paper: *Optimal Preference Transport for Cross-Domain Recommendation via Flow Matching*.



## Requirements

- Python 3.7
- Pytorch
- Pandas
- Numpy
- Tqdm

## Dataset

The Amazon datasets we used: 
1. CDs and Vinyl: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz
2. Movies and TV: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz  
3. Books: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz

Put the data files in `./data/raw`.

Data process via:
```python
python entry.py --process_data_mid 1 --process_data_ready 1
```

## Experiments

Parameters settings:

- use_cuda: using GPU `1` or CPU as `0`
- task: different tasks within `1, 2 or 3`, default as `1`
- ratio: train/test ratio within `[0.8, 0.2], [0.5, 0.5] or [0.2, 0.8]`, default as `[0.8, 0.2]`
- exp_part: experiments with options `[None_CDR, CDR, ss_CDR, la_CDR, diff_CDR, Flow_CDR]`
- epoch: pre-training and CDR mapping training epoches, default as `10`
- seed: random seed, default as `1`
- root: root path, default as `./`
- save_path: path to save model files for base models and load model for CDRs, default as `./model_save_default/model.pth`
- flow_lr: learning rate of DiffCDR,default as `0.01`.


You can run models through:

```powershell
# Run the base models and augment model:
python entry.py --exp_part None_CDR 

# Run EMCDR and PTUPCDR
python entry.py --exp_part CDR

# Run  SSCDR
python entry.py --exp_part ss_CDR

# Run  LACDR
python entry.py --exp_part la_CDR

# Run  DiffCDR
python entry.py --exp_part diff_CDR

# run FlowCDR
python entry.py --exp_part flow_CDR

```


## Acknowledgement: 
Our code is based on [DiffCDR](https://github.com/breezeyuner/DiffCDR).

