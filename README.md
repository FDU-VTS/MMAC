# MMAC

This repo covers the winning solutions from Team fdvts_mm in the MICCAI 2023 Myopic Maculopathy Analysis Challenge.


## Dataset
We download the datasets from [MMAC2023](https://codalab.lisn.upsaclay.fr/competitions/12441) and [DDR](https://github.com/nkicsl/DDR-dataset).

## Task 1. Classification of Myopic Maculopathy

```
cd MMAC_task1
```
**train a ResNet50 model**
```
python main.py --challenge 1 --model resnet50 --visname resnet50
```
