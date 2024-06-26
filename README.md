# [ImageNet-OOD: Deciphering Modern Out-of-distribution Detection Algorithms]()

## Setup
Download the following datasets: [ImageNet-1K](https://image-net.org/), [ImageNet-21k-P](https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), [ImageNet-R](https://github.com/hendrycks/imagenet-r), [ImageNet-C](https://github.com/hendrycks/robustness), and [OpenImage-O](https://github.com/haoqiwang/vim).

Create the conda environment
```
conda create -n imagenetood python=3.10.12
conda activate imagenetood
pip install -r requirements.txt
```

## Analysis
First, preprocess logits and features of a subset of ImageNet-1K with the following command
```
python preprocess.py --subset_file imagenet_random_200k.txt -
-imagenet_path [path of the imagenet-1k train set] --result_pat
h .
```
Next, generate the OOD scores for each of the datasets (both in-distribution and out-of-distribution) with the command
```
python generate_scores.py --dataset [ImageNet(any image net format dataset)/ImageNetOOD/OpenImageO] --root_path [path of the dataset] --subset_file [file that provide the subset (only used for ImageNetOOD and OpenImageO).] --result_file [output pickle file] --semantic [0 if the dataset non-semantic shift (ImageNet-1K/R/C/Sketch), 1 if the dataset is semantic shift (ImageNet-OOD, OpenImageO)]
```
Finally, use similarity_analysis.ipynb for analysis done in Figure 3 of the paper or obtain the OOD detection performance by running
```
python evaluate.py --in_pkl [pickle of in-distribution scores] --out_pkl [pickle of out-of-distribution scores]
``` 
