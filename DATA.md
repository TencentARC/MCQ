
# Data Preparation
We utilize two datasets for pre-train: WebVid-2M and CC3M.

For downstream datasets, we provide the MSR-VTT benchmark for evaluation.

We train models on these raw videos directly, instead of using off-line extracted features. We do not distribute datasets because of the license issue. Please download these datasets by yourself with following instructions:

## Download Pre-training Datasets
1. Download WebVid-2M (see https://github.com/m-bain/webvid), and put the dataset under the folder "data/WebVid".

2. Download CC3M (see https://ai.google.com/research/ConceptualCaptions/download), and put the dataset under the folder "data/CC3M".

## Download Captions with Extracted Noun and Verb Phrases for Pre-training
1. Download the captions of WebVid-2M and CC3M with the extracted noun and verb phrases in [meta_data](https://drive.google.com/drive/folders/1thdilupXvb14B6QOTm_AGwuSjQh3Kfkh?usp=sharing), and put them under the folder "meta_data".

## Download Downstream Datasets
1. Download MSR-VTT "wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -P data; unzip data/MSRVTT.zip -d data".

