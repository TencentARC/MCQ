# Bridging Video-text Retrieval with Multiple Choice Questions, CVPR 2022 [[Paper]](https://arxiv.org/pdf/2201.04850.pdf)

![image](https://github.com/TencentARC/MCQ/blob/main/demo/MCQ.jpg?raw=true)

## Dependencies and Installation
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.7](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo

    ```bash
    git clone https://github.com/TencentARC/MCQ.git
    cd MCQ
    ```

1. Install dependent packages

    ```bash
    pip install -r requirements.txt
    ```
    
## Pre-training
1. Download WebVid-2M (see https://github.com/m-bain/webvid), and put the dataset under the folder "data".

2. Download CC3M (see https://ai.google.com/research/ConceptualCaptions/download), and put the dataset under the folder "data".

3. Download the captions of WebVid-2M and CC3M with the extracted noun and verb phrases in <https://drive.google.com/drive/folders/1thdilupXvb14B6QOTm_AGwuSjQh3Kfkh?usp=sharing>, and put them under the folder "meta_data".

4. Download the DistilBERT base model from Hugging Face in <https://huggingface.co/distilbert-base-uncased> or in <https://drive.google.com/drive/folders/1WFWyTFFOCEK0P5zvt2aQYX77XK9p9MYc?usp=sharing>.

5. We adopt the curriculum learning to train the model, which pre-trains the model on the image dataset CC3M and
video dataset WebVid-2M using 1 frame, and then on the video dataset WebVid-2M using 4 frames.
    - Run "bash sctripts/train_1frame_mask_noun.sh" and get model*.
    - Run "bash sctripts/train_4frame_mask_noun.sh" with model* loaded in "configs/dist-4frame-mask-noun.json", and get model**.
    - Run "bash sctripts/train_4frame_mask_verb.sh" with model** loaded in "configs/dist-4frame-mask-verb.json", and get model***.
6. Our repo adopts Multi-Machine and Muiti-GPU training, with 32 A100 GPU for 1-frame pre-training and 40 A100 GPU for 4-frame pre-training.

## Downstream Retrieval (Zero-shot on MSR-VTT)
1. Download MSR-VTT "wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -P data; unzip data/MSRVTT.zip -d data".

2. Load the model in  "configs/zero_msrvtt_4f_i21k.json".

3. Run "bash sctripts/test_retrieval.sh".

## License


## Acknowledgement
Our code is based on the implementation of "Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval" <https://github.com/m-bain/frozen-in-time.git>.

## Citation
If our code is helpful to your work, please cite:
```
@article{ge2022bridgeformer,
  title={BridgeFormer: Bridging Video-text Retrieval with Multiple Choice Questions},
  author={Ge, Yuying and Ge, Yixiao and Liu, Xihui and Li, Dian and Shan, Ying and Qie, Xiaohu and Luo, Ping},
  journal={arXiv preprint arXiv:2201.04850},
  year={2022}
}
```
