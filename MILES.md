# MILES: Visual BERT Pre-training with Injected Language Semantics for Video-text Retrieval

[Paper](https://arxiv.org/abs/2204.12408) | [Pre-trained Model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuyingge_connect_hku_hk/EewsJ8SvaetNjHBnaopKelkBpIhyARHKoHAFkHm9uAZhGA?e=K6XXEI)
![image](https://github.com/TencentARC/MCQ/blob/main/demo/MILES/MILES.jpg?raw=true)


## Main Results on Downstream Tasks
### Text-to-video Retrieval on MSR-VTT
![image](https://github.com/TencentARC/MCQ/blob/main/demo/MILES/msrvtt.png?raw=true)
### Text-to-video Retrieval on MSVD, LSMDC and DiDeMo
![image](https://github.com/TencentARC/MCQ/blob/main/demo/MILES/msvd.png?raw=true)

## Visualization
### Local Visual Semantics Capture
We visualize the self- attention map from the video encoder through computing the self-attention of the [CLS] token in the last block. Our pre-trained model pays high attention to those significant local regions in the video.

![image](https://github.com/TencentARC/MCQ/blob/main/demo/MILES/MILES_vis_self.jpg?raw=true)
### Fine-grained Video-text Alignment 
We visualize the cross-modality alignment between text and video tokens by calculating the similarity map between features embedded from the text encoder and video encoder. Our pre-trained model aligns words with corresponding visual regions accurately.

![image](https://github.com/TencentARC/MCQ/blob/main/demo/MILES/MILES_vis_cross.jpg?raw=true)


## Pre-trained Model
Our pre-trained model can be downloaded in [Pre-trained Model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuyingge_connect_hku_hk/EewsJ8SvaetNjHBnaopKelkBpIhyARHKoHAFkHm9uAZhGA?e=K6XXEI), which contains the weights of Video Encoder and Text Encoder.
## Downstream Retrieval (Zero-shot on MSR-VTT)
 1. Download our pre-trained model in [Pre-trained Model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuyingge_connect_hku_hk/EewsJ8SvaetNjHBnaopKelkBpIhyARHKoHAFkHm9uAZhGA?e=K6XXEI).
 
 3. Load the pre-trained model in  "configs/zero_msrvtt_4f_i21k_MILES.json".
     ```
    bash sctripts/test_retrieval_MILES.sh
    ```


## Acknowledgement
Our code is based on the implementation of "Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval" <https://github.com/m-bain/frozen-in-time.git>.

## Citation
If our code is helpful to your work, please cite:
```
@article{ge2022miles,
  title={MILES: Visual BERT Pre-training with Injected Language Semantics for Video-text Retrieval},
  author={Ge, Yuying and Ge, Yixiao and Liu, Xihui and Wang, Alex Jinpeng and Wu, Jianping and Shan, Ying and Qie, Xiaohu and Luo, Ping},
  journal={arXiv preprint arXiv:2204.12408},
  year={2022}
}
```
