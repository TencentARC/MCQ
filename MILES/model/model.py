import torch
import timm
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils.util import state_dict_data_parallel_fix
from transformers import AutoModel, AutoTokenizer
from model.video_former import Video_Former


class MCQ(BaseModel):
    def __init__(self,
                 video_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal'):
        super().__init__()

        self.video_params = video_params
        self.text_params = text_params
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()
         
        pretrained = video_params['pretrained']
        num_frames = video_params.get('num_frames', 4)
        arch_config = video_params.get('arch_config', 'base_patch16_224')
        vit_init = video_params.get('vit_init', 'imagenet-21k')
        if arch_config == 'base_patch16_224':
            model = Video_Former(num_frames=num_frames)
        else:
            raise NotImplementedError

        model.head = nn.Identity()
        model.pre_logits = nn.Identity()
        ftr_dim = model.embed_dim
        self.video_model = model

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        # Project to a common embedding
        if projection == 'minimal':
            text_proj = nn.Sequential(nn.ReLU(),
                                     nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                     )

            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
        self.text_proj = text_proj
        self.vid_proj = vid_proj

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            self.load_state_dict(new_state_dict, strict=False)

    def set_device(self, device):
        self.device = device

    def forward(self, data, return_embeds=True):

        text_data = data['text']
        video_data = data['video']

        text_embeddings = self.compute_text(text_data)
        video_embeddings = self.compute_video(video_data)

        if return_embeds:
            return text_embeddings, video_embeddings

        return sim_matrix(text_embeddings, video_embeddings)


    def compute_text(self, text_data):
        if self.text_params['model'].startswith('distilbert') or self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError
        text_embeddings = self.text_proj(text_embeddings)
        return text_embeddings

    def compute_video(self, video_data):
        video_embeddings = self.video_model(video_data)
        video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

if __name__ == "__main__":
    pass
