import os
import torch.nn as nn
import torch
import timm
from model.video_bridge_former import Video_Bridge_Former
from base import BaseModel
from utils.util import state_dict_data_parallel_fix
from transformers import AutoModel


class MCQ(BaseModel):

    def __init__(self,
                 args,
                 video_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal'):
        super().__init__()
        self.args = args
        self.video_params = video_params
        self.text_params = text_params
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        self.text_model = AutoModel.from_pretrained(text_params['model'], output_hidden_states=True)
        self.text_model.train()

        pretrained = video_params['pretrained']
        num_frames = video_params.get('num_frames', 4)

        arch_config = video_params.get('arch_config', 'base_patch16_224')
        if arch_config == 'base_patch16_224':
            if load_checkpoint in ["", None]:
                vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained).cuda()
                vit_model = torch.nn.parallel.DistributedDataParallel(vit_model, device_ids=[self.args.local_rank])
            model = Video_Bridge_Former(num_frames=num_frames)

        model.head = nn.Identity()
        model.pre_logits = nn.Identity()
        ftr_dim = model.embed_dim
        if load_checkpoint in ["", None]:
            vit_checkpoint = vit_model.state_dict()
            new_vit_dict = state_dict_data_parallel_fix(vit_checkpoint, model.state_dict())
            model.load_state_dict(new_vit_dict, strict=False)
        self.video_model = model
        self.video_model.fc = nn.Identity()

        # Project to a common embedding
        if projection == 'minimal':
            text_proj = nn.Sequential(nn.ReLU(),
                                     nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                     )
            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
            bridge_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )

        self.text_proj = text_proj
        self.vid_proj = vid_proj
        self.bridge_proj = bridge_proj

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint, map_location='cuda:{}'.format(self.args.local_rank))
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            self.load_state_dict(new_state_dict, strict=True)

    def set_device(self, device):
        self.device = device

    def forward(self, data):

        text_data = data['text']
        answer_data = data['answer']
        question_data = data['question']
        video_data = data['video']

        text_cls_embeddings, answer_cls_embeddings, question_embeddings, question_mask = \
            self.compute_text(text_data, answer_data, question_data)
        bridge_cls_embeddings, video_cls_embeddings = \
            self.compute_video(video_data, question_embeddings, question_mask)

        return text_cls_embeddings, answer_cls_embeddings, bridge_cls_embeddings, video_cls_embeddings

    def compute_text(self, text_data, answer_data, question_data):

        text_cls_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        answer_cls_embeddings = self.text_model(**answer_data).last_hidden_state[:, 0, :]

        text_cls_embeddings = self.text_proj(text_cls_embeddings)
        answer_cls_embeddings = self.text_proj(answer_cls_embeddings)

        question_embeddings = self.text_model(**question_data).hidden_states
        question_mask = question_data['attention_mask']

        return text_cls_embeddings, answer_cls_embeddings, question_embeddings, question_mask

    def compute_video(self, video_data, question_embeddings, question_mask):
        bridge_cls_embeddings, video_cls_embeddings = self.video_model(video_data, question_embeddings, question_mask)

        bridge_cls_embeddings = self.bridge_proj(bridge_cls_embeddings)
        video_cls_embeddings = self.vid_proj(video_cls_embeddings)

        return bridge_cls_embeddings, video_cls_embeddings


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
