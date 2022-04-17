import torch  
import cv2
import numpy as np
from torchvision import transforms
from model.model_clip import build_model


def init_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.48145466, 0.4578275, 0.40821073),
                        norm_std=(0.26862954, 0.26130258, 0.27577711)):                        
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        'test': transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ])
    }
    return tsfm_dict

def sample_frames(num_frames, vlen):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    return frame_idxs


def read_frames_cv2(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs

video_transforms = init_transform_dict()['test']
video_path = ''
num_frames = 4
video, idxs = read_frames_cv2(video_path, num_frames)
video = video_transforms(video).unsqueeze(0).cuda()
print(video.shape)

model_path = './MCQ_CLIP.pth'
model_clip = torch.load(model_path, map_location="cpu")
state_dict = model_clip['state_dict']

model = build_model(state_dict)
model = model.cuda()
video_features = model.encode_image(video)
video_features = video_features / video_features.norm(dim=-1, keepdim=True)
print(video_features.shape)