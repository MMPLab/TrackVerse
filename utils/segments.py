import os
import tqdm
from enum import Enum


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms as T
from scenedetect import detect, AdaptiveDetector
import open_clip

from utils import misc as misc_utils
from utils import avio


class STATUS(Enum):
    SUCCESS = 0
    DONE = 1


class CartoonDetector:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load('assets/cartoon_detector.pth', map_location='cpu'))
        self.model.eval()
        self.model = self.model.to(self.device)

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

    @torch.no_grad()
    def is_cartoon(self, video_filepath, start_time, end_time):
        video = avio.VideoDB(video_filepath, start_time=start_time, max_dur=min(end_time-start_time, 8), frame_rate=2, transform=self.transform)
        loader = DataLoader(video, batch_size=16, num_workers=0, drop_last=False)
        cartoon_logits = []
        for frames, ts in loader:
            pred = self.model(frames.to(self.device)).cpu()
            cartoon_logits.append(pred)
        if cartoon_logits:
            return torch.cat(cartoon_logits).mean(0)[0] > 0.
        else:
            return False


class AestheticPredictor:
    def __init__(self, device='cpu'):
        self.device = device
        self.model, _, self.transform = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        self.predictor = nn.Linear(768, 1)
        self.predictor.load_state_dict(torch.load('assets/aesthetic_predictor_vit_l14_linear.pth'))
        self.model = self.model.to(self.device)
        self.predictor = self.predictor.to(self.device)
        self.model.eval()
        self.predictor.eval()

    @torch.no_grad()
    def get_aesthetic_score(self, video_filepath, start_time, end_time):
        video = avio.VideoDB(video_filepath, start_time=start_time, max_dur=min(end_time-start_time, 8), frame_rate=2, transform=self.transform)
        loader = DataLoader(video, batch_size=16, num_workers=0, drop_last=False)
        ascores = []
        for frames, ts in loader:
            x = self.model.encode_image(frames.to(self.device))
            x /= x.norm(dim=-1, keepdim=True)
            ascores.append(self.predictor(x))
        if ascores:
            return torch.cat(ascores).mean()
        else:
            return False


class SegmentExtractor(object):
    def __init__(self, video_dir, segm_dir, skip_cartoon_filter=False, skip_aesthetics_filter=False):
        # Input directory
        self.video_dir = video_dir

        # Output directory
        misc_utils.check_dirs(segm_dir)
        self.segm_dir = segm_dir
        self.segm_tracker = misc_utils.ProgressTracker(os.path.join(segm_dir, 'completed.txt'))

        # Filters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.skip_cartoon_filter = skip_cartoon_filter
        if not self.skip_cartoon_filter:
            self.cartoon_detector = CartoonDetector(self.device)
        self.skip_aesthetics_filter = skip_aesthetics_filter
        if not self.skip_aesthetics_filter:
            self.aesthetic_predictor = AestheticPredictor(self.device)

    def extract_segments(self, video_filepath):
        youtube_id = video_filepath.split('/')[-1][:11]
        segments_fn = f"{self.segm_dir}/{youtube_id[:2]}/{youtube_id}.txt"
        if self.segm_tracker.check_completed(youtube_id):
            return STATUS.DONE, segments_fn

        # Detect scene changes
        segments = detect(video_filepath, AdaptiveDetector())
        segments = [(misc_utils.ts2sec(ts[0]), misc_utils.ts2sec(ts[1])) for ts in segments]
        if len(segments) == 0:
            video = avio.VideoDB(video_filepath)
            segments = [(float(video.reader.start_time), float(video.reader.start_time + video.reader.duration))]

        # Trim scenes larger than 10 min
        segments = [(segm_st, segm_end) if segm_end - segm_st < 600 else (segm_st, segm_st+600)
                    for segm_st, segm_end in segments]

        to_rm = {}
        for segm_id, (segm_st, segm_end) in enumerate(tqdm.tqdm(segments)):
            # Filter out scenes smaller than 3sec
            if segm_end - segm_st < 3:
                to_rm[segm_id] = 'Length'
                continue
            # Filter out cartoons
            if not self.skip_cartoon_filter and \
                    self.cartoon_detector.is_cartoon(video_filepath, segm_st, segm_end):
                to_rm[segm_id] = 'Cartoon'
                continue
            # Filter out low aesthetics
            if not self.skip_aesthetics_filter and \
                    self.aesthetic_predictor.get_aesthetic_score(video_filepath, segm_st, segm_end) < 4.0:
                to_rm[segm_id] = 'Low Aesthetics'
                continue

        # Save to file
        segments_filtered = [segm for segm_id, segm in enumerate(segments) if segm_id not in to_rm]
        os.makedirs(os.path.dirname(segments_fn), exist_ok=True)
        open(segments_fn, 'w').write('\n'.join([f"{segm_st},{segm_end}" for segm_st, segm_end in segments_filtered]))

        self.segm_tracker.add(youtube_id)
        return STATUS.SUCCESS, segments_fn
