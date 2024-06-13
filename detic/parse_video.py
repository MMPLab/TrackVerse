# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import os
import time
import cv2
import tqdm
import sys
import av
import numpy as np

import torch
from torch.utils.data import DataLoader, IterableDataset

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import MetadataCatalog, get_clip_embeddings, reset_cls_test
from detic.predictor import BUILDIN_METADATA_PATH, BUILDIN_CLASSIFIER


class VideoReader(IterableDataset):
    def __init__(self, video_fn, frame_rate=None, max_size=None):
        self.video_fn = video_fn
        # container = cv2.VideoCapture(self.video_fn)
        # self.num_frames = int(container.get(cv2.CAP_PROP_FRAME_COUNT))
        # self.original_fps = container.get(cv2.CAP_PROP_FPS)
        container = av.open(self.video_fn)
        self.num_frames = container.streams.video[0].frames
        self.original_fps = float(container.streams.video[0].average_rate)
        self.skip = max(int(self.original_fps // frame_rate), 0)
        self.frame_rate = self.original_fps / self.skip
        self.max_size = max_size
        self.max_dur = 60

    def _frame_from_video(self):
        container = av.open(self.video_fn)
        frame_id = 0
        t_init = None
        for frame in container.decode(video=0):
            frame_ts = float(frame.pts * frame.time_base)
            frame = np.array(frame.to_image())

            if t_init is None:
                t_init = frame_ts
            if frame_ts - t_init > self.max_dur:
                break

            if frame_id % self.skip == 0:   # Skip frames to match desired framerate
                yield frame, frame_ts
            frame_id += 1

        # container = cv2.VideoCapture(self.video_fn)
        # while container.isOpened():
        #     success, frame = container.read()
        #     ts = container.get(cv2.CAP_PROP_POS_MSEC) / 1000.
        #     if t_init is None:
        #         t_init = ts
        #     if ts - t_init > self.max_dur:
        #         break
        #
        #     if success:
        #         if frame_id % self.skip == 0:   # Skip frames to match desired framerate
        #             # # Resize
        #             # if self.max_size is not None:
        #             #     max_size = max(frame.shape[0], frame.shape[1])
        #             #     width = int(frame.shape[1] * self.max_size/max_size)
        #             #     height = int(frame.shape[0] * self.max_size/max_size)
        #             #     frame = cv2.resize(frame, (width, height))
        #
        #             yield frame, ts
        #         frame_id += 1
        #     else:
        #         break

    def __iter__(self):
        return self._frame_from_video()

    def __len__(self):
        return self.num_frames


class BatchPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def __call__(self, batch, return_feats=False):
        # Apply pre-processing to image.
        single_img = batch.ndim == 3
        if single_img:
            batch = batch[None]

        # whether the model expects BGR inputs or RGB
        if self.input_format == "RGB":
            batch = batch[:, :, :, ::-1]

        height, width = batch[0].shape[:2]
        transform = self.aug.get_transform(batch[0])
        batch = [transform.apply_image(img) for img in batch]
        batch = [torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) for img in batch]
        inputs = [{"image": img, "height": height, "width": width} for img in batch]
        preds = self.model.inference(inputs, return_feats=return_feats)
        return preds[0] if single_img else preds


class DETIC(object):
    def __init__(self, detic_cfg, vocabulary, custom=None):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        if vocabulary == 'custom':
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = custom.split(',')
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        else:
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[vocabulary])
            classifier = BUILDIN_CLASSIFIER[vocabulary]

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")

        self.predictor = BatchPredictor(detic_cfg)
        reset_cls_test(self.predictor.model, classifier, num_classes)


def setup_detic_cfg(config_file, confidence_threshold, pred_all_class=True, opts=[], cpu=False):
    cfg = get_cfg()
    if cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def build_model(frame_max_size=640, confidence_thr=0.3):
    model_cfg = 'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
    model_weights = 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    dictionary = 'custom'
    custom = [l.strip().replace(',', ' ') for l in open('datasets/metadata/coco.txt')] + \
             [l.strip() for l in open('datasets/metadata/imagenet1k.txt')]

    opts = ['MODEL.WEIGHTS', model_weights, 'MODEL.MASK_ON', False]
    if frame_max_size is not None:
        opts.extend(['INPUT.MIN_SIZE_TEST', frame_max_size, 'INPUT.MAX_SIZE_TEST', frame_max_size])
    cfg = setup_detic_cfg(
        config_file=model_cfg,
        opts=opts,
        confidence_threshold=confidence_thr
    )
    model = DETIC(cfg, dictionary, custom=','.join(custom))
    return model


@torch.no_grad()
def extract_detections(model, loader, video, batch_size=64):
    all_detections = []
    for batch, timestamps in tqdm.tqdm(loader, total=video.num_frames // (video.skip * batch_size)):
        results = model.predictor(batch.numpy(), return_feats=True)

        results = [r['instances'] for r in results]
        for r, ts in zip(results, timestamps):
            r.remove('proposal_boxes')
            r.remove('pred_classes')
            r.remove('roi_feats')
            r.timestamp = ts[None].repeat(len(r))

            for k in r._fields:
                if k == 'pred_boxes':
                    r.set(k, r.get(k).tensor.half())
                else:
                    r.set(k, r.get(k).half())
        results = [r.to('cpu') for r in results]
        all_detections.extend(results)

    return all_detections


if __name__ == '__main__':
    db_dir = '../data/hdvila-100m/'
    batch_size = 64
    frame_rate = 8
    frame_max_size = 640
    threshold = 0.5

    model = build_model(frame_max_size, confidence_thr=threshold)
    all_clips_fn = sorted(glob.glob(f'{db_dir}/video_clips/*/*.mp4'))
    video_id = [fn.split('/')[-2] for fn in all_clips_fn]
    ids = sorted(list(set(video_id)))

    clips_fns = []
    for vid in ids:
        fns = [fn for fn, i in zip(all_clips_fn, video_id) if i == vid]
        fns = sorted(fns, key=lambda x: int(x.split('.')[-2]))
        clips_fns.extend(fns[len(fns)//2])

    for clip_i, video_fn in enumerate(clips_fns):
        video = VideoReader(video_fn, frame_rate=frame_rate, max_size=frame_max_size)
        video_loader = DataLoader(video, batch_size=batch_size, num_workers=0)

        print(f"Proc {clip_i}/{len(clips_fns)}: {video_fn} ({video.num_frames/video.original_fps} secs)")
        detections, t_els = extract_detections(model, video_loader, video, batch_size)

        output_fn = video_fn.replace(f'{db_dir}/video_clips', f'{db_dir}/detections').replace('.mp4', '.pth')
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        torch.save(detections, output_fn)

        t_proc = len(detections) / video.frame_rate
        eff_min_per_hour = t_els / (t_proc / 3600) / 60
        print(f'Processing Time: {eff_min_per_hour:.1f} processing minutes per input hour.', flush=True)

    ### Accessing results
    # len(all_detections) -> number frames in batch
    # len(results[i]) -> number bounding boxes in frame i
    # results[i].pred_boxes -> coordinates of bounding boxes in frame i (Ki x 4 Tensor)
    # results[i].scores -> objectness score of bounding boxes in frame i (Ki-dim Tensor)
    # results[i].pred_classes -> index of predicted class for each bounding box in frame i (Ki-dim Tensor)
    # results[i].feat -> ROIPool'ed feature for each bounding box in frame i (Ki x D x 7 x 7 Tensor)
    # model.metadata.thing_classes -> Class names
