import os
import glob
import argparse
import math
import time
import re
import gzip
import json
from collections import Counter, defaultdict
import submitit
import tqdm

import numpy as np
import PIL.Image
import multiprocessing as mp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms as T

from scenedetect import detect, AdaptiveDetector
from bytetrack.byte_tracker import BYTETracker

from utils import misc as misc_utils, avio
from utils import detic as detic_utils
from utils import youtube as yt_utils
from utils.misc import ProgressTracker

class DETIC_CFG:
    frame_size = 480
    conf = 0.1
    nms = 0.7
    class_prompts = "assets/lvis-prompts.txt"   # TSV file of class names and prompts

class BYTETRACK_CFG:
    frame_rate = 16
    track_thresh = 0.55             # tracking confidence threshold
    track_iou_low_thresh = 0.5      # tracking confidence threshold
    match_thresh = 0.45             # matching threshold for tracking
    motion_weight = 0.4             # how much to weight motion information versus appearance information
    track_buffer = 24               # buffer size to find lost tracks
    mot20 = False


def parse_arguments():
    parser = argparse.ArgumentParser(description='youtube video downloader')
    parser.add_argument("--slurm", default=False, action="store_true")
    parser.add_argument("--partition", default='research')
    parser.add_argument("--world_size", default=1, type=int, help="scheduling chunks")
    parser.add_argument("--rank", default=0, type=int, help="scheduling chunk id")

    parser.add_argument('--base_dir', default='./TrackVerseDB',
                        help='Dataset directory')
    parser.add_argument('--yid_index_fn', default="assets/trackverse-yids-all.txt",
                        help='index of youtube ids to download.')
    parser.add_argument('--dataset_domain', default="LVIS", help='The class domain of the dataset.')
    parser.add_argument('--min_track_area', type=float, default=0.1, help='filter out tiny boxes')
    parser.add_argument('--min_track_len', type=float, default=3., help='filter out small tracks')
    return parser.parse_args()


@torch.no_grad()
def get_max_batch_size(detector, frame_size):
    bs = int(128 / 0.75)
    while bs > 0:
        try:
            inputs = torch.randn((bs, frame_size[0], frame_size[1], 3))
            detector.predictor(inputs.numpy())
            return int(bs * 0.75)
        except RuntimeError as e:
            if "out of memory" in str(e):
                bs = int(bs * 0.75)
            else:
                raise e
        except AssertionError as e:
            continue
    return 1


class ObjectTracksManager(object):
    def __init__(self, output_dir, class_desc):
        self.output_dir = output_dir
        self.class_prompts = class_desc
        self.tracks_saved = 0
        self.parse_label = lambda t: t.replace('photo of a ', '').split(' (also')[0]

    def construct_meta_dict(self, youtube_id, track, video_size):
        obj_scores = np.array([float(h['score'].item()) for h in track.history])
        cls_scores = np.stack([h['logit'].numpy() for h in track.history])
        wcls_scores = cls_scores * obj_scores[:, None]
        top10_idx = np.argsort(-wcls_scores.mean(0)[:-1])[:10]
        top10_desc = [self.parse_label(self.class_prompts[lbl]) for lbl in top10_idx]

        track_ts = [h['ts'] for h in track.history]
        track_bbox = [h['bbox'].tolist() for h in track.history]
        track_id = self.get_hash(youtube_id, track_ts, track_bbox)
        track_fn = f"{youtube_id[:2]}/{youtube_id}-{top10_desc[0].replace(' ', '-')}-{track_id}.mp4"
        meta_dict = {
            'yid': youtube_id,
            'tid': track_id,
            'fn': track_fn,
            'video_size': video_size,
            'track_ts': track_ts,
            'track_bbox': track_bbox,
            'top10_lbl': top10_idx.tolist(),
            'top10_desc': top10_desc,
            'top10_logit_mu': cls_scores.mean(0)[top10_idx].tolist(),
            'top10_logit_std': cls_scores.std(0)[top10_idx].tolist(),
            'top10_wlogit_mu': wcls_scores.mean(0)[top10_idx].tolist(),
            'top10_wlogit_std': wcls_scores.std(0)[top10_idx].tolist(),
        }
        return meta_dict

    @staticmethod
    def get_hash(vid, ts, bboxes):
        import hashlib  # unlike hash(), hashlib uses a constant seed for creating hash codes.
        ts = tuple([t for t in ts])
        boxes = tuple([tuple(b) for b in bboxes])
        return hashlib.md5(f"{vid}-{str(ts)}-{str(boxes)}".encode('utf-8')).hexdigest()

    def save_tracks_meta(self, tracks, vid, video_size):
        meta_json = os.path.join(self.output_dir, vid[:2], f"{vid}-meta.jsonl.gzip")
        misc_utils.check_dirs(os.path.dirname(meta_json))

        with gzip.open(meta_json, 'at') as fp:
            for track in tracks:
                track_meta = self.construct_meta_dict(vid, track, video_size)
                fp.write(json.dumps(track_meta) + '\n')
                self.tracks_saved += 1


class ObjectTracksParser(object):
    def __init__(self, base_dir, yid_index_fn, dataset_domain, detic_cfg, bytetrack_cfg, min_track_area=0.1, min_track_len=3., world_size=1, rank=0):
        self.base_dir = base_dir
        self.index_fn = yid_index_fn
        self.dataset_domain = dataset_domain
        self.world_size = world_size
        self.rank = rank

        self.frame_size = detic_cfg.frame_size
        self.frame_rate = bytetrack_cfg.frame_rate
        self.min_track_area = min_track_area
        self.min_track_len = min_track_len
        self.class_prompts = [l.strip() for l in open(detic_cfg.class_prompts)]
        self.detector = detic_utils.build_detic(
            self.class_prompts,
            detic_cfg.frame_size,
            detic_cfg.nms,
            detic_cfg.conf,
            gpu_id=0
        )
        self.tracker = BYTETracker(
            bytetrack_cfg.track_thresh,
            bytetrack_cfg.track_iou_low_thresh,
            bytetrack_cfg.match_thresh,
            bytetrack_cfg.frame_rate,
            bytetrack_cfg.track_buffer,
            bytetrack_cfg.motion_weight,
            bytetrack_cfg.mot20
        )

        # Output directories
        self.videos_dir = os.path.join(self.base_dir, 'videos_mp4')
        self.segments_dir = os.path.join(self.base_dir, 'videos_segm')
        self.tracks_meta_dir = os.path.join(self.base_dir, 'tracks_meta', self.dataset_domain)
        misc_utils.check_dirs(self.tracks_meta_dir)
        self.progress_tracker = misc_utils.ProgressTracker(os.path.join(self.tracks_meta_dir, 'completed.txt'))

    def scheduled_jobs(self):
        for job_id, ln in enumerate(open(self.index_fn)):
            youtube_id = ln.strip()
            segm_filepath = os.path.join(self.segments_dir, youtube_id[:2], f"{youtube_id}.txt")
            if len(youtube_id) != 11 or not misc_utils.check_file(segm_filepath):
                continue
            if job_id % self.world_size == self.rank:
                segments = [ln.strip().split(',') for ln in open(segm_filepath, "r")]
                segments = [(float(start), float(end)) for start, end in segments]
                yield job_id, youtube_id, segments

    @torch.no_grad()
    def parse_object_tracks(self, video_filepath, segment, batch_size, job_id):
        segm_start, segm_end = segment
        youtube_id = video_filepath.split('/')[-1][:11]
        objtrack_manager = ObjectTracksManager(self.tracks_meta_dir, class_desc=self.class_prompts)

        print(f'[{job_id}][{youtube_id}] Start parsing segment {segment}.', flush=True)
        video = avio.VideoDB(video_filepath, frame_rate=self.frame_rate, start_time=segm_start, max_dur=segm_end-segm_start)
        loader = DataLoader(video, batch_size=batch_size, num_workers=0)

        t = time.time()
        self.tracker.reset_tracker()
        for batch_id, (frames, frames_ts) in enumerate(loader):
            if frames_ts[0].item() >= segm_end:      # Reached end of segment.
                break
            if frames_ts[-1].item() < segm_start:    # Has not reached begin of the segment yet.
                continue

            idx = torch.logical_and(frames_ts >= segm_start, frames_ts < segm_end)
            frames, frames_ts = frames[idx], frames_ts[idx]

            # Run detectors on batch of images
            detections = self.detector.predictor(frames.numpy())
            detections = [det['instances'].to('cpu') for det in detections]

            # Run tracker frame by frame
            for dets, ts in zip(detections, frames_ts):
                # Run tracker
                self.tracker.update(ts.item(), dets.pred_boxes, dets.scores, dets.box_feats, dets.logit_classes)

            # Log
            if batch_id % 10 == 0:
                sec_vid = frames_ts[-1].item()
                progress_frac = sec_vid / video.reader._dur
                print(f"[{job_id}][{youtube_id}][{progress_frac*100:.1f}%] Parsing object tracks | "
                      f"InferenceSpeed={float(sec_vid - video.reader.start_time) / (time.time() - t): .2f} sec video/sec | "
                      f"NumTracks={objtrack_manager.tracks_saved}.", flush=True)

        # Scene change. Save tracks and reset tracker.
        min_area = video.reader.frame_size[0] * video.reader.frame_size[1] * self.min_track_area
        tracks = self.tracker.get_tracks(min_secs=self.min_track_len, min_area=min_area)
        objtrack_manager.save_tracks_meta(tracks, vid=youtube_id, video_size=video.reader.frame_size)
        print(f"[{job_id}][{youtube_id}] Finished parsing segment. Found {objtrack_manager.tracks_saved} tracks.", flush=True)

    def parse_video(self, youtube_id, job_id, segments=None):
        if self.progress_tracker.check_completed(youtube_id):
            return  # Skip. Already processed

        video_filepath = f"{self.videos_dir}/{youtube_id[:2]}/{youtube_id}.mp4"
        if not misc_utils.check_video(video_filepath):
            return

        # Search for the largest batch size that fits available gpu
        batch_size = get_max_batch_size(self.detector, avio.VideoDB(video_filepath).reader.frame_size)
        print(f"[{job_id}][{youtube_id}] Optimal Batch Size={batch_size}", flush=True)

        # Parse object tracks segment by segment
        video_progress_tracker = ProgressTracker(os.path.join(self.tracks_meta_dir, youtube_id[:2], f'{youtube_id}-progress.txt'))
        for segm in segments:
            segment_id = f"{youtube_id}-{segm[0]:.2f}-{segm[1]:.2f}"
            if video_progress_tracker.check_completed(segment_id):
                continue

            try:
                self.parse_object_tracks(video_filepath, segm, batch_size, job_id)
                video_progress_tracker.add(segment_id)
            except AssertionError:
                continue

        self.progress_tracker.add(youtube_id)
        print(f'[{job_id}][{youtube_id}] Object parsing done.', flush=True)

    def process_all(self):
        for job_id, youtube_id, segments in self.scheduled_jobs():
            self.parse_video(youtube_id, segments=segments, job_id=job_id)


class Launcher:
    def __call__(self, args):
        torch.multiprocessing.set_start_method('spawn')
        ObjectTracksParser(
            args.base_dir,
            args.yid_index_fn,
            args.dataset_domain,
            detic_cfg=DETIC_CFG(),
            bytetrack_cfg=BYTETRACK_CFG(),
            min_track_area=args.min_track_area,
            min_track_len=args.min_track_len,
            world_size=args.world_size,
            rank=args.rank,
        ).process_all()


if __name__ == '__main__':
    args = parse_arguments()

    if args.slurm:
        job_names = os.popen('squeue -o %j -u $USER').read().split("\n")
        slurm_job_name = f"parse-tracks-{args.rank}of{args.world_size}"
        if slurm_job_name in job_names:
            print(f'Skipping {slurm_job_name} because already in queue')
            exit(0)

        # Submit jobs
        executor = submitit.AutoExecutor(folder='./slurm_logs/', slurm_max_num_timeout=20, cluster=None)
        executor.update_parameters(
            timeout_min=1440,                # Requeue every 12hr
            slurm_partition=args.partition,
            cpus_per_task=10,
            gpus_per_node=1,                # Only one gpu per task. Trouble initializing multiple processes within the submitit launcher.
            nodes=1,
            tasks_per_node=1,
            mem_gb=32,
            slurm_additional_parameters={"exclude": "euler05"},
            slurm_signal_delay_s=20)
        executor.update_parameters(name=slurm_job_name)
        executor.submit(Launcher(), args)
        print(f"Job submitted: {slurm_job_name}")
    else:
        Launcher()(args)
