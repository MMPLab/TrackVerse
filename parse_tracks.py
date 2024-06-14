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


def parse_arguments():
    parser = argparse.ArgumentParser(description='youtube video downloader')
    parser.add_argument("--slurm", default=False, action="store_true")
    parser.add_argument("--partition", default='research')
    parser.add_argument("--world_size", default=1, type=int, help="scheduling chunks")
    parser.add_argument("--rank", default=0, type=int, help="scheduling chunk id")

    parser.add_argument('--base_dir', default='/home/pmorgado/datasets/TrackVerse/',
                        help='Dataset directory')
    parser.add_argument('--yid_index_fn', default="assets/trackverse-yids-all.txt",
                        help='index of youtube ids to download.')
    parser.add_argument('--dataset_name', default="TrackVerseLVIS", help='Name of dataset.')
    parser.add_argument('--class_prompts', default="assets/lvis-prompts.txt",
                        help='TSV file of class names and prompts.')

    # video loading args
    parser.add_argument("--frame-rate", default=16, type=int, help="test conf")
    parser.add_argument("--frame-size", default=480, type=int, help="test conf")
    # detection args
    parser.add_argument('--vocab', default='in1k_coco', type=str, choices=["in1k_coco", "lvis"], help='Vocabulary to search for.')
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.55, help="tracking confidence threshold")
    parser.add_argument("--match_thresh", type=float, default=0.45, help="matching threshold for tracking")
    parser.add_argument("--motion_weight", type=float, default=0.4, help="how much to weight motion information versus appearance information")
    parser.add_argument("--track_iou_low_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=24, help="the frames for keep lost tracks")
    parser.add_argument("--min-track-area", type=float, default=0.1, help='filter out tiny boxes')
    parser.add_argument("--min-track-len", type=float, default=3., help='filter out small tracks')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    return parser.parse_args()


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
    def __init__(self, args):
        self.base_dir = args.base_dir
        self.index_fn = args.yid_index_fn
        self.dataset_name = args.dataset_name
        self.world_size = args.world_size
        self.rank = args.rank
        self.args = args

        # Output directories
        self.videos_dir = os.path.join(self.base_dir, 'videos_mp4')
        self.segments_dir = os.path.join(self.base_dir, 'videos_segm')
        self.tracks_meta_dir = os.path.join(self.base_dir, 'tracks_meta', self.dataset_name)
        misc_utils.check_dirs(self.tracks_meta_dir)
        self.progress_tracker = misc_utils.ProgressTracker(os.path.join(self.tracks_meta_dir, 'completed.txt'))

        # Setup vocabulary for detic classification
        self.class_prompts = [l.strip() for l in open(args.class_prompts)]

    def scheduled_jobs(self):
        for job_id, ln in enumerate(open(self.index_fn)):
            youtube_id = ln.strip()
            segm_filepath = os.path.join(self.segments_dir, youtube_id[:2], f"{youtube_id}.txt")
            if len(youtube_id) != 11:
                continue
            if job_id % self.world_size == self.rank:
                segments = [ln.strip().split(',') for ln in open(segm_filepath, "r")]
                segments = [(float(start), float(end)) for start, end in segments]
                yield job_id, youtube_id, segments

    @torch.no_grad()
    def get_max_batch_size(self, detector, frame_size):
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

    @torch.no_grad()
    def parse_object_tracks(self, video_filepath, segment, detector, tracker, batch_size, job_id):
        segm_start, segm_end = segment
        youtube_id = video_filepath.split('/')[-1][:11]
        objtrack_manager = ObjectTracksManager(self.tracks_meta_dir, class_desc=self.class_prompts)

        print(f'[{job_id}][{youtube_id}] Start parsing segment {segment}.', flush=True)
        video = avio.VideoDB(video_filepath, frame_rate=self.args.frame_rate, start_time=segm_start, max_dur=segm_end-segm_start)
        loader = DataLoader(video, batch_size=batch_size, num_workers=0)

        t = time.time()
        tracker.reset_tracker()
        for batch_id, (frames, frames_ts) in enumerate(loader):
            if frames_ts[0].item() >= segm_end:      # Reached end of segment.
                break
            if frames_ts[-1].item() < segm_start:    # Has not reached begin of the segment yet.
                continue

            idx = torch.logical_and(frames_ts >= segm_start, frames_ts < segm_end)
            frames, frames_ts = frames[idx], frames_ts[idx]

            # Run detectors on batch of images
            detections = detector.predictor(frames.numpy())
            detections = [det['instances'].to('cpu') for det in detections]

            # Run tracker frame by frame
            for dets, ts in zip(detections, frames_ts):
                # Run tracker
                tracker.update(ts.item(), dets.pred_boxes, dets.scores, dets.box_feats, dets.logit_classes)

            # Log
            if batch_id % 10 == 0:
                sec_vid = frames_ts[-1].item()
                progress_frac = sec_vid / video.reader._dur
                print(f"[{job_id}][{youtube_id}][{progress_frac*100:.1f}%] Parsing object tracks | "
                      f"InferenceSpeed={float(sec_vid - video.reader.start_time) / (time.time() - t): .2f} sec video/sec | "
                      f"NumTracks={objtrack_manager.tracks_saved}.", flush=True)

        # Scene change. Save tracks and reset tracker.
        min_area = video.reader.frame_size[0] * video.reader.frame_size[1] * self.args.min_track_area
        tracks = tracker.get_tracks(min_secs=self.args.min_track_len, min_area=min_area)
        objtrack_manager.save_tracks_meta(tracks, vid=youtube_id, video_size=video.reader.frame_size)
        print(f"[{job_id}][{youtube_id}] Finished parsing segment. Found {objtrack_manager.tracks_saved} tracks.", flush=True)

    def process_video(self, youtube_id, detector, tracker, job_id, segments=None):
        if self.progress_tracker.check_completed(youtube_id):
            return  # Skip. Already processed

        video_filepath = f"{self.videos_dir}/{youtube_id[:2]}/{youtube_id}.mp4"
        if not misc_utils.check_video(video_filepath):
            return

        # Search for the largest batch size that fits available gpu
        batch_size = self.get_max_batch_size(detector, avio.VideoDB(video_filepath).reader.frame_size)
        print(f"[{job_id}][{youtube_id}] Optimal Batch Size={batch_size}", flush=True)

        # Parse object tracks segment by segment
        video_progress_tracker = ProgressTracker(os.path.join(self.tracks_meta_dir, youtube_id[:2], f'{youtube_id}-progress.txt'))
        for segm in segments:
            segment_id = f"{youtube_id}-{segm[0]:.2f}-{segm[1]:.2f}"
            if video_progress_tracker.check_completed(segment_id):
                continue

            try:
                self.parse_object_tracks(video_filepath, segm, detector, tracker, batch_size, job_id)
                video_progress_tracker.add(segment_id)
            except AssertionError:
                continue

        self.progress_tracker.add(youtube_id)
        print(f'[{job_id}][{youtube_id}] Object parsing done.', flush=True)

    def process_all(self):
        detector = detic_utils.build_detic(
            self.class_prompts,
            self.args.frame_size,
            self.args.nms,
            self.args.conf,
            gpu_id=0
        )
        tracker = BYTETracker(
            self.args.track_thresh,
            self.args.track_iou_low_thresh,
            self.args.match_thresh,
            self.args.frame_rate,
            self.args.track_buffer,
            self.args.motion_weight,
            self.args.mot20
        )

        for job_id, youtube_id, segments in self.scheduled_jobs():
            self.process_video(youtube_id, detector, tracker, segments=segments, job_id=job_id)

    @staticmethod
    def load_track(track_db):
        parse_label = lambda t: t.replace('photo of a ', '').split(' (also')[0]
        # Takes about 5 min to load ~600k tracks
        track_class = []
        for i, line in enumerate(gzip.open(track_db)):
            data = json.loads(line)
            track_class.append(parse_label(data['top10_desc'][0]))
        return track_class

    def print_progress(self):
        dataset = Counter()
        files = glob.glob(f'{self.tracks_meta_dir}/*/*gzip')
        total_videos = len(glob.glob(f'{self.downl_dir}/*/*mp4'))
        pool = mp.Pool(processes=4)
        results = pool.imap_unordered(self.load_track, files)
        for it, result in enumerate(tqdm.tqdm(results, total=len(files))):
            dataset.update(result)
            if it % 100 == 0 and it != 0:
                total_tracks = sum([dataset[cls] for cls in dataset])
                tracks_per_class = list(map(lambda x: x[1], dataset.most_common())) + [0]
                hindex = [i for i, t, in enumerate(tracks_per_class) if t < i][0]
                i500 = len([t for t in tracks_per_class if t > 500])
                proj_tracks_per_class = [n / (it+1) * total_videos for n in tracks_per_class] + [0]
                proj_hindex = [i for i, t, in enumerate(proj_tracks_per_class) if t < i][0]
                proj_i500 = len([t for t in proj_tracks_per_class if t > 500])
                print(f'\nNClasses={len(dataset)}\t{total_tracks}=tracks\tHIndex={hindex} (Est={proj_hindex})\ti-500={i500} (Est={proj_i500})')


class Launcher:
    def __call__(self, args):
        torch.multiprocessing.set_start_method('spawn')
        ObjectTracksParser(args).process_all()


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
