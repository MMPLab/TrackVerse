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
import avio

from scenedetect import detect, AdaptiveDetector
from bytetrack.byte_tracker import BYTETracker

from utils import misc as misc_utils
from utils import detic as detic_utils
from utils import youtube as yt_utils


def parse_arguments():
    parser = argparse.ArgumentParser(description='youtube video downloader')
    parser.add_argument("--slurm", default=False, action="store_true")
    parser.add_argument("--partition", default='research')
    parser.add_argument("--skip_processing", default=False, action="store_true")
    parser.add_argument("--skip_download", default=False, action="store_true")
    parser.add_argument("--skip_filters", default=False, action="store_true")
    parser.add_argument("--push_to_euler", default=False, action="store_true")
    parser.add_argument("--print_progress", default=False, action="store_true")
    parser.add_argument("--clean_videos", default=False, action="store_true")

    parser.add_argument('--base_dir', default='data/tracks_meta/in1k_coco_x1000',type=str, help='Working Directory')
    parser.add_argument('--dataset_name', type=str, help='Name of dataset.')
    parser.add_argument('--metafile', default=None, type=str, help='youtube video meta')
    parser.add_argument('--clips_metafile', default=None, type=str, help='youtube video meta')
    parser.add_argument("--num_chunks", default=1, type=int, help="scheduling chunks")
    parser.add_argument("--chunk_id", default=0, type=int, help="scheduling chunk id")

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


def ts2sec(ts):
    m = re.match('(\d+):(\d+):(\d+)\.(\d+)', str(ts))
    sec = int(m.group(1))*3600+int(m.group(2))*60+int(m.group(3))+float(m.group(4))/1000
    return sec


class ObjectTracksManager(object):
    def __init__(self, output_dir, class_desc):
        self.output_dir = output_dir
        self.class_desc = class_desc
        self.track_id = 0

    def construct_meta_dict(self, track, video_size):
        obj_scores = np.array([float(h['score'].item()) for h in track.history])
        cls_scores = np.stack([h['logit'].numpy() for h in track.history])
        wcls_scores = cls_scores * obj_scores[:, None]
        top10_idx = np.argsort(-wcls_scores.mean(0)[:-1])[:10]
        frame_top_lbl = [int(wcls[:-1].argmax()) for wcls in wcls_scores]
        meta_dict = {
            'track_id': self.track_id,
            'video_size': video_size,
            'track_ts': (track.history[0]['ts'], track.history[-1]['ts']),
            'top10_lbl': top10_idx.tolist(),
            'top10_desc': [self.class_desc[lbl] for lbl in top10_idx],
            'top10_cls': (cls_scores.mean(0)[top10_idx].tolist(), cls_scores.std(0)[top10_idx].tolist()),
            'top10_wcls': (wcls_scores.mean(0)[top10_idx].tolist(), wcls_scores.std(0)[top10_idx].tolist()),
            'frame_ts': [h['ts'] for h in track.history],
            'frame_bboxes': [h['bbox'].tolist() for h in track.history],
            'frame_obj': obj_scores.tolist(),
            'frame_cls': [float(cls_scores[t, frame_top_lbl[t]]) for t in range(len(track.history))],
            'frame_wcls': [float(wcls_scores[t, frame_top_lbl[t]]) for t in range(len(track.history))],
            'frame_lbl': frame_top_lbl,
            'frame_desc': [self.class_desc[lbl] for lbl in frame_top_lbl],
        }
        return meta_dict

    def save_tracks_meta(self, tracks, vid, video_size):
        meta_json = os.path.join(self.output_dir, vid[:2], f"{vid}-meta.jsonl.gzip")
        misc_utils.check_dirs(os.path.dirname(meta_json))

        saved_tracks = []
        with gzip.open(meta_json, 'at') as fp:
            for track in tracks:
                track_meta = self.construct_meta_dict(track, video_size)
                fp.write(json.dumps(track_meta) + '\n')
                saved_tracks.append(self.track_id)
                self.track_id += 1

        return saved_tracks

    def load_tracks_meta(self, vid):
        meta_json = os.path.join(self.output_dir, vid[:2], f"{vid}-meta.jsonl.gzip")
        with gzip.open(meta_json, 'rt') as fp:
            for line in fp:
                yield json.loads(line.strip())

    @staticmethod
    def extract_crop(frame, box, target_ar, target_w):
        x, y, w, h = box
        frame_h, frame_w = frame.shape[:2]

        # Extend one edge to match average aspect ratio
        if w/h > target_ar:
            y -= (w / target_ar - h) / 2
            h = w / target_ar
        else:
            x -= (h * target_ar - w) / 2
            w = h * target_ar

        # Crop and resize
        crop = frame[max(int(y), 0):int(y+h)+1, max(int(x), 0):int(x+w)+1]
        pad = ((int(max(-y, 0)), int(max(y+h-frame_h, 0))),
               (int(max(-x, 0)), int(max(x+w-frame_w, 0))),
               (0, 0))
        crop = np.pad(crop, pad, 'constant', constant_values=0)
        crop = np.array(PIL.Image.fromarray(crop).resize((int(target_w), int(target_w / target_ar))))
        return crop

    def extract_track(self, boxes, timestamps, reader):
        avg_ar = np.mean(boxes[:, 2]/boxes[:, 3])
        avg_w = int(np.mean(boxes[:, 2]))

        for frame, ts in reader.read():
            idx = np.where((timestamps - ts) > 0)[0][0]
            if idx == 0:
                box = boxes[0]
            else:
                prev_bbox, next_bbox = boxes[idx-1], boxes[idx]
                prev_ts, next_ts = timestamps[idx-1], timestamps[idx]
                alpha = (ts - prev_ts) / (next_ts - prev_ts)
                box = (prev_bbox * alpha + next_bbox * (1 - alpha)).tolist()

            yield self.extract_crop(np.array(frame), box, avg_ar, avg_w)

    def display_tracks(self, vid, video_filepath, track_ids):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib import cm
        import numpy as np

        # Load bboxes for the video
        all_tracks = {track_meta['track_id']: track_meta for track_meta in self.load_tracks_meta(vid) if track_meta['track_id'] in track_ids}
        min_ts = min([v['frame_ts'][0] for k, v in all_tracks.items()])
        max_ts = max([v['frame_ts'][-1] for k, v in all_tracks.items()])

        # Display video with tracks
        reader = avio.VideoReader(video_filepath, start_time=min_ts, duration=max_ts-min_ts, rate=4)
        colors = [cm.jet(int(i))[:3] for i in np.linspace(0, 256, 8)]
        rows, cols = min(int(math.ceil(len(reader) / 8)), 10), 8
        fig, ax = plt.subplots(rows, cols, figsize=(cols*6, rows*4), sharex=True, sharey=True)
        [ax2.axis('off') for ax1 in ax for ax2 in ax1]
        for fid, (frame, ts) in enumerate(reader.read()):
            if fid >= 80:
                break
            i, j = fid//cols, fid%cols
            ax[i, j].imshow(frame)

            for track_id in all_tracks:
                timestamps = np.array(all_tracks[track_id]['frame_ts'])
                boxes = np.array(all_tracks[track_id]['frame_bboxes'])
                classes = all_tracks[track_id]['frame_desc']
                obj_scores = np.array(all_tracks[track_id]['frame_obj'])
                cls_scores = np.array(all_tracks[track_id]['frame_cls'])

                # Check if track is active
                if float(ts) < timestamps[0] or float(ts) > timestamps[-1]:
                    continue

                # Find timestamp in track and interpolate box
                idx = np.where((timestamps - ts) >= 0)[0][0]
                if idx == 0:
                    box = boxes[0]
                else:
                    prev_bbox, next_bbox = boxes[idx-1], boxes[idx]
                    prev_ts, next_ts = timestamps[idx-1], timestamps[idx]
                    alpha = (ts - prev_ts) / (next_ts - prev_ts)
                    box = (prev_bbox * alpha + next_bbox * (1 - alpha)).tolist()

                # Display the box
                ax[i, j].add_patch(patches.Rectangle(
                    (box[0], box[1]), box[2], box[3],
                    linewidth=1, edgecolor=colors[track_id % len(colors)], facecolor='none'))

                desc = f"{track_id}: {classes[idx]} ({obj_scores[idx]:.2f};{cls_scores[idx]:.2f})"
                ax[i, j].text(box[0], box[1], desc, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 2})
        fig.show()


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
        video = avio.VideoDB(video_filepath, start_time=start_time, max_dur=min(end_time-start_time, 10), frame_rate=3, transform=self.transform)
        loader = DataLoader(video, batch_size=32, num_workers=0)
        outputs = torch.cat([self.model(frames.to(self.device)).cpu() for frames, ts in loader]).mean(0)
        return outputs[0] > 0.


class AestheticPredictor:
    def __init__(self, device='cpu'):
        import open_clip
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
        video = avio.VideoDB(video_filepath, start_time=start_time, max_dur=min(end_time-start_time, 10), frame_rate=3, transform=self.transform)
        loader = DataLoader(video, batch_size=32, num_workers=0)
        ascores = []
        for frames, ts in loader:
            x = self.model.encode_image(frames.to(self.device))
            x /= x.norm(dim=-1, keepdim=True)
            ascores.append(self.predictor(x))
        return torch.cat(ascores).mean()


class ObjectTracksParser(object):
    def __init__(self, metafile, clips_metafile, base_dir, args, num_gpus=1, dataset_name='panda70m_lvis'):
        self.metafile = metafile
        self.clips_metafile = clips_metafile
        self.args = args
        self.num_chunks = args.num_chunks
        self.chunk_id = args.chunk_id

        self.num_gpus = num_gpus
        self.viz_object_tracks = False

        # Output directories
        self.downl_dir = os.path.join(base_dir, 'videos_mp4')
        self.downloader = yt_utils.YoutubeDL(self.downl_dir, args.push_to_euler)
        self.removed_tracker = misc_utils.ProgressTracker(os.path.join(self.downl_dir, 'removed.txt'))

        self.tracks_meta_dir = os.path.join(base_dir, 'tracks_meta', dataset_name)
        misc_utils.check_dirs(self.tracks_meta_dir)
        self.procs_tracker = misc_utils.ProgressTracker(os.path.join(self.tracks_meta_dir, 'processed.txt'))
        self.clean_tracker = misc_utils.ProgressTracker(os.path.join(self.tracks_meta_dir, 'cleaned.txt'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not args.skip_processing and not self.args.skip_filters:
            self.cartoon_detector = CartoonDetector(self.device)
            self.aesthetic_predictor = AestheticPredictor(self.device)

        # Setup vocabulary for detic classification
        if args.vocab == 'in1k_coco':
            self.class_desc = [l.strip() for l in open('assets/class_prompts/coco-prompts.txt')] + \
                              [l.strip() for l in open('assets/class_prompts/imagenet1k-prompts.txt')]
            self.class_desc = sorted(list(set(self.class_desc)))
        else:
            self.class_desc = [l.strip() for l in open('assets/class_prompts/lvis-prompts.txt')]

    def read_video_urls(self, metafile):
        jobs = defaultdict(list)
        for meta_fn in sorted(glob.glob(metafile)):
            print(f'Reading video urls from: {meta_fn}', flush=True)
            for line in tqdm.tqdm(open(meta_fn)):
                m = json.loads(line)
                video_id = m['video_id']
                video_fn = f"{self.downl_dir}/{video_id[:2]}/{video_id}.mp4"
                if self.args.skip_download:
                    if not os.path.exists(video_fn):
                        continue
                jobs[video_id].extend([(ts2sec(c['span'][0]), ts2sec(c['span'][1])) for c in m['clip']])
        jobs = [{'youtube_id': yid, 'segments': jobs[yid]} for yid in sorted(jobs.keys())]
        if self.num_chunks > 1:
            jobs = jobs[self.chunk_id::self.num_chunks]
        return jobs

    def read_clip_urls(self, metafile):
        jobs = {}
        for line in open(metafile):
            m = json.loads(line)
            yid = m['clip_id'][:11]
            if yid not in jobs:
                jobs[yid] = {
                    'youtube_id': yid,
                    'segments': []
                }
            start_ts, end_ts = m['duration']
            jobs[yid]['segments'].append((ts2sec(start_ts), ts2sec(end_ts)))

        return [jobs[k] for k in sorted(list(jobs.keys()))]

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
    def detect_cartoon(self, vid, video_fn, tracks, job_id):
        start_time = tracks[0].ts[0]
        url = f"https://www.youtube.com/watch?v={vid}&t={int(start_time)}"
        if self.cartoon_detector.is_cartoon(video_fn,  start_time, start_time+30):
            print(f'[{job_id}][{vid}]: Cartoon. {url}', flush=True)
            return True
        else:
            print(f'[{job_id}][{vid}]: Not Cartoon. {url}', flush=True)
            return False

    @torch.no_grad()
    def detect_low_aesthetics(self, vid, video_fn, tracks, job_id):
        start_time = tracks[0].ts[0]
        url = f"https://www.youtube.com/watch?v={vid}&t={int(start_time)}"
        ascore = self.aesthetic_predictor.get_aesthetic_score(video_fn, start_time, start_time + 30)
        print(f'[{job_id}][{vid}]: Aesthetic Score {ascore}. {url}', flush=True)
        return ascore < 4.0

    @torch.no_grad()
    def parse_object_tracks(self, vid, video_filepath, detector, tracker, batch_size, job_id, segment):
        segments_complete = []
        objtrack_manager = ObjectTracksManager(self.tracks_meta_dir, class_desc=self.class_desc)
        progress_fn = os.path.join(self.tracks_meta_dir, vid[:2], f'{vid}-progress.json')
        misc_utils.check_dirs(os.path.dirname(progress_fn))
        if os.path.exists(progress_fn):
            try:
                progress = json.load(open(progress_fn))
                segments_complete = progress['segments_complete']
                objtrack_manager.track_id = progress['num_tracks']
                if segment[0] in set([seg[0] for seg in segments_complete]):
                    return
            except Exception:
                pass

        segm_start, segm_end = segment
        print(f'[{job_id}][{vid}] Parsing segment {segment}).', flush=True)

        # Start parsing video segment
        video = avio.VideoDB(video_filepath, frame_rate=self.args.frame_rate, start_time=segm_start)
        loader = DataLoader(video, batch_size=batch_size, num_workers=0)

        t = time.time()
        tracker.reset_tracker()
        for batch_id, (batch_frames, batch_timestamps) in enumerate(loader):
            if batch_timestamps[0].item() >= segm_end:  # Reached end of segment.
                break
            if batch_timestamps[-1].item() < segm_start: # Has not reached begin of the segment yet.
                continue

            # Run detectors on batch of images
            detections = detector.predictor(batch_frames.numpy())
            detections = [det['instances'].to('cpu') for det in detections]

            # Run tracker frame by frame
            for dets, ts in zip(detections, batch_timestamps):
                if ts >= segm_end:
                    # Scene change. Save tracks and reset tracker.
                    min_area = dets.image_size[0]*dets.image_size[1] * self.args.min_track_area
                    tracks = tracker.get_tracks(min_secs=self.args.min_track_len, min_area=min_area)
                    track_ids = objtrack_manager.save_tracks_meta(tracks, vid=vid, video_size=dets.image_size)
                    if self.viz_object_tracks and len(track_ids) > 0:
                        objtrack_manager.display_tracks(vid=vid, video_filepath=video_filepath, track_ids=track_ids)
                    tracker.reset_tracker()
                    break

                # Run tracker
                tracker.update(ts.item(), dets.pred_boxes, dets.scores, dets.box_feats, dets.logit_classes)

            # Log
            if batch_id % 10 == 0:
                sec_vid = batch_timestamps[-1].item()
                progress_frac = sec_vid / video.reader._dur
                print(f"[{job_id}][{vid}][{progress_frac*100:.1f}%] Parsing object tracks | "
                      f"InferenceSpeed={float(sec_vid - video.reader.start_time) / (time.time() - t): .2f} sec video/sec | "
                      f"NumTracks={objtrack_manager.track_id}.", flush=True)

        # Make sure we don't parse the same segment next time around.
        segments_complete.append(segment)
        json.dump({'segments_complete': segments_complete, 'num_tracks': objtrack_manager.track_id},
                  open(progress_fn, 'w'))
        print(f"[{job_id}][{vid}] Finished parsing segment. Found {objtrack_manager.track_id} tracks.", flush=True)

    def parse_scenes(self, video_filepath, segments=None):
        scenes_ts = detect(video_filepath, AdaptiveDetector())
        scenes_ts = [(ts2sec(ts[0]), ts2sec(ts[1])) for ts in scenes_ts]
        if len(scenes_ts) == 0:
            video = avio.VideoDB(video_filepath, frame_rate=self.args.frame_rate)
            scenes_ts = [(float(video.reader.start_time), float(video.reader.start_time + video.reader.duration))]

        # Filter scenes within pre-specified segments
        if segments is not None:
            for scene_id, (scene_start, scene_end) in enumerate(scenes_ts):
                scene_in_segm = False
                for segm_start, segm_end in segments:
                    if scene_end > segm_start and scene_start < segm_end:
                        scene_in_segm = True
                        break
                if not scene_in_segm:
                    scenes_ts[scene_id] = None
                else:
                    # Fit scene to scene.
                    scenes_ts[scene_id] = (max(scenes_ts[scene_id][0], segm_start), min(scenes_ts[scene_id][1], segm_end))
            scenes_ts = [ts for ts in scenes_ts if ts is not None]

        # Filter out scenes with smaller than 3sec
        scenes_ts = [(start_ts, end_ts) for start_ts, end_ts in scenes_ts
                     if end_ts - start_ts > 3]
        # Cut scenes larger than 10 min
        scenes_ts = [(start_ts, end_ts) if end_ts - start_ts < 600 else (start_ts, start_ts+600)
                     for start_ts, end_ts in scenes_ts]
        return scenes_ts

    def process_video(self, vid, detector, tracker, job_id, segments=None):
        if self.procs_tracker.check_completed(vid):
            return  # Skip. Already processed

        # Download
        if self.args.skip_download:
            video_filepath = f"{self.downl_dir}/{vid[:2]}/{vid}.mp4"
            if not misc_utils.check_video(video_filepath):
                return
        else:
            dl_status, video_filepath = self.downloader.download_video(vid)
            if dl_status == yt_utils.STATUS.FAIL:
                print(f'[{job_id}][{vid}] Download failed.', flush=True)
                return
            if dl_status == yt_utils.STATUS.DONE:
                print(f'[{job_id}][{vid}] Already downloaded. Skipping', flush=True)
            else:
                print(f'[{job_id}][{vid}] Download successful.', flush=True)

        # Detect scene changes
        scenes_ts = self.parse_scenes(video_filepath, segments)
        if not scenes_ts:
            print(f'[{job_id}][{vid}] No scenes to parse.', flush=True)
            self.procs_tracker.add(vid)
            return
        print(f"[{job_id}][{vid}] PySceneDetect: Found {len(scenes_ts)} scenes.", flush=True)

        # Filter out cartoons and low aesthetics
        if not self.args.skip_filters:
            start_time = scenes_ts[0][0]
            if self.cartoon_detector.is_cartoon(video_filepath,  start_time, start_time+30):
                print(f'[{job_id}][{vid}]: Cartoon.', flush=True)
                self.procs_tracker.add(vid)
                self.removed_tracker.add(vid)
                os.remove(video_filepath)
                return
            if self.aesthetic_predictor.get_aesthetic_score(video_filepath, start_time, start_time+30) < 4.0:
                print(f'[{job_id}][{vid}]: Low Aesthetics.', flush=True)
                self.procs_tracker.add(vid)
                self.removed_tracker.add(vid)
                os.remove(video_filepath)
                return

        if not self.args.skip_processing:
            # Search for the largest batch size that fits available gpu
            batch_size = self.get_max_batch_size(detector, avio.VideoDB(video_filepath).reader.frame_size)
            print(f"[{job_id}][{vid}] Optimal Batch Size={batch_size}", flush=True)

            # Parse object tracks segment by segment
            for segm in scenes_ts:
                try:
                    self.parse_object_tracks(vid, video_filepath, detector, tracker, batch_size, job_id, segment=segm)
                except AssertionError:
                    continue
            self.procs_tracker.add(vid)
            print(f'[{job_id}][{vid}] Object parsing done.', flush=True)

    def process_all(self):
        detector = tracker = None
        if not self.args.skip_processing:
            detector = detic_utils.build_detic(self.class_desc, args=self.args, gpu_id=0)
            tracker = BYTETracker(self.args)
        if self.clips_metafile is None:
            all_jobs = self.read_video_urls(self.metafile)
        else:
            all_jobs = self.read_clip_urls(self.clips_metafile)
        for job_id, job_meta in enumerate(all_jobs):
            self.process_video(job_meta['youtube_id'], detector, tracker, job_id=f"{job_id}/{len(all_jobs)}", segments=job_meta['segments'])

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

    def clean_videos(self):
        # all_videos = glob.glob(f"{self.downl_dir}/*/*.mp4")
        # all_videos_vids = sorted(list(set([fn.split('/')[-1][:11] for fn in all_videos])))
        # all_tracks = glob.glob(f"{self.tracks_meta_dir}/*/*.jsonl.gzip")
        # all_tracks_vids = sorted(list(set([fn.split('/')[-1][:11] for fn in all_tracks])))
        # to_rm = sorted(list(set(all_videos_vids).difference(set(all_tracks_vids))))
        # for vid in to_rm:
        #     self.removed_tracker.add(vid)
        #     os.remove(f"{self.downl_dir}/{vid[:2]}/{vid}.mp4")

        all_jobs = self.read_clip_urls(self.clips_metafile)
        for job_id, job_meta in enumerate(all_jobs):
            video_id = job_meta['youtube_id']
            print(f'[{job_id}][{video_id}] Start', flush=True)

            video_fn = f"{self.downl_dir}/{video_id[:2]}/{video_id}.mp4"
            progress_fn = f"{self.tracks_meta_dir}/{video_id[:2]}/{video_id}-progress.json"
            meta_json = f"{self.tracks_meta_dir}/{video_id[:2]}/{video_id}-meta.jsonl.gzip"
            if self.clean_tracker.check_completed(video_id):
                continue
            if not os.path.exists(video_fn):
                self.clean_tracker.add(video_id)
                continue
            if not os.path.exists(progress_fn) and not os.path.exists(meta_json):
                self.clean_tracker.add(video_id)
                continue

            # Detect scene changes
            scenes_ts = self.parse_scenes(video_fn, job_meta['segments'])
            if not scenes_ts:
                self.clean_tracker.add(video_id)
                continue

            # Filter out cartoons and low aesthetics
            start_time = scenes_ts[0][0]
            to_rm = False
            if self.cartoon_detector.is_cartoon(video_fn, start_time, start_time + 30):
                print(f'[{job_id}][{video_id}]: Cartoon.', flush=True)
                to_rm = True
                # continue

            elif self.aesthetic_predictor.get_aesthetic_score(video_fn, start_time, start_time + 30) < 4.0:
                print(f'[{job_id}][{video_id}]: Low Aesthetics.', flush=True)
                to_rm = True
                # continue

            if to_rm:
                if os.path.exists(progress_fn):
                    os.remove(progress_fn)
                if os.path.exists(meta_json):
                    os.remove(meta_json)
            self.clean_tracker.add(video_id)


class Launcher:
    def __call__(self, args):
        print(args.clips_metafile)
        torch.multiprocessing.set_start_method('spawn')
        yvd = ObjectTracksParser(args.metafile,
                                 args.clips_metafile,
                                 args.base_dir,
                                 args,
                                 num_gpus=torch.cuda.device_count(),
                                 dataset_name=args.dataset_name)
        if args.print_progress:
            yvd.print_progress()
        elif args.clean_videos:
            yvd.clean_videos()
        else:
            yvd.process_all()


if __name__ == '__main__':
    args = parse_arguments()

    if args.slurm:
        job_names = os.popen('squeue -o %j -u $USER').read().split("\n")
        if args.clips_metafile is None:
            slurm_job_name = f"parse-hdvila-{args.metafile.split('/')[-1]}-{args.chunk_id}of{args.num_chunks}"
        else:
            slurm_job_name = f"parse-hdvila-clips-{args.clips_metafile.split('/')[-1]}-{args.chunk_id}of{args.num_chunks}"
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
