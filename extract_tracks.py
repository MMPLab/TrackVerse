# Adapted from https://github.com/microsoft/XPretrain
import os
import gzip
import argparse
import json

import PIL.Image
import numpy as np
import scipy.ndimage as ndi
import tqdm
import submitit

from avio import VideoReader, VideoWriter
from utils import misc as misc_utils


def parse_arguments():
    parser = argparse.ArgumentParser(description='youtube video downloader')
    parser.add_argument("--slurm", default=False, action="store_true")
    parser.add_argument("--partition", default='research')
    parser.add_argument('--base_dir', default='.', type=str, help='Working Directory')
    parser.add_argument('--dataset_name', default='lvis', type=str, help='Working Directory')
    parser.add_argument('--db_meta_file', default=None, type=str,
                        help='GZIP file containing tracks in a dataset.')
    parser.add_argument('--num_chunks', default=1, type=int, help='Number of chunks')
    parser.add_argument('--chunk_id', default=0, type=int, help='Number of chunks')
    return parser.parse_args()


class Track:
    def __init__(self, yid, ts, boxes, meta):
        self.yid = yid
        self.ts = ts
        self.boxes = boxes
        parse_label = lambda t: t.replace('photo of a ', '').split(' (also')[0]
        label = parse_label(meta['top10_desc'][0]).replace(' ', '-')
        if 'mp4_filename' not in meta:
            meta['mp4_filename'] = f"{yid[:2]}/{yid}-{label}-{self.get_hash()}.mp4"
        if 'yid' not in meta:
            meta['yid'] = yid
        self.meta = meta

        self.mp4_filename = meta['mp4_filename']
        # obj_scores = np.array(self.meta['frame_obj'])
        # self.meta['obj_scores'] = [obj_scores.mean(), obj_scores.std()]
        # Delete some stuff to avoid OOM
        for k in ['frame_obj', 'frame_cls', 'frame_wcls', 'frame_lbl', 'frame_desc']:
            if k in self.meta:
                del self.meta[k]

    def get_hash(self):
        import hashlib  # unlike hash(), hashlib uses a constant seed for creating hash codes.
        ts = tuple([t for t in self.ts])
        boxes = tuple([tuple(b) for b in self.boxes.tolist()])
        yid = self.yid
        return hashlib.md5(f"{yid}-{str(ts)}-{str(boxes)}".encode('utf-8')).hexdigest()

    def overlap(self, track2):
        if self.yid != track2.yid or self.ts[0]>track2.ts[-1] or self.ts[-1]<track2.ts[0]:
            return 0

        interAreaSum, unionAreaSum = 0, 0
        for ts, box in zip(self.ts, self.boxes):
            if ts < track2.ts[0] or ts > track2.ts[-1]:
                unionAreaSum += box[2]*box[3]
                continue
            box2 = track2.boxes[np.abs(ts - track2.ts).argmin()]

            xA = max(box[0], box2[0])
            yA = max(box[1], box2[1])
            xB = min(box[0]+box[2], box2[0]+box2[2])
            yB = min(box[1]+box[3], box2[1]+box2[3])

            interArea = max(0, xB - xA) * max(0, yB - yA)
            unionArea = box[2]*box[3] + box2[2]*box2[3] - interArea
            interAreaSum += interArea
            unionAreaSum += unionArea
        return interAreaSum / unionAreaSum


def load_tracks(fn, yid):
    tracks = []
    for line in gzip.open(fn, mode='rt'):
        m = json.loads(line.strip())
        tracks.append(Track(
            yid,
            ts=np.array(m['frame_ts']).astype(float),
            boxes=np.array(m['frame_bboxes']).astype(float),
            meta=m,
        ))

    # Remove repeated tracks (ie, tracks with too much overlap)
    to_rm = []
    for i, t in enumerate(tracks):
        if i in to_rm:
            continue
        for j, t2 in enumerate(tracks[i+1:]):
            if t.overlap(t2) > 0.9:
                to_rm.append(j+i+1)
    tracks = [t for i, t in enumerate(tracks) if i not in to_rm]
    return tracks


def extract_crop_bilinear(frame, box, target_shape):
    frame = np.array(frame)
    frame_h, frame_w = frame.shape[:2]

    # Create coordinate grid
    x, y, w, h = box
    x_grid = np.linspace(x, x+w, target_shape[0])
    y_grid = np.linspace(y, y+h, target_shape[1])

    # Make sure grid coords are in the frame
    pad = [((y_grid<0).sum(), (y_grid>frame_h-1).sum()),
           ((x_grid<0).sum(), (x_grid>frame_w-1).sum()),
           (0, 0)]
    x_grid = x_grid[x_grid>=0]
    x_grid = x_grid[x_grid<=frame_w-1]
    y_grid = y_grid[y_grid>=0]
    y_grid = y_grid[y_grid<=frame_h-1]

    # Interpolate
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    x0 = np.floor(x_grid).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y_grid).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, frame_w-1)
    x1 = np.clip(x1, 0, frame_w-1)
    y0 = np.clip(y0, 0, frame_h-1)
    y1 = np.clip(y1, 0, frame_h-1)

    Ia = frame[ y0, x0 ]
    Ib = frame[ y1, x0 ]
    Ic = frame[ y0, x1 ]
    Id = frame[ y1, x1 ]

    wa = ((x1-x_grid) * (y1-y_grid))[:, :, None]
    wb = ((x1-x_grid) * (y_grid-y0))[:, :, None]
    wc = ((x_grid-x0) * (y1-y_grid))[:, :, None]
    wd = ((x_grid-x0) * (y_grid-y0))[:, :, None]

    im_interp = wa*Ia + wb*Ib + wc*Ic + wd*Id

    # Pad with zeros regions in the grid outside the frame
    return np.pad(im_interp, pad, 'constant', constant_values=0)


def extract_crop_nearest(frame, box, target_shape):
    frame = np.array(frame)
    # frame_h, frame_w = frame.shape[:2]

    # Create coordinate grid
    x, y, w, h = [int(b) for b in box]
    # assert x>=0
    # assert y>=0
    # assert x+w<=frame_w
    # assert y+h<=frame_h, f"{y} {h} {frame_h}"
    frame = frame[y:y+h, x:x+w]

    frame = np.array(PIL.Image.fromarray(frame).resize(target_shape))
    return frame


def extract_track(boxes, timestamps, reader, box_expansion=0.):
    timestamps = np.array(timestamps)
    boxes = np.array(boxes)

    # smooth boxes
    ts_diff = np.diff(timestamps).mean()
    std = int(0.66/ts_diff)
    boxes_smooth = np.stack([ndi.gaussian_filter(b, std) for b in boxes.T], 1)

    # normalize aspect ratio across track
    x, y, w, h = boxes_smooth.copy().T
    cx, cy, ar = x + w/2, y + h/2, w/h
    avg_ar = ar.mean()

    if avg_ar < 2/3:   trg_ar = 2/3
    elif avg_ar > 3/2: trg_ar = 3/2
    else:              trg_ar = avg_ar

    # Adjust xywh
    h = h * (1 + box_expansion)
    w = w * (1 + box_expansion)
    h[ar > trg_ar] = w[ar > trg_ar] / trg_ar
    w[ar < trg_ar] = h[ar < trg_ar] * trg_ar

    # Initial adjustments might make boxes too large.
    too_tall = h > reader.frame_size[0]
    too_wide = w > reader.frame_size[1]
    h[too_tall] = reader.frame_size[0]
    w[too_tall] = reader.frame_size[0] * trg_ar
    w[too_wide] = reader.frame_size[1]
    h[too_wide] = reader.frame_size[1] / trg_ar

    # Ensure boxes don't go outside the frame
    y = np.clip(cy-h/2, a_min=0, a_max=reader.frame_size[0]-h)
    x = np.clip(cx-w/2, a_min=0, a_max=reader.frame_size[1]-w)
    boxes_norm = np.stack([x, y, w, h], axis=1)

    # Loop video, find nearest boxes and interpolate
    out_shape = (int(w.mean())//2*2, int(h.mean())//2*2)
    for frame, ts in reader.read():
        idx = np.where(timestamps > ts)[0]
        if len(idx) == 0:
            box = boxes_norm[-1]
        elif idx[0] == 0:
            box = boxes_norm[0]
        else:
            idx = idx[0]
            prev_bbox, next_bbox = boxes_norm[idx-1], boxes_norm[idx]
            prev_ts, next_ts = timestamps[idx-1], timestamps[idx]
            alpha = (float(ts) - prev_ts) / (next_ts - prev_ts)
            box = (prev_bbox * (1-alpha) + next_bbox * alpha).tolist()

        # crop = extract_crop_bilinear(frame, box, out_shape)
        crop = extract_crop_nearest(frame, box, out_shape)
        yield crop


class ObjectTrackExtractor:
    def __init__(self, base_dir, db_meta_file=None, dataset_name='lvis', num_chunks=1, chunk_id=1):
        self.db_meta_file = None
        if db_meta_file is not None:
            assert db_meta_file is not None and os.path.exists(base_dir)
            self.db_meta_file = db_meta_file

        self.num_chunks = num_chunks
        self.chunk_id = chunk_id
        self.box_exp = [0., 0.25, 0.5, 1.]
        self.max_track_len = 30

        self.base_dir = base_dir
        self.dataset_name = dataset_name

        self.videos_mp4_dir = f"{self.base_dir}/videos_mp4"
        self.tracks_mp4_dir = f"{self.base_dir}/tracks_mp4/{self.dataset_name}"
        misc_utils.check_dirs(self.tracks_mp4_dir)

    def get_jobs_from_meta_dir(self):
        import glob
        parse_yid = lambda t: t.split('/')[-1][:11]
        all_files = sorted(glob.glob(f"{self.base_dir}/tracks_meta/{self.dataset_name}/*/*.gzip"))
        for fn in all_files[self.chunk_id::self.num_chunks]:
            yid = parse_yid(fn)
            tracks = load_tracks(fn, yid)
            if tracks:
                yield {'yid': yid, 'tracks': tracks}

    def get_jobs_from_db_file(self):
        to_skip = self.chunk_id + 1
        job = {'yid': None, 'tracks': []}
        for line in tqdm.tqdm(gzip.open(self.db_meta_file, mode='rt')):
            m = json.loads(line.strip())
            yid = m['yid']
            if yid != job['yid']:
                to_skip -= 1
                if len(job['tracks']) > 0:
                    yield job
                    to_skip = self.num_chunks - 1
                job = {'yid': yid, 'url': f"https://www.youtube.com/watch?v={yid}", 'tracks': []}
            if to_skip != 0:
                continue

            job['tracks'].append(
                Track(
                    yid,
                    ts=np.array(m['frame_ts']).astype(float),
                    boxes=np.array(m['frame_bboxes']).astype(float),
                    meta=m
                )
            )
        if to_skip < 0 and job['yid'] is not None:
            yield job

    def extract_tracks_from_video(self, vid, video_filepath, tracks, job_id):
        # Extract object tracks
        for t, track in enumerate(tracks):
            for box_exp in self.box_exp:
                track_fn = f"{self.tracks_mp4_dir}/BoxExp{box_exp}/{track.mp4_filename}"
                misc_utils.check_dirs(os.path.dirname(track_fn))

                # Check if already extracted
                if misc_utils.check_video(track_fn):
                    continue
                print(f'[{job_id}][{vid}][Track {t+1}/{len(tracks)}][BoxExp {box_exp}][{track.mp4_filename}].', flush=True)

                try:    # Somehow, we have a small number of tracks that start after the video ends(??)
                    vreader = VideoReader(video_filepath, start_time=track.ts[0], duration=min(track.ts[-1]-track.ts[0], self.max_track_len))
                except:
                    continue
                vwriter = None  # Lazy init of vwriter, bc frame size is unknown until 1st frame extraction.
                for frame in extract_track(boxes=track.boxes, timestamps=track.ts, reader=vreader, box_expansion=box_exp):
                    if vwriter is None:
                        vwriter = VideoWriter(track_fn, int(round(vreader.rate)), frame.shape[:2])
                    vwriter.write(frame)
                vwriter.container.close() if vwriter is not None else None

        print(f'[{job_id}][{vid}] Track extraction done.', flush=True)

    def extract_all(self):
        get_jobs_fcn = self.get_jobs_from_db_file if self.db_meta_file is not None else self.get_jobs_from_meta_dir
        for job_id, meta in enumerate(get_jobs_fcn()):
            vid, tracks = meta['yid'], meta['tracks']
            print(f'[{job_id}][{vid}] Start', flush=True)

            # Check video download
            video_filepath = os.path.join(self.videos_mp4_dir, vid[:2], vid + '.mp4')
            if not misc_utils.check_video(video_filepath):
                print(f"Video loading error. Skipping video file: {video_filepath}", flush=True)
                return

            self.extract_tracks_from_video(vid, video_filepath, tracks, job_id=f"{job_id}")


class Launcher:
    def __call__(self, args):
        for k in args.__dict__:
            print(f"{k}: {args.__dict__[k]}")
        ObjectTrackExtractor(
            args.base_dir,
            db_meta_file=args.db_meta_file,
            dataset_name=args.dataset_name,
            num_chunks=args.num_chunks,
            chunk_id=args.chunk_id
        ).extract_all()


if __name__ == '__main__':
    args = parse_arguments()

    if args.slurm:
        job_names = os.popen('squeue -o %j -u $USER').read().split("\n")
        slurm_job_name = f"extract-tracks-{args.db_meta_file.split('/')[-1].split('.jsonl')[0]}-{args.num_chunks}-{args.chunk_id}"
        if slurm_job_name in job_names:
            print(f'Skipping {slurm_job_name} because already in queue')
            exit(0)

        # Submit jobs
        executor = submitit.AutoExecutor(folder='./slurm_logs/', slurm_max_num_timeout=20, cluster=None)
        executor.update_parameters(
            timeout_min=1440,                # Requeue every 12hr
            slurm_partition=args.partition,
            cpus_per_task=4,
            gpus_per_node=0,
            nodes=1,
            tasks_per_node=1,
            mem_gb=32,
            slurm_additional_parameters={"exclude": "euler01,euler03,euler09,euler27,euler24,euler141,euler128,euler129,euler130,euler131,euler132,euler133"})
        executor.update_parameters(name=slurm_job_name)
        executor.submit(Launcher(), args)
        print(f"Job submitted: {slurm_job_name}")
    else:
        Launcher()(args)