# Adapted from https://github.com/microsoft/XPretrain
import os

import PIL.Image
import numpy as np
import scipy.ndimage as ndi

from utils.avio import VideoReader, VideoWriter
from utils import misc as misc_utils


class Track:
    def __init__(self, yid, fn, ts, boxes, meta):
        self.yid = yid
        self.fn = fn
        self.ts = ts
        self.boxes = boxes
        self.meta = meta

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


def extract_track(boxes, timestamps, reader):
    timestamps = np.array(timestamps)
    boxes = np.array(boxes)

    # smooth boxes
    ts_diff = np.diff(timestamps).mean()
    std = int(0.66/ts_diff)
    boxes_smooth = np.stack([ndi.gaussian_filter(b, std) for b in boxes.T], 1)

    # make aspect ratio constant across the track
    x, y, w, h = boxes_smooth.copy().T
    cx, cy, ar = x + w/2, y + h/2, w/h
    avg_ar = ar.mean()

    if avg_ar < 2/3:   trg_ar = 2/3
    elif avg_ar > 3/2: trg_ar = 3/2
    else:              trg_ar = avg_ar

    # Adjust xywh
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

        crop = extract_crop_nearest(frame, box, out_shape)
        yield crop


class ObjectTrackExtractor:
    def __init__(self, base_dir):
        """ Extract object tracks from videos.
            Args:
                base_dir (str): The base directory.
                box_exp (list, optional): The box expansion values. 
        """
        self.base_dir = base_dir

        self.videos_mp4_dir = f"{self.base_dir}/videos_mp4"
        self.tracks_mp4_dir = f"{self.base_dir}/tracks_mp4"
        misc_utils.check_dirs(self.tracks_mp4_dir)

    def extract_tracks_from_video(self, vid, tracks, job_id):
        # Check video download
        video_filepath = os.path.join(self.videos_mp4_dir, vid[:2], vid + '.mp4')
        if not misc_utils.check_video(video_filepath):
            print(f"Video loading error. Skipping video file: {video_filepath}", flush=True)
            return

        # Extract object tracks
        print(f'[{job_id}][{vid}] Start track extraction', flush=True)
        for t, track in enumerate(tracks):
            track_fn = f"{self.tracks_mp4_dir}/{track.fn}"
            misc_utils.check_dirs(os.path.dirname(track_fn))

            # Check if already extracted
            if misc_utils.check_video(track_fn):
                continue
            print(f'[{job_id}][{vid}][Track {t+1}/{len(tracks)}][{track.fn}].', flush=True)

            try:    # Somehow, we found a small number of tracks that start after the video ends(??)
                vreader = VideoReader(video_filepath, start_time=track.ts[0], duration=min(track.ts[-1]-track.ts[0], 180))
            except:
                continue
            vwriter = None  # Lazy init of vwriter, bc frame size is unknown until 1st frame extraction.
            for frame in extract_track(boxes=track.boxes, timestamps=track.ts, reader=vreader):
                if vwriter is None:
                    vwriter = VideoWriter(track_fn, int(round(vreader.rate)), frame.shape[:2])
                vwriter.write(frame)
            vwriter.container.close() if vwriter is not None else None

        print(f'[{job_id}][{vid}] Track extraction done.', flush=True)
