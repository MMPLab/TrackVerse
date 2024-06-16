import os
import argparse
import submitit
import torch
from utils import youtube as yt_utils
from utils import segments as segm_utils
import tqdm, json, gzip
from collections import defaultdict

from utils.avio import VideoReader, VideoWriter
from utils import misc as misc_utils

from extract_tracks import ObjectTrackExtractor

import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='youtube video downloader')
    # Scheduling
    parser.add_argument("--slurm", default=False, action="store_true")
    parser.add_argument("--partition", default='research')
    parser.add_argument("--world_size", default=1, type=int, help="scheduling chunks")
    parser.add_argument("--rank", default=0, type=int, help="scheduling chunk id")
    # Downloader args
    parser.add_argument('--base_dir', default='./TrackVerse', help='Dataset directory')
    parser.add_argument('--db_meta_file',  default='tracks_subsets/TrackVerseLVIS-Full-4M.jsonl.gzip',type=str,
                        help='The path to the database jsonl meta file.')
    parser.add_argument('--dataset_domain', default="TrackVerseLVIS", help='The class domain of the dataset.')
    parser.add_argument("--remove_video_mp4", default=False, action="store_true", help="Remove video after processing")

    return parser.parse_args()


class Track:
    def __init__(self, yid, ts, boxes, meta):
        self.yid = yid
        self.ts = ts
        self.boxes = boxes
        self.fn = meta['mp4_filename']
        self.meta = meta

def load_tracks(yid, tracks_meta):
    tracks = []
    for m in tracks_meta:
        tracks.append(Track(
            yid,
            ts=np.array(m['track_ts']).astype(float),
            boxes=np.array(m['frame_bboxes']).astype(float),
            meta=m,
        ))
    return tracks

class TrackDownloader(object):
    def __init__(self, args):
        self.base_dir = args.base_dir
        self.db_meta_file = args.db_meta_file
        self.remove_video_mp4 = args.remove_video_mp4
        self.args = args

        # Output directories
        self.downl_dir = os.path.join(self.base_dir, 'videos_mp4')
        self.tracks_mp4_dir = os.path.join(self.base_dir, 'tracks_mp4', args.dataset_domain)

        self.downloader = yt_utils.YoutubeDL(self.downl_dir)
        self.extractor = ObjectTrackExtractor(args.base_dir, args.dataset_domain)

    def process_video(self, youtube_id, tracks, job_id):
        # Download the orignal video
        dl_status, video_filepath = self.downloader.download_video(youtube_id)
        if dl_status == yt_utils.STATUS.FAIL:
            print(f'[{job_id}][{youtube_id}] Download failed.', flush=True)
            return
        elif dl_status == yt_utils.STATUS.DONE:
            print(f'[{job_id}][{youtube_id}] Already downloaded.', flush=True)
        else:
            print(f'[{job_id}][{youtube_id}] Download successful.', flush=True)

        # Extract tracks
        self.extractor.extract_tracks_from_video(youtube_id, tracks, job_id)
            
        if self.remove_video_mp4:
            os.remove(video_filepath)


def scheduled_jobs(downloader, rank=0, world_size=1):
    """Generate jobs assigned to a particular worker based on the rank and total number of workers.

    Args:
        downloader (TrackDownloader): The track downloader.
        world_size (int, optional): How many chunks to split the work in. 
        rank (int, optional): Chunk ID.

    Yields:
        tuple: A tuple containing the job index, YouTube ID, and the associated tracks data.
    """
    yid2meta = defaultdict(list)
    
    subset_gzip = f'{downloader.base_dir}/{downloader.db_meta_file}'
    for line in tqdm.tqdm(gzip.open(subset_gzip, 'rt')):
        data = json.loads(line)
        yid2meta[data['yid']].append(data)
        
    for job_id, (youtube_id, tracks_meta) in enumerate(yid2meta.items()):
        if job_id % world_size == rank:
            tracks = load_tracks(youtube_id, tracks_meta)
            yield job_id, youtube_id, tracks

            
def main_worker(args):
    downloader = TrackDownloader(args)
    for job_id, youtube_id, tracks_meta in scheduled_jobs(downloader, args.rank, args.world_size):
        downloader.process_video(youtube_id, tracks_meta, job_id=job_id)

class Launcher:
    def __call__(self, args):
        for k in args.__dict__:
            print(f"{k}: {args.__dict__[k]}")
        main_worker(args)


if __name__ == '__main__':
    args = parse_arguments()
    if args.slurm:
        job_names = os.popen('squeue -o %j -u $USER').read().split("\n")
        slurm_job_name = f"download-videos-{args.rank}of{args.world_size}"
        if slurm_job_name in job_names:
            print(f'Skipping {slurm_job_name} because already in queue')
            exit(0)

        # Submit jobs
        executor = submitit.AutoExecutor(folder='./slurm_logs/', slurm_max_num_timeout=20, cluster=None)
        executor.update_parameters(
            timeout_min=1440,                # Requeue every 24hr
            slurm_partition=args.partition,
            cpus_per_task=10,
            gpus_per_node=1,
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
