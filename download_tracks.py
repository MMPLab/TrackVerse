import os
import argparse
import submitit
from utils import youtube as yt_utils
import json, gzip

from extract_tracks import ObjectTrackExtractor, Track

import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='youtube video downloader')
    # Scheduling
    parser.add_argument("--slurm", default=False, action="store_true")
    parser.add_argument("--partition", default='research')
    parser.add_argument("--world_size", default=1, type=int, help="scheduling chunks")
    parser.add_argument("--rank", default=0, type=int, help="scheduling chunk id")
    # Downloader args
    parser.add_argument('--base_dir', default='./', help='Dataset directory')
    parser.add_argument('--db_meta_file',  default='metadata/LVIS-NoStatic-1121K-cls1171CB2500-processed.jsonl.gzip',
                        help='The path to the database jsonl meta file.')
    parser.add_argument('--remove_video_mp4', default=False, action='store_true',
                        help='Remove the original video mp4 files after extracting tracks.')
    # Cookie file for youtube-dl
    parser.add_argument('--cookiefile', default=None, help='The path to the cookie file.')
    return parser.parse_args()

class TrackDownloader(object):
    def __init__(self, base_dir, db_meta_file, cookiefile, remove_video_mp4=False):
        """ Downloader for YouTube videos and track extraction.
            Args:
                base_dir (str): The base directory for the dataset.
                db_meta_file (str): The path to the database metadata file.
                cookiefile (str): The path to the cookie file for youtube-dl.
                remove_video_mp4 (bool): Whether to remove the original video mp4 files after extraction.
        """
        self.base_dir = base_dir
        self.db_meta_file = db_meta_file
        self.remove_video_mp4 = remove_video_mp4

        # Output directories
        self.videos_dir = os.path.join(self.base_dir, 'videos_mp4')
        self.tracks_dir = os.path.join(self.base_dir, 'tracks_mp4')

        self.downloader = yt_utils.YoutubeDL(self.videos_dir, cookiefile)
        self.extractor = ObjectTrackExtractor(base_dir)

    def process_video(self, youtube_id, tracks, job_id, remove_video_mp4):
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
        
        # Remove the original video if requested
        if self.remove_video_mp4:
            if os.path.exists(video_filepath):
                os.remove(video_filepath)
                print(f'[{job_id}][{youtube_id}] Removed original video file: {video_filepath}', flush=True)
            else:
                print(f'[{job_id}][{youtube_id}] Original video file not found: {video_filepath}', flush=True)


def scheduled_jobs(meta_file_path, rank=0, world_size=1):
    """Generate jobs assigned to a particular worker based on the rank and total number of workers.
    Each job consists of a YouTube video ID and its associated tracks.

    Yields:
        tuple: A tuple containing the job index, YouTube ID, and the associated tracks data.
    """
    # Note: In the metadata file, tracks from the same YouTube video are grouped together.
    job_id, youtube_id, tracks = -1, '', []
    for line in gzip.open(meta_file_path, 'rt'):
        m = json.loads(line)
        # Check if the YouTube ID has changed (indicating a new video).
        if youtube_id != m['yid']:
            if job_id % world_size == rank and job_id >= 0:
                yield job_id, youtube_id, tracks
            job_id += 1
            youtube_id, tracks = m['yid'], []
        tracks.append(Track(
            youtube_id,
            fn=m['track_mp4_filename'],
            ts=np.array(m['track_ts']).astype(float),
            boxes=np.array(m['frame_bboxes']).astype(float),
            meta=m,
        ))


class Launcher:
    def __call__(self, args):
        downloader = TrackDownloader(args.base_dir, args.db_meta_file, args.cookiefile, args.remove_video_mp4)
        for job_id, youtube_id, tracks_meta in scheduled_jobs(f'{args.base_dir}/{args.db_meta_file}', args.rank, args.world_size):
            downloader.process_video(youtube_id, tracks_meta, job_id=job_id)


if __name__ == '__main__':
    args = parse_arguments()
    if args.slurm:
        job_names = os.popen('squeue -o %j -u $USER').read().split("\n")
        slurm_job_name = f"download-tracks-{args.rank}of{args.world_size}"
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
