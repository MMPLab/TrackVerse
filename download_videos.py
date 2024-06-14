import os
import argparse
import submitit
import torch
from utils import youtube as yt_utils
from utils import segments as segm_utils


def parse_arguments():
    parser = argparse.ArgumentParser(description='youtube video downloader')
    # Scheduling
    parser.add_argument("--slurm", default=False, action="store_true")
    parser.add_argument("--partition", default='research')
    parser.add_argument("--world_size", default=1, type=int, help="scheduling chunks")
    parser.add_argument("--rank", default=0, type=int, help="scheduling chunk id")
    # Downloader args
    parser.add_argument('--base_dir', default='/home/pmorgado/datasets/TrackVerse/',
                        help='Dataset directory')
    parser.add_argument('--yid_index_fn', default="assets/trackverse-yids-all.txt",
                        help='index of youtube ids to download.')
    parser.add_argument("--skip_cartoon_filter", default=False, action="store_true")
    parser.add_argument("--skip_aesthetics_filter", default=False, action="store_true")

    return parser.parse_args()


class TrackVerseDL(object):
    def __init__(self, args):
        self.base_dir = args.base_dir
        self.index_fn = args.yid_index_fn
        self.world_size = args.world_size
        self.rank = args.rank
        self.args = args

        # Output directories
        self.downl_dir = os.path.join(self.base_dir, 'videos_mp4')
        self.segm_dir = os.path.join(self.base_dir, 'videos_segm')

        self.downloader = yt_utils.YoutubeDL(self.downl_dir)
        self.segm_extractor = segm_utils.SegmentExtractor(self.downl_dir, self.segm_dir, args.skip_cartoon_filter, args.skip_aesthetics_filter)

    def scheduled_jobs(self):
        for job_id, ln in enumerate(open(self.index_fn)):
            youtube_id = ln.strip()
            if len(youtube_id) != 11:
                continue
            if job_id % self.world_size == self.rank:
                yield job_id, youtube_id

    def process_video(self, youtube_id, job_id):
        # Download
        dl_status, video_filepath = self.downloader.download_video(youtube_id)
        if dl_status == yt_utils.STATUS.FAIL:
            print(f'[{job_id}][{youtube_id}] Download failed.', flush=True)
            return
        elif dl_status == yt_utils.STATUS.DONE:
            print(f'[{job_id}][{youtube_id}] Already downloaded.', flush=True)
        else:
            print(f'[{job_id}][{youtube_id}] Download successful.', flush=True)

        # Extract segments
        segm_status, segm_filepath = self.segm_extractor.extract_segments(video_filepath)
        if segm_status == segm_utils.STATUS.DONE:
            print(f'[{job_id}][{youtube_id}] Already split into segments.', flush=True)
        else:
            num_segments = len(list(open(segm_filepath, "r")))
            print(f'[{job_id}][{youtube_id}] Found {num_segments} segments.', flush=True)

    def process_all(self):
        for job_id, youtube_id in self.scheduled_jobs():
            self.process_video(youtube_id, job_id=job_id)


class Launcher:
    def __call__(self, args):
        torch.multiprocessing.set_start_method('spawn')
        TrackVerseDL(args).process_all()


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
