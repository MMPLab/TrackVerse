import os
import gzip
import argparse
import json
import submitit

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.models.optical_flow import raft_large
from torchvision import transforms as T

from utils import avio
from utils import misc as misc_utils


def parse_arguments():
    parser = argparse.ArgumentParser(description='youtube video downloader')
    parser.add_argument("--slurm", default=False, action="store_true")
    parser.add_argument("--partition", default="research")
    parser.add_argument('--world_size', default=1, type=int, help='Number of chunks')
    parser.add_argument('--rank', default=0, type=int, help='Number of chunks')

    parser.add_argument('--base_dir', default='./TrackVerseDB', type=str, help='Working Directory')
    parser.add_argument('--dataset_domain', default='LVIS', help='The class domain of the dataset.')
    parser.add_argument('--db_meta_file',  default='tracks_subsets/TrackVerseLVIS-Full.jsonl.gzip',type=str,
                        help='The path to the database jsonl meta file.')
    parser.add_argument('--metric', default='embeddings', type=str,
                        choices=["embeddings", "motion", 'blur'],
                        help='Curation criterion')
    return parser.parse_args()


class VisualMetrics(object):
    def __init__(self, base_dir, dataset_domain, db_meta_file, metric='embedding', world_size=1, rank=0):
        self.base_dir = base_dir
        self.dataset_domain = dataset_domain
        self.db_meta_file = f"{base_dir}/{db_meta_file}"
        self.metric = metric
        self.world_size = world_size
        self.rank = rank

        self.tracks_mp4_dir = f"{self.base_dir}/tracks_mp4/{self.dataset_domain}"
        self.tracks_metric_dir = f"{self.base_dir}/tracks_{self.metric}/{self.dataset_domain}"
        misc_utils.check_dirs(self.tracks_metric_dir)
        self.procs_tracker = misc_utils.ProgressTracker(os.path.join(self.tracks_metric_dir, 'completed.txt'))
        self.model = None

    def scheduled_jobs(self):
        for job_id, line in enumerate(gzip.open(self.db_meta_file, mode='rt')):
            if job_id % self.world_size == self.rank:
                m = json.loads(line.strip())
                track_fn = f"{self.tracks_mp4_dir}/{m['fn']}"
                if misc_utils.check_video(track_fn):
                    yield job_id, m

    @torch.no_grad()
    def compute_optical_flow(self, job_meta, job_id):
        device = 'cuda'
        if self.model is None:
            self.model = raft_large(pretrained=True, progress=False).to(device)
            self.model = self.model.eval()

        frame_rate = 16
        batch_size = 16
        skip_rate = 2   # it will actually skip twice the skip rate
        q = torch.tensor([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.]).cuda()
        track_fn = f"{self.tracks_mp4_dir}/{job_meta['fn']}"
        transform = T.Compose([T.ToTensor(), T.Normalize(mean=0.5, std=0.5)])
        video = avio.VideoDB(track_fn, frame_rate=frame_rate, transform=transform, skip_nonkey_frames=False)
        loader = DataLoader(video, batch_size=batch_size*skip_rate, num_workers=1)
        flow_stats = []
        for frames, ts in loader:
            t, d, h, w = frames.shape
            if t % 2 == 1:  # we need an even number of frames
                frames = frames[:-1]
            if t < 2:       # not enough frames for optical flow
                break

            # create pairs of consecutive frames and skip some for speed up
            t, d, h, w = frames.shape
            frames = frames.view(t//2, 2, d, h, w)[::skip_rate]     # Tx2x3xHxW

            # Resize all frames
            npairs, _, d, h, w = frames.shape
            frames = frames.flatten(0, 1)
            if h < w:   # Wide frame
                frames = F.interpolate(frames.to(device), size=(360, int(w/h*360)//8*8), mode='bilinear')
            else:       # Tall frame
                frames = F.interpolate(frames.to(device), size=(int(h/w*360)//8*8, 360), mode='bilinear')
            _, _, nh, nw = frames.shape
            frames = frames.view(npairs, 2, d, nh, nw)     # Tx2x3xHxW

            # Compute flow
            flows = self.model(frames[:, 0], frames[:, 1])     # Tx2xHxW
            flows = flows[-1]       # Raft output intermediate flow estimations. Keep only last iteration
            flow_magn = flows.norm(p=2, dim=1).flatten(1)       # TxHW
            flow_stats.append(flow_magn.quantile(q=q).T.cpu())  # Q (should have been dim=1 to get TxQ, but this is fine)
        flow_stats = torch.cat(flow_stats, 0).numpy()

        output_fn = f"{self.tracks_metric_dir}/{job_meta['fn'].replace('.mp4', '.npy')}"
        misc_utils.check_dirs(os.path.dirname(output_fn))
        np.save(output_fn, flow_stats)
        print(f"[{job_id}] Finished processing {output_fn}.", flush=True)

    @torch.no_grad()
    def compute_embeddings(self, job_meta, job_id):
        device = 'cuda'
        if self.model is None:
            self.model = resnet50(pretrained=True).to(device)
            self.model.fc = nn.Identity()
            self.model = self.model.eval()

        track_fn = f"{self.tracks_mp4_dir}/{job_meta['fn']}"
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        video = avio.VideoDB(track_fn, transform=transform, skip_nonkey_frames=True)
        loader = DataLoader(video, batch_size=32, num_workers=1)
        x_list = [self.model(frames.to(device)).cpu() for frames, ts in loader]
        x = torch.concat(x_list).mean(0).numpy()

        # Save to file
        output_fn = f"{self.tracks_metric_dir}/{job_meta['fn'].replace('.mp4', '.npy')}"
        misc_utils.check_dirs(os.path.dirname(output_fn))
        np.save(output_fn, x)
        print(f"[{job_id}] Finished processing {output_fn}.", flush=True)

    def process_video(self, job_meta, job_id):
        if self.procs_tracker.check_completed(job_meta['fn']):
            return  # Skip. Already processed

        if self.metric == 'embeddings':
            self.compute_embeddings(job_meta, job_id)
        elif self.metric == 'motion':
            self.compute_optical_flow(job_meta, job_id)
        else:
            raise NotImplementedError

        self.procs_tracker.add(job_meta['fn'])

    def process_all(self):
        for job_id, job_meta in self.scheduled_jobs():
            self.process_video(job_meta, job_id=job_id)


class Launcher:
    def __call__(self, args):
        VisualMetrics(
            args.base_dir, args.dataset_domain, args.db_meta_file,
            metric=args.metric, world_size=args.world_size, rank=args.rank
        ).process_all()


if __name__ == '__main__':
    args = parse_arguments()

    if args.slurm:
        job_names = os.popen('squeue -o %j -u $USER').read().split("\n")
        slurm_job_name = f"compute-{args.metric}-{args.rank}of{args.world_size}"
        if slurm_job_name in job_names:
            print(f'Skipping {slurm_job_name} because already in queue', flush=True)
            exit(0)

        # Submit jobs
        executor = submitit.AutoExecutor(folder='./slurm_logs/', slurm_max_num_timeout=20, cluster=None)
        executor.update_parameters(
            timeout_min=1440,                # Requeue every day
            slurm_partition=args.partition,
            cpus_per_task=10,
            gpus_per_node=1,
            nodes=1,
            tasks_per_node=1,
            mem_gb=64,
            slurm_additional_parameters={"exclude": "euler05"},
            slurm_signal_delay_s=20)
        executor.update_parameters(name=slurm_job_name)
        executor.submit(Launcher(), args)
        print(f"Job submitted: {slurm_job_name}", flush=True)
    else:
        Launcher()(args)
