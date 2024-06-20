import os
import argparse
import submitit
import glob
from utils import misc as misc_utils


def parse_arguments():
    parser = argparse.ArgumentParser(description='youtube video downloader')
    # Scheduling
    parser.add_argument("--slurm", default=False, action="store_true")
    parser.add_argument("--partition", default='research')
    parser.add_argument("--world_size", default=1, type=int, help="scheduling chunks")
    parser.add_argument("--rank", default=0, type=int, help="scheduling chunk id")
    # Downloader args
    parser.add_argument('--base_dir', default='./TrackVerseDB', help='Dataset directory')
    parser.add_argument('--dataset_domain', default="LVIS", help='The class domain of the dataset.')
    parser.add_argument('--deface_cmd', default="deface", help='Deface command.')
    return parser.parse_args()

def deface(cmd, mp4_file):
    os.system(f"{cmd} {mp4_file} -o {mp4_file.replace('tracks_mp4', 'tracks_mp4_defaced')}")
    
def scheduled_jobs(tracks_dir, rank=0, world_size=1):
    for mp4_dir in  glob.glob(f"{tracks_dir}/*"):
        misc_utils.check_dirs(mp4_dir.replace('tracks_mp4', 'tracks_mp4_defaced'))
    
    mp4_files = glob.glob(f"{tracks_dir}/*/*.mp4")
    jobs = sorted(mp4_files)[rank::world_size]
    for job in jobs:
        yield job

class Launcher:
    def __call__(self, args):
        tracks_dir = f"{args.base_dir}/tracks_mp4/{args.dataset_domain}"
        for mp4_file in scheduled_jobs(tracks_dir, rank=0, world_size=1):
            deface(args.deface_cmd, mp4_file)
        
if __name__ == '__main__':
    args = parse_arguments()
    if args.slurm:
        job_names = os.popen('squeue -o %j -u $USER').read().split("\n")
        slurm_job_name = f"deface-tracks-{args.rank}of{args.world_size}"
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
