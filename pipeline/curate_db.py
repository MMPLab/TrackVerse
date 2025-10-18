import argparse
import os
import glob
import json
import gzip
import random
import tqdm
from collections import defaultdict
import numpy as np
import torch
import torch.multiprocessing as mp
import submitit


def parse_arguments():
    parser = argparse.ArgumentParser(description='youtube video downloader')
    parser.add_argument("--slurm", default=False, action="store_true")
    parser.add_argument("--partition", default="research")
    parser.add_argument('--base_dir', default='./TrackVerseDB', type=str, help='Dataset Directory')
    parser.add_argument('--dataset_domain', default='LVIS', type=str, help='The class domain of the dataset.')
    parser.add_argument('--index_file',  default='tracks_subsets/hdvila_lvis/LVIS-4M.jsonl.gzip',
                        help='File containing all parsed tracks.')
    parser.add_argument('--action', default='embeddings', type=str,
                        choices=["index", "sample_random", "sample_class_balanced", "sample_max_motion", "sample_min_red"], help='What to do.')
    parser.add_argument('--N', default=[82, 184, 259, 392], nargs='+', help='Number of tracks in thousands')
    parser.add_argument('--Nc', default=[100, 300, 500, 1000], nargs='+', help='Number of tracks per class')
    parser.add_argument('--min_nn_dist', default='0', help='Maximum embedding distance between selected tracks')
    parser.add_argument('--max_spt_iou', default='1', help='Maximum IoU between selected tracks')
    parser.add_argument('--iou_tbuff', default='inf', help='No penalty time buffer when computing IoU')
    parser.add_argument('--min_motion', default='0', help='Minimum motion strength of selected tracks')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of chunks')
    args = parser.parse_args()
    args.N = [int(n) for n in args.N]
    args.Nc = [int(n) for n in args.Nc]
    return args


def softmax(x, temperature=1.0):
    """Compute softmax values for each sets of scores in x"""
    x = np.array(x)
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    bbox1: tuple, (x1, y1, w1, h1)
    bbox2: tuple, (x2, y2, w2, h2)

    Returns:
    float, IoU value
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # If the intersection is valid (i.e., non-negative area)
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # Calculate the IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou


class Curator(object):
    def __init__(self, base_dir, dataset_domain, index_file, num_workers=4):
        self.base_dir = base_dir
        self.dataset_domain = dataset_domain
        self.index_file = os.path.join(self.base_dir, index_file)
        self.num_workers = num_workers

        self.track_fns, self.track_yid, self.track_class, self.track_logit, self.track_conf, self.track_tid = [], [], [], [], [], []
        self.track_ts, self.track_bbox = [], []
        self.class2idx = defaultdict(list)

        self.labeled_vids = set([ln[:11] for ln in open(f'assets/trackverse-verified-6perclass.txt') if ln.strip()])
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('Using {} device'.format(self.device))

    @staticmethod
    def parse_yid(t):
        return t.split('/')[-1][:11]

    @staticmethod
    def parse_label(t):
        return t.replace('photo of a ', '').split(' (also')[0]

    def get_emb(self, tid):
        try:
            return np.load(f"{self.base_dir}/tracks_embeddings/{self.dataset_domain}/{self.track_fns[tid].replace('.mp4', '.npy')}")
        except Exception:
            return np.randn(2048)

    @staticmethod
    def index_worker(tracks_db_part, meta_fns_part, chunk_id):
        print(f"Start chunk {chunk_id}. Storing results in {tracks_db_part}.")
        with gzip.open(tracks_db_part, 'wt') as fp:
            for it, fn in enumerate(meta_fns_part):
                for ln in gzip.open(fn, 'rt'):
                    fp.write(ln)
                if it % 100 == 0:
                    print(f'[Chunk {chunk_id}] Processed {it}/{len(meta_fns_part)}', flush=True)

    def index_db(self):
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)

        # Split up processing into multiple chunks
        pool = mp.Pool(self.num_workers)
        num_chunks = 32
        meta_fns = sorted(glob.glob(f'{self.base_dir}/tracks_meta/{self.dataset_domain}/*/*.gzip'))
        results = [
            pool.apply_async(self.index_worker, args=(self.index_file + f'.part{ck}', meta_fns[ck::num_chunks], ck))
            for ck in range(num_chunks)]
        for res in results: res.get()

        # Join parts
        for ck in tqdm.tqdm(range(num_chunks)):
            os.system(f'cat {args.index_file}.part{ck} >> {args.index_file}')
        for ck in tqdm.tqdm(range(num_chunks)):
            os.system(f'rm {args.index_file}.part{ck}')

    def load_tracks(self):
        # Takes about 5 min to load ~600k tracks
        print("Loading all tracks from {}".format(self.index_file))
        db = set()
        for tid, line in tqdm.tqdm(enumerate(gzip.open(self.index_file))):
            data = json.loads(line)
            if data['fn'] in db:
                continue
            db.add(data['fn'])
            top10_probs = softmax(data['top10_wcls'][0], temperature=0.1)

            self.track_tid.append(tid)
            self.track_fns.append(data['fn'])
            self.track_yid.append(data['yid'])
            self.track_class.append(self.parse_label(data['top10_desc'][0]))
            self.track_logit.append(data['top10_wcls'][0][0])
            self.track_conf.append(top10_probs[0] - top10_probs[1])
            self.track_ts.append(np.array(data['track_ts']))
            self.track_bbox.append(np.array(data['track_bbox']))

        # Filter out test videos/tracks
        keep_idx = [i for i, vid in enumerate(self.track_yid) if vid not in self.labeled_vids]
        self.filter(keep_idx)

        # Index by class
        self.class2idx = defaultdict(list)
        for idx, cls in enumerate(self.track_class):
            self.class2idx[cls].append(idx)

        print(
            f"Found {len(self.track_fns)} tracks.\n"
            f" - fn: {self.track_fns[0]}\n"
            f" - yid: {self.track_yid[0]}\n"
            f" - class: {self.track_class[0]}\n"
            f" - logit: {self.track_logit[0]}\n"
            f" - conf: {self.track_conf[0]}\n"
        )

    def filter(self, keep_idx):
        self.track_tid = [self.track_tid[i] for i in keep_idx]
        self.track_fns = [self.track_fns[i] for i in keep_idx]
        self.track_yid = [self.track_yid[i] for i in keep_idx]
        self.track_class = [self.track_class[i] for i in keep_idx]
        self.track_logit = [self.track_logit[i] for i in keep_idx]
        self.track_conf = [self.track_conf[i] for i in keep_idx]
        self.track_ts = [self.track_ts[i] for i in keep_idx]
        self.track_bbox = [self.track_bbox[i] for i in keep_idx]

    def save_subsets(self, subset_fn, subset_tids):
        # Takes about 2 min per 50k samples
        if not isinstance(subset_fn, dict):
            subset_fn = {'mySubset': subset_fn}
            subset_tids = {'mySubset': subset_tids}
        for k in subset_tids:
            subset_tids[k] = set(subset_tids[k])
            print(f'Saving {k} with {len(subset_tids[k])} samples into {subset_fn[k]}')

        fps = {k: gzip.open(subset_fn[k], 'wt') for k in subset_tids}
        for idx, ln in tqdm.tqdm(enumerate(gzip.open(self.index_file, 'rt'))):
            for k in subset_tids:
                if idx in subset_tids[k]:
                    fps[k].write(ln)

    @staticmethod
    def sort_population(population, values=None, T=0., mode='High'):
        if T == float('inf'):
            output = [i for i in population]
            random.shuffle(output)
        elif T == 0:
            c = -1 if mode == 'High' else 1
            idx = np.argsort(values*c)
            output = [population[i] for i in idx]
        else:
            c = 1 if mode == 'High' else -1
            sample_probs = softmax(values*c, T)    # skew sampling towards large or small value
            output = np.random.choice(population, size=len(population), replace=False, p=sample_probs)

        return output

    @staticmethod
    def spt_iou(bbox1, bbox2, iou_tbuff):
        # Pair of frames with minimum time diff
        t1, t2 = bbox1[:, 0], bbox2[:, 0]
        if t1[-1] < t2[0]:
            t1, t2, bbox1, bbox2 = t1[-1:], t2[:1], bbox1[-1:, 1:], bbox2[:1, 1:]
        elif t2[-1] < t1[0]:
            t1, t2, bbox1, bbox2 = t1[:1], t2[-1:], bbox1[:1, 1:], bbox2[-1:, 1:]
        else:
            st, et = max(t1[0], t2[0]), min(t1[-1], t2[-1])
            m1 = np.logical_and(t1 >= st, t1 <= et)
            m2 = np.logical_and(t2 >= st, t2 <= et)
            t1, t2, bbox1, bbox2 = t1[m1], t2[m2], bbox1[m1, 1:], bbox2[m2, 1:]
            if len(t1) != len(t2):
                nframes = min(len(t1), len(t2))
                idx1 = np.linspace(0, len(t1)-1, nframes, endpoint=True).astype(int)
                idx2 = np.linspace(0, len(t2)-1, nframes, endpoint=True).astype(int)
                t1, t2, bbox1, bbox2 = t1[idx1], t2[idx2], bbox1[idx1], bbox2[idx2]
        if iou_tbuff < float('inf'):
            dt_decays = [1. / max(np.sqrt(np.abs(tt2 - tt1)/iou_tbuff), 1) for tt1, tt2 in zip(t1, t2)]
        else:
            dt_decays = [1.] * len(bbox1)

        return np.mean([calculate_iou(bb1, bb2) * decay
                        for bb1, bb2, decay in zip(bbox1, bbox2, dt_decays)])
        # else:

    def constrained_sampling(self, sorted_population, N, min_nn_dist=0., max_spt_iou=1., iou_tbuff=float('inf'), min_motion=0.):
        N = min(N, len(sorted_population))
        sorted_track_names = [self.track_fns[i] for i in sorted_population]
        db = VisualMetricsDB(self.base_dir, self.dataset_domain,
                             track_fns=sorted_track_names, track_ids=sorted_population,
                             return_embs=min_nn_dist>0, return_motion_stats=min_motion>0)
        loader = DataLoader(db, num_workers=self.num_workers, shuffle=False, batch_size=1)

        selected_population, selected_embs = [], None
        max_rounds = 5
        for ep in range(max_rounds):
            for data in tqdm.tqdm(loader):
                tid = data['tid'][0].item()
                if tid in selected_population:
                    continue

                # check for low motion
                if min_motion > 0:
                    if data['motion']['q90'][0] < min_motion:
                        continue

                # check for low bbox overlap
                if max_spt_iou < 1:
                    # Find tracks from same video
                    track_yid = self.track_fns[tid][3:14]
                    same_video_tids = [sel_tid for sel_tid in selected_population
                                       if self.track_fns[sel_tid][3:14] == track_yid]
                    if same_video_tids:
                        bbox = np.concatenate((np.array(self.track_ts[tid])[:, None], np.array(self.track_bbox[tid])), 1)
                        same_video_bbox = [np.concatenate((np.array(self.track_ts[i])[:, None], np.array(self.track_bbox[i])), 1)
                                           for i in same_video_tids]
                        spt_ious = [self.spt_iou(bbox, sel_bbox, iou_tbuff) for sel_bbox in same_video_bbox]
                        if max(spt_ious) > max_spt_iou:
                            continue

                # check for low nn distance (high redundancy)
                if min_nn_dist > 0:
                    emb = data['embedding'].to(self.device)
                    if selected_embs is not None:
                        nn_dist = torch.cdist(selected_embs, emb).min()
                        if nn_dist < min_nn_dist:
                            continue

                # add to subset
                selected_population.append(tid)
                if min_nn_dist > 0:
                    selected_embs = emb if selected_embs is None else torch.cat((selected_embs, emb))
                if len(selected_population) == N:
                    return sorted(selected_population)

            # Lower thresholds if couldn't find enough samples
            if min_nn_dist > 0:
                min_nn_dist = min_nn_dist * 0.8 if ep < max_rounds-2 else 0.
            if min_motion > 0:
                min_motion = min_motion * 0.8 if ep < max_rounds-2 else 0.
            if max_spt_iou < 1:
                max_spt_iou = 1 - (1 - max_spt_iou) * 0.8 if ep < max_rounds-2 else 1.

    def sample_random(self, N_list):
        if not isinstance(N_list, (list, tuple)):
            N_list = [N_list]

        # Load track index
        if not self.track_fns:
            self.load_tracks()

        selected_tracks, subset_gzip = {}, {}
        sorted_seq = self.sort_population(range(len(self.track_fns)), T=float('inf'))
        for N in N_list:
            subset_name = f'N{N}K'
            subset_gzip[subset_name] = f'{self.base_dir}/tracks_subsets/{self.dataset_domain}-{N}K-RND.jsonl.gzip'
            selected_tracks[subset_name] = [self.track_tid[idx] for idx in sorted_seq[:N*1000]]
        self.save_subsets(subset_gzip, selected_tracks)

    def sample_class_balanced(self, Nc_list, min_nn_dist=0., max_spt_iou=1., iou_tbuff=float('inf'), min_motion=0.):
        if not isinstance(Nc_list, (list, tuple)):
            Nc_list = [Nc_list]

        # Load track index
        if not self.track_fns:
            self.load_tracks()

        # Sample
        sampled_seq = defaultdict(dict)
        for i, cls in enumerate(self.class2idx):
            logits = np.array([self.track_logit[idx] for idx in self.class2idx[cls]])
            for temp in [0.]:   # [0., 0.1, 1., float('inf')]:
                # Stochastic sampling based on class logits
                print(f'Sampling class {i}/{len(self.class2idx)} "{cls}" with logit temperature "{temp}"')
                sorted_seq = self.sort_population(self.class2idx[cls], logits, T=temp)
                sampled_seq[f'Logit-T{temp}'][cls] = self.constrained_sampling(sorted_seq, max(Nc_list),
                    min_nn_dist=min_nn_dist, max_spt_iou=max_spt_iou, iou_tbuff=iou_tbuff, min_motion=min_motion)

        # Prep for saving
        selected_tracks, subset_gzip = {}, {}
        suffix = ''
        if min_nn_dist > 0.:
            suffix += f"MinNNDist{min_nn_dist:0.1f}"
        if max_spt_iou < 1.:
            suffix += f"MaxIoU{max_spt_iou:0.2f}Buf{iou_tbuff:0.2f}"
        if min_motion > 0.:
            suffix += f"MinQ90Motion{min_motion:0.1f}"
        for Nc in Nc_list:
            for temp in [0.]: # [0., 0.1, 1., float('inf')]:
                subset_name = f'Logit-T{temp}-Nc{Nc}'
                selected_tracks[subset_name] = [self.track_tid[idx] for cls in self.class2idx
                                                for idx in sampled_seq[f'Logit-T{temp}'][cls][:Nc]]
                N = len(selected_tracks[subset_name])
                subset_gzip[subset_name] = f'{self.base_dir}/tracks_subsets/{self.dataset_domain}-CB{Nc}-{N//1000}K-T{temp}{suffix}.jsonl.gzip'

        self.save_subsets(subset_gzip, selected_tracks)

    def minimum_redundancy_curator(self, all_idx, Nc, cls, epochs=5):
        print("=" * 30)
        print(f"MinRed({cls}, Nc={Nc}) | len(tids)={len(all_idx)}")

        if len(all_idx) <= Nc:
            return all_idx, cls

        subset_list = random.sample(all_idx, Nc)
        subset_set = set(subset_list)
        remaining_idx = [tid for tid in all_idx if tid not in subset_set]
        subset_embs = torch.stack([self.get_emb(tid) for tid in subset_list]).to(self.device)

        subset_dist = torch.cdist(subset_embs, subset_embs) * (1-torch.eye(Nc).to(self.device))
        subset_nndist, subset_nn = torch.topk(subset_dist, k=2, dim=1, sorted=True, largest=False)
        subset_nndist, subset_nn = subset_nndist[:, 1], subset_nn[:, 1]
        swaps = 0
        for ep in range(epochs):
            converged = True
            remaining_fns = [self.track_fns[tid] for tid in remaining_idx]
            db = VisualMetricsDB(self.base_dir, self.dataset_domain, remaining_fns, remaining_idx, return_embs=True)
            loader = DataLoader(db, num_workers=self.num_workers, shuffle=True, batch_size=1)
            for it, data in enumerate(tqdm.tqdm(loader)):
                idx = data['idx'][0].item()
                candidate_tid = data['tid'][0].item()
                candidate_emb = data['embedding'][0].to(self.device)

                # candidate_tid = remaining_idx[idx]
                candidate_dist = torch.cdist(subset_embs, candidate_emb[None])[:, 0]
                candidate_nndist = candidate_dist.min()
                if candidate_nndist > subset_nndist.min():
                    converged = False
                    remove_idx1 = subset_nndist.argmin().item()
                    remove_idx2 = subset_nn[remove_idx1].item()
                    remove_idx = random.sample([remove_idx1, remove_idx2], 1)[0]
                    remove_tid = subset_list[remove_idx]

                    # Swap candidate
                    subset_list[remove_idx] = candidate_tid
                    subset_embs[remove_idx] = candidate_emb
                    subset_set.remove(remove_tid)
                    subset_set.add(candidate_tid)
                    remaining_idx[idx] = remove_tid

                    # Update distances
                    subset_dist[remove_idx, :] = candidate_dist
                    subset_dist[:, remove_idx] = candidate_dist
                    subset_dist[remove_idx, remove_idx] = 0.
                    subset_nndist, subset_nn = torch.topk(subset_dist, k=2, dim=1, sorted=True, largest=False)
                    subset_nndist, subset_nn = subset_nndist[:, 1], subset_nn[:, 1]

                    # Log
                    swaps += 1
                if it % 10000 == 0:
                    print(f'MinRed({cls}, Nc={Nc})   | Epoch {ep + 1} | Swaps {swaps}. Dist',
                          subset_dist[subset_dist > 0].min().item(), flush=True)
            if converged:
                break
        return subset_list, cls

    def sample_min_red(self, Nc_list):
        if not isinstance(Nc_list, (list, tuple)):
            Nc_list = [Nc_list]

        # Load track index
        if not self.track_fns:
            self.load_tracks()

        # Create subsets of varying size
        selected_tracks, subset_gzip = defaultdict(list), defaultdict(str)
        for Nc in Nc_list:
            subset_name = f'MinRed-Nc{Nc}'
            selected_tracks[subset_name] = []
            for cls in self.class2idx:
                minred_idx, cls = self.minimum_redundancy_curator(self.class2idx[cls], Nc, cls)
                selected_tracks[subset_name] += [self.track_tid[idx] for idx in minred_idx]
            N = len(selected_tracks[subset_name])
            subset_gzip[subset_name] = f'{self.base_dir}/tracks_subsets/{self.dataset_domain}-MinRed-CB{Nc}-{N//1000}K.jsonl.gzip'
        self.save_subsets(subset_gzip, selected_tracks)

    def sample_max_motion(self, Nc_list):
        if not isinstance(Nc_list, (list, tuple)):
            Nc_list = [Nc_list]

        # Load track index
        if not self.track_fns:
            self.load_tracks()

        selected_tracks, subset_gzip = defaultdict(list), defaultdict(str)
        for i, cls in enumerate(self.class2idx):
            print(f"({i}/{len(self.class2idx)}) Loading motion for class {cls}")
            track_fns = [self.track_fns[i] for i in self.class2idx[cls]]
            db = VisualMetricsDB(self.base_dir, self.dataset_domain, track_fns=track_fns, track_ids=self.class2idx[cls], return_motion_stats=True)
            loader = DataLoader(db, num_workers=self.num_workers, batch_size=64)
            track_idx, motion_q90 = [], []
            for dt in tqdm.tqdm(loader):
                track_idx.append(dt['tid']), motion_q90.append(dt['motion']['q90'])
            track_idx, motion_q90 = torch.cat(track_idx), torch.cat(motion_q90)

            max_motion_idx = [track_idx[idx] for idx in (-motion_q90).argsort()]
            for Nc in Nc_list:
                subset_name = f'MaxQ90Motion-Nc{Nc}'
                selected_tracks[subset_name] += [self.track_tid[idx] for idx in max_motion_idx[:Nc]]

        for Nc in Nc_list:
            subset_name = f'MaxQ90Motion-Nc{Nc}'
            N = len(selected_tracks[subset_name])
            subset_gzip[subset_name] = f'{self.base_dir}/tracks_subsets/{self.dataset_domain}-MaxQ90Motion-CB{Nc}-{N//1000}K.jsonl.gzip'

        self.save_subsets(subset_gzip, selected_tracks)


from torch.utils.data import Dataset, DataLoader
class VisualMetricsDB(Dataset):
    def __init__(self, base_dir, dataset_domain, track_fns, track_ids, return_embs=False, return_motion_stats=False):
        self.base_dir = base_dir
        self.dataset_domain = dataset_domain
        self.track_fns = track_fns
        self.track_ids = track_ids
        self.return_embs = return_embs
        self.return_motion_stats = return_motion_stats

    def __len__(self):
        return len(self.track_fns)

    def load_embeddings(self, idx):
        fn = f"{self.base_dir}/tracks_embeddings/{self.dataset_domain}/{self.track_fns[idx].replace('.mp4', '.npy')}"
        try:
            return np.load(fn)
        except Exception:
            return np.randn(2048)

    def load_motion(self, idx):
        try:
            flow_stats_fn = f"{self.base_dir}/tracks_motion/{self.dataset_domain}/{self.track_fns[idx].replace('.mp4', '.npy')}"
            flow_stats = np.load(flow_stats_fn).reshape(-1, 7).mean(0)
            return {'q50': flow_stats[-4], 'q75': flow_stats[-3], 'q90': flow_stats[-2]}
        except Exception:
            return {'q50': 0., 'q75': 0., 'q90': 0.}

    def __getitem__(self, idx):
        output = {'idx': idx, 'tid': self.track_ids[idx]}
        if self.return_embs:
            output['embedding'] = self.load_embeddings(idx)
        if self.return_motion_stats:
            output['motion'] = self.load_motion(idx)
        return output


class Launcher:
    def __call__(self, args):
        for k in args.__dict__:
            print(f"{k}: {args.__dict__[k]}")
        curator = Curator(args.base_dir, args.dataset_domain, args.index_file, num_workers=args.num_workers)

        if args.action == 'index':
            curator.index_db()
        elif args.action == 'sample_random':
            curator.sample_random(args.N)
        elif args.action == 'sample_class_balanced':
            curator.sample_class_balanced(args.Nc, min_nn_dist=float(args.min_nn_dist), max_spt_iou=float(args.max_spt_iou), iou_tbuff=float(args.iou_tbuff), min_motion=float(args.min_motion))
        elif args.action == 'sample_min_red':
            curator.sample_min_red(args.Nc)
        elif args.action == 'sample_max_motion':
            curator.sample_max_motion(args.Nc)


if __name__ == '__main__':
    args = parse_arguments()

    if args.slurm:
        job_names = os.popen('squeue -o %j -u $USER').read().split("\n")
        slurm_job_name = f"{args.action}-NN{args.min_nn_dist}-Motion{args.min_motion}-IoU{args.max_spt_iou}Buf{args.iou_tbuff}"
        if slurm_job_name in job_names:
            print(f'Skipping {slurm_job_name} because already in queue', flush=True)
            exit(0)

        # Submit jobs
        executor = submitit.AutoExecutor(folder='./slurm_logs/', slurm_max_num_timeout=20, cluster=None)
        executor.update_parameters(
            timeout_min=1440,                # Requeue every day
            slurm_partition=args.partition,
            cpus_per_task=args.num_workers,
            gpus_per_node=1,
            nodes=1,
            tasks_per_node=1,
            mem_gb=32,
            slurm_additional_parameters={"exclude": "euler01,euler03,euler05,euler09,euler27"},
            slurm_signal_delay_s=20)
        executor.update_parameters(name=slurm_job_name)
        executor.submit(Launcher(), args)
        print(f"Job submitted: {slurm_job_name}", flush=True)
    else:
        Launcher()(args)
