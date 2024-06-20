# Download TrackVerse
Assume the dataset will be downloaded in the following folder.
```bash
export TRACKVERSE_DB=/pth/to/trackverase/folder
```
Here is the expected folder structure for the TrackVerse dataset:
```bash
├── tracks_mp4                <- TrackVerse MP4 files
│   └── TrackVerseLVIS                <- The daataset domain
│       ├── --
│           ├── --0Y9qLxvOM-atomizer-2fb79c5812756f9c961d9f5f0f95b12e.mp4
│           ├── ...
│           └── ...
│       ├── ...
│       └── ...
├── tracks_subsets            <- TrackVerse JSONL files
│   ├── LVIS-4M.jsonl.gzip 
│   ├── LVIS-184K-CB300-T0.0-NoTestVids.jsonl.gzip
│   ├── LVIS-259K-CB500-T0.0-NoTestVids.jsonl.gzip
│   ├── LVIS-392K-CB1000-T0.0-NoTestVids.jsonl.gzip
│   └── LVIS-82K-CB100-T0.0-NoTestVids.jsonl.gzip
└── videos_mp4                <- Original video MP4 files
    ├── --
    │   ├── --0Y9qLxvOM.mp4
    │   ├── ...
    │   └── ...
    ├── ...
    └── ...
```
First, place the downloaded JSONL files into the appropriate directory:
```bash
mv path_to_your_downloaded_jsonl_files/*.jsonl.gzip ${TRACKVERSE_DB}/tracks_subsets/
```

Now, use the following bash script to initiate the parallel download of TrackVerse across multiple nodes using Slurm. 
> [!NOTE]
> Slurm is used to split the work across nodes. Each job uses 10 cpu cores and 1 gpu. 
> It is safe to split the download into as many jobs as the cluster can handle.

```bash
WORLD_SIZE=128
INDEX_FILE="tracks_subsets/TrackVerseLVIS-Full-4M.jsonl.gzip"
DB_DOMAIN="LVIS"
for ((JOB_NO=0; JOB_NO<${WORLD_SIZE}; JOB_NO++)); do
    python download_tracks.py --slurm --remove_video_mp4 \
        --base_dir ${TRACKVERSE_DB} \
        --dataset_domain ${DB_DOMAIN} \
        --db_meta_file ${INDEX_FILE} \
        --world_size ${WORLD_SIZE} \
        --rank ${JOB_NO}
done
```