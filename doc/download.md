# Download TrackVerse

### Set Up Environment

Create and activate a Python virtual environment, then install the required dependencies:

```bash
python3 -m venv trackverse
source trackverse/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset Folder Structure

Specify the root directory for storing the dataset:

```bash
export TRACKVERSE_DB=/path/to/trackverse/folder
```

TrackVerse expects the following directory layout:

```bash
├── tracks_mp4/                   # Extracted object track videos
│   └── TrackVerseLVIS/
│       ├── <track_id_1>.mp4
│       └── ...
├── metadata/               # Subset metadata (JSONL)
│   ├── TrackVerse-4M.jsonl.gzip
│   ├── TrackVerse-1121K-cls1171CB2500.jsonl.gzip
│   └── ...
└── videos_mp4/                   # Original YouTube MP4 videos
    ├── <youtube_id_1>.mp4
    └── ...
```

Move the downloaded `.jsonl.gzip` files into the `metadata` directory:

```bash
mv path_to_downloaded_jsonls/*.jsonl.gzip ${TRACKVERSE_DB}/metadata/
```


###  Download Track Videos

You can download object track videos using either **Slurm for parallel execution** or a **single-process fallback**.

#### Option 1: Parallel Download with Slurm

Use the following script to launch parallel jobs across multiple nodes:

```bash
WORLD_SIZE=128
INDEX_FILE="metadata/TrackVerse-4M.jsonl.gzip"

for ((JOB_NO=0; JOB_NO<${WORLD_SIZE}; JOB_NO++)); do
    python download_tracks.py --slurm \
        --base_dir ${TRACKVERSE_DB} \
        --db_meta_file ${INDEX_FILE} \
        --world_size ${WORLD_SIZE} \
        --rank ${JOB_NO}
done
```

> ⚠️ **Note:**
> Each job uses 10 CPU cores and 1 GPU. You can safely split the download into as many jobs as your cluster allows.

#### Option 2: Download Without Slurm (Single Process 🐢 )

For local or small-scale downloads, you can run without Slurm:

```bash
INDEX_FILE="metadata/TrackVerse-4M.jsonl.gzip"
python download_tracks.py --base_dir ${TRACKVERSE_DB} --db_meta_file ${INDEX_FILE}
```

> 🔑 **YouTube Download Tip:**
> YouTube videos may require authentication to download.
> Refer to the [yt-dlp cookie FAQ](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) to export your browser cookies, and use the `--cookiefile` option:
>
> ```bash
> --cookiefile /path/to/your/cookiefile
> ```
