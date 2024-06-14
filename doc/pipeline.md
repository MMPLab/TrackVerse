# Dataset creation pipeline
Assume the dataset will be created in the following folder.
```bash
export TRACKVERSE_DB=/pth/to/trackverase/folder
```

The dataset of object tracks was generated as follows:

<div align="center">
  <img src="./figs/pipeline.png" alt="drawing" width="70%"/>
</div>

<!---------------------------------------------------------------------------------------------->

<details><summary><h3>Step 1-3: Download videos, find and filter segments</h3></summary>

For our dataset, we downloaded 64K YouTube videos, sampled from the [HD-VILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m) dataset, and specified in `assets/trackverse-yids-all.txt`. Videos were acquired at 720p resolution and original frame rates.


> [!NOTE]
> Slurm is used to split the work across nodes. Each job uses 10 cpu cores and 1 gpu. 
> It is safe to split the download into as many jobs as the cluster can handle.

> [!TIP]
> The videos in `assets/trackverse-yids-all.txt` have already been filtered for cartoon 
> and low aesthetic contents. To simply download TrackVerse, these filters should be skipped.
> If you want to extend the dataset, or create a customized version of it, the filters may 
> help obtaining a better data distribution. 

```bash
WORLD_SIZE=128
for ((JOB_NO=0; i<${WORLD_SIZE}; i++)); do
    python download_videos.py --slurm \
        --base_dir ${TRACKVERSE_DB} \
        --yid_index_fn assets/trackverse-yids-all.txt \
        --skip_cartoon_filter --skip_aesthetics_filter \
        --world_size ${WORLD_SIZE} \
        --rank ${JOB_NO}
done
```

This code creates 2 folders under `${TRACKVERSE_DB}`:
- `${TRACKVERSE_DB}/videos_mp4`: Folder with the original videos at 720p resolution.
- `${TRACKVERSE_DB}/videos_segm`: Folder with one text file per video indicating the selected segments (List of start and end timestamps). 
</details>

<!---------------------------------------------------------------------------------------------->

<details><summary><h3>Step 3a: Deface</h3></summary>

To comply with [GDPR](https://gdpr.eu/what-is-gdpr/), we also try to blur out all faces and license plates appearing in the video using [Deface](https://github.com/ORB-HD/deface)  

To do this for all videos in the dataset:
```bash
python3 -m pip install deface
```

Then run Deface on all videos using the bash script:
```bash
chmod a+x gdpr_blur_faces.sh  
./gdpr_blur_faces.sh
```
</details>

<!---------------------------------------------------------------------------------------------->


<details><summary><h3>Step 4: Parse Object Tracks</h3></summary>
Next, we parse all object tracks within each video segment using DETIC and ByteTrack.

> [!NOTE]
> Slurm is used to split the work across nodes. Each job uses 10 cpu cores and 1 gpu. 
It is safe to split the download into as many jobs as the cluster can handle.

> [!TIP]
> The set of object categories is given in `assets/lvis-prompts.txt`. You can specify your own 
set of categories for DETIC to detect by proving a new list of class prompts.

```bash
WORLD_SIZE=128
for ((JOB_NO=0; i<${WORLD_SIZE}; i++)); do
    python parse_tracks.py --slurm \
        --base_dir ${TRACKVERSE_DB} \
        --yid_index_fn assets/trackverse-yids-all.txt \
        --dataset_name TrackVerseLVIS \
        --class_prompts assets/lvis-prompts.txt \
        --world_size ${WORLD_SIZE} \
        --rank ${JOB_NO}
done
```

This code stores all parsed track metadata in the folder `${TRACKVERSE_DB}/tracks_meta`. For each video, it creates a `[YID]-meta.jsonl.gzip` file containing, among others, detic class predictions, and the spatial and temporal coordinates of the track tubelet.
</details>

<!---------------------------------------------------------------------------------------------->

<details><summary><h3>Step 4a: Save tracks as video clips</h3></summary>
It's now time to extract all tracks into mp4 files. To optimize the files for deep learning workloads, we save the video file using a small key-frame rate.

```bash
WORLD_SIZE=128
for ((JOB_NO=0; i<${WORLD_SIZE}; i++)); do
    python extract_tracks.py --slurm \
        --base_dir ${TRACKVERSE_DB} \
        --yid_index_fn assets/trackverse-yids-all.txt \
        --dataset_name TrackVerseLVIS \
        --world_size ${WORLD_SIZE} \
        --rank ${JOB_NO}
done
```

All tracks are saved to `${TRACKVERSE_DB}/tracks_mp4`.

</details>

<!---------------------------------------------------------------------------------------------->

<details><summary><h3>Step 5: Dataset curation</h3></summary>
Finally, we define subsets of object tracks with more balanced class distributions by selecting for each class the `K` samples with the highest classification logits weighted by the objectness score.

>[!NOTE]
> When creating these subsets, we make sure that the sampled tracks do not come from evaluation videos, whose labels have been verified (`assets/trackverse-verified-6perclass.txt`).

>[!NOTE]
> The generated subsets are saved in `${TRACKVERSE_DB}/tracks_subsets/` using a hardcoded naming convention. For example, the generated subset `TrackVerseLVIS-CB1000-392K-T0.jsonl.gzip` means that the data is class-balanced (CB) with at most 1000 samples per class, and contains 392K tracks. The `T0` suffix indicates that the subset was generated deterministically by selecting the samples with highest logits (sampled with 0 temperature). Subsets with `Tinf` suffix are sampled with infinite temperature, which means that the samples are selected at random within each class.

```bash
# First create a single index file containing all track metadata extracted to ${BASE_DIR}/tracks_meta
INDEX_FILE="tracks_subsets/TrackVerseLVIS/TrackVerseLVIS-Full-4M.jsonl.gzip"
python curate_db.py --base_dir ${BASE_DIR} --index_file ${INDEX_FILE} --dataset_name TrackVerseLVIS --action index --num_workers 16   # num_workers speed up reading of the metadata

# Then sample both random and class-balanced subsets
python curate_db.py --base_dir ${BASE_DIR} --index_file ${INDEX_FILE} --action sample_random --N 82 184 259 392
python curate_db.py --base_dir ${BASE_DIR} --index_file ${INDEX_FILE} --action sample_class_balanced --Nc 100 300 500 1000
```

</details>

<!---------------------------------------------------------------------------------------------->

<details><summary><h3>Step 5a: Dataset curation (Bonus)</h3></summary>
We have also implemented motion and diversity-based curation strategies, by ensuring the selected tracks meet certain threshold criteria. To use these strategies, we first need to generate either the motion or feature representations used to encode track appearance.

```bash
# Compute embeddings and optical flow
# Saves tracks under ${BASE_DIR}/tracks_${METRIC}/${DB_NAME}
WORLD_SIZE=128
INDEX_FILE="tracks_subsets/TrackVerseLVIS-Full-4M.jsonl.gzip"
DB_NAME="TrackVerseLVIS"
for ((JOB_NO=0; i<${WORLD_SIZE}; i++)); do
    python visual_metrics.py --slurm \
        --base_dir ${BASE_DIR} \
        --dataset_name ${DB_NAME} \
        --index_file ${INDEX_FILE} \
        --metric motion \
        --world_size ${WORLD_SIZE} \
        --rank ${JOB_NO}
done
for ((JOB_NO=0; i<${WORLD_SIZE}; i++)); do
    python visual_metrics.py --slurm \
        --base_dir ${BASE_DIR} \
        --dataset_name ${DB_NAME} \
        --db_meta_file ${INDEX_FILE} \
        --metric embeddings \
        --world_size ${WORLD_SIZE} \
        --rank ${JOB_NO}
done
```

Then, we can use these metrics to curate the dataset. For example, to sample a class-balanced subset of tracks with a minimum motion of 1.0.

```bash
python curate_db.py --base_dir ${BASE_DIR} --index_file ${INDEX_FILE} --action sample_class_balanced --min_motion 1.
```