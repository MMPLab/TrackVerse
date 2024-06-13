# Dataset creation pipeline
Assume the dataset will be created in the following folder.
```bash
export TRACKVERSE_DB=/pth/to/trackverase/folder
```

The dataset of object tracks was generated as follows:

<p align="center">
  <img src="./figs/pipeline.png" alt="drawing" width="70%"/>
</p>

### Step 1: Download the videos.

You have the option to download a customized set of YouTube videos or videos from other sources that hold valid licenses. 

For our dataset,  we downloaded 64K YouTube videos, sampled from the [HD-VILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m) dataset. Videos were acquired at 720p resolution and original frame rates.
 
Begin by downloading the HD-VILA-100M metadata file from the link below and uncompress it into the `assets/hdvila/` directory:
```plaintext
https://hdvila.blob.core.windows.net/dataset/hdvila100m.zip?sp=r&st=2022-06-28T03:33:11Z&se=2026-01-01T11:33:11Z&spr=https&sv=2021-06-08&sr=b&sig=VaqQkLFDqKinfkaPNs1jJ1EQIYCB%2FUPYiqFqmjWye6Y%3D
```

Next, use the following script to download videos:
```bash
# Each job below uses a single gpu to download and parse a chunk of videos. It is safe to run in parallel as many as the cluster can handle.
# Each job should also auto-reschedule itself, if preempted without finishing (although this sometimes does not work, in which case you need to schedule it again). 

python download_videos.py --workdir data/hdvila-100m --metafile assets/hdvila/hdvila100m_metadata.json
```

### Step 2:  Cut Videos into Scenes Clips
```bash
python cut_videos.py      --workdir data/hdvila-100m --metafile assets/hdvila/hdvila_part0.jsonl
```

### Step 3: Apply Filters
We use two content-aware filters to remove unwanted scenes from the dataset:
- Cartoon Filter
- Low Aesthetics Filter

In order to comply with [GDPR](https://gdpr.eu/what-is-gdpr/), we also try to blur out all faces and license plates appearing in the video using [Deface](https://github.com/ORB-HD/deface)  

To do this for all videos in the dataset:
```
python3 -m pip install deface
```
Then run Deface on all videos using the bash script:
```
chmod a+x gdpr_blur_faces.sh  
./gdpr_blur_faces.sh
```



### Step 4: Parse Objtecs
Parse all object tracks within the selected video segments, using Detic for detection and bytetrack for tracking.
- Creates a folder structure based on the first 2 characters of the youtube-id. 
- Within each folder, it stores two files per video: `[YID]-meta.jsonl.gzip` and `[YID]-progress.json`. 
  - Progress file stores information regarding which segments of the video have been parsed already.
  - Meta file stores all parsed object tracks, including detic class predictions, logits, objectness scores, etc.
```bash
python parse_tracks.py --slurm --output_dir ${TRACKVERSE_DB}/tracks_meta/lvis --clips_metafile ${TRACKVERSE_DB}/mined_segments/lvis/part0.jsonl --vocab lvis
...
python parse_tracks.py --slurm --output_dir ${TRACKVERSE_DB}/tracks_meta/lvis --clips_metafile ${TRACKVERSE_DB}/mined_segments/lvis/part127.jsonl --vocab lvis
```


### Detect BBoxes
```bash
python track_objects.py --workdir data/hdvila-100m --frame-rate 8 --conf 0.2 --track_thresh 0.6 --match_thresh 0.3
```

```bash
TRACKVERSE_DB="path/to/dataset/root"
CRITERION="name_of_curation_type"

python curate_db.py \
--output-file ${TRACKVERSE_DB}/tracks_subsets/in1k_coco_x1000/${SUBSET_NAME}.jsonl.gzip \
--tracks-dir ${TRACKVERSE_DB}/tracks_meta/in1k_coco_x1000/ \
--criterion $CRITERION \
--num-samples $NUM_SAMPLES \
--n-workers 12 \
--batch-size 20 \
--num-samples-per-class 500 \
--max-swaps 2000
```

### Step 5: Curate dataset
- Selects a subset of the parsed object tracks using a variety of curation criteria (eg, class balanced sampling, video balanced sampling, etc).
- Saves the subset in a `jsonl.gzip` file (`--output_file`), one track per line.
- Use `--num-workers` to speed up reading of `--tracks_dir`.
```bash
# Random Selection
python curate_db.py --slurm --output_file ${TRACKVERSE_DB}/tracks_subsets/lvis/random-50k.jsonl.gzip --tracks_dir ${TRACKVERSE_DB}/tracks_meta/lvis --criterion random --num-samples 50000 --num-workers 16
# Video Balanced Selection
python curate_db.py --slurm --output_file ${TRACKVERSE_DB}/tracks_subsets/lvis/video-balanced-x3-50k.jsonl.gzip --tracks_dir ${TRACKVERSE_DB}/tracks_meta/lvis --criterion video_balanced --num-samples-per-video 3 --num-samples 50000 --num-workers 16
# Class Balanced Selection
python curate_db.py --slurm --output_file ${TRACKVERSE_DB}/tracks_subsets/lvis/class-balanced-x50-50k.jsonl.gzip --tracks_dir ${TRACKVERSE_DB}/tracks_meta/lvis --criterion class_balanced --num-samples-per-class 50 --num-samples 50000 --num-workers 16
```
