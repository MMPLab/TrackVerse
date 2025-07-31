# TrackVerse: A Large-scale Dataset of Object Tracks

<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-black?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://github.com/MMPLab/TrackVerse">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<!-- <a target="_blank" href="https://tiger-ai-lab.github.io/VLM2Vec/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-red?style=flat"></a> -->
<a target="_blank" href="https://huggingface.co/datasets/yibingwei/TrackVerse/">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<br>

This repository provides the data, tools, and code to download, explore, and utilize the TrackVerse dataset.

<img src="./doc/figs/banner.gif" alt="trackverse" width="80%"/>

The TrackVerse dataset is a large-scale collection of 31.9 million object tracks, each capturing the motion and appearance of an object over time. These tracks are automatically extracted from YouTube videos using state-of-the-art object detection ([DETIC](https://github.com/facebookresearch/Detic)) and tracking ([ByteTrack](https://github.com/ifzhang/ByteTrack)) algorithms. The dataset spans 1203 object categories from the [LVIS](https://www.lvisdataset.org) ontology, ensuring a diverse and long-tailed distribution of object classes.

TrackVerse is designed to ensure object-centricity, class diversity, and rich object motions and states.  Each track is enriched with metadata, including bounding boxes, timestamps, and prediction labels, making it a valuable resource for research in object-centric representation learning, video analysis, and robotics.

In our paper, we explore the use of TrackVerse for learning unsupervised image representations. By introducing natural temporal augmentations—i.e., viewing an object across time and motion—TrackVerse enables models to learn fine-grained, state-aware representations that are more sensitive to object transformations and behaviors (See paper and [Variance-Aware Contrastive Learning](#variance-aware-contrastive-learning) for details).

## 🚀 News
- **[July 2025]** TrackVerse dataset and download scripts are now publicly released!
- **[June 2025]** 🎉 Our paper TrackVerse has been accepted to ICCV 2025 🌺

Stay tuned for future updates and improvements!


## Table of Content
- [Download TrackVerse](#download-trackverse)
- [Variance-Aware Contrastive Learning](#variance-aware-contrastive-learning)
- [Maintenance](#maintenance)
- [License](#license)

## Download TrackVerse

TrackVerse is released as a collection of object track metadata stored in **JSONL files**, where each line represents a single track with the following fields:
<details>
<summary>metadata keys</summary>

* `track_id`: Unique ID for the track
* `track_ts`: Start and end timestamps of the track
* `frame_ts`: Timestamps for each frame in the track
* `frame_bboxes`: Bounding boxes `[x, y, width, height]` for each frame
* `yid`: YouTube video ID
* `track_mp4_filename`: Local filename of the track video
* `top10_label_ids`: Top-10 predicted class IDs
* `top10_label_names`: Top-10 predicted class names
</details>

To support diverse research needs, we provide the full TrackVerse dataset, curated subsets at various scales to ensure more balanced class distributions, and a human-verified validation set for in-domain evaluation:

| Subset        | #Tracks | Max Tracks per Class | Link |
|---------------|---------|----------------------|------|
| Full TrackVerse | 31.9M   | ---                  | Coming soon. |
| 82K-CB100      | 82K    | 100                  | [Google Drive](https://drive.google.com/file/d/181WNhqewLnj-Ais3rL7cYoIwzowbUob4/view?usp=drive_link) |
| 184K-CB300     | 184K   | 300                  | [Google Drive](https://drive.google.com/file/d/1410JsoHwsY8eiFvpfWk8M0yYFh7EC62I/view?usp=drive_link) |
| 259K-CB500     | 259K   | 500                  | [Google Drive](https://drive.google.com/file/d/16jM3_IoSD59k33LfhDm87r0oip7W3ZNw/view?usp=drive_link) |
| 392K-CB1000    | 392K   | 1000                 | [Google Drive](https://drive.google.com/file/d/1qT2HvJumzdcNqMapP8SJZ18j9D1ABDwj/view?usp=drive_link) |
| 1121K-CB2500    | 1.1M   | 2500                 | [🤗 Link](https://huggingface.co/datasets/yibingwei/TrackVerse/resolve/main/TrackVerse-1121K-cls1171CB2500.jsonl.gzip) |
| 3778K-CB8000    | 3.8M   | 8000                 | [🤗 Link](https://huggingface.co/datasets/yibingwei/TrackVerse/resolve/main/TrackVerse-3778K-cls1182CB8000.jsonl.gzip) |
| Validation Set | 4188   | 6                    | [Link](assets/trackverse-verified-6perclass.txt) |


For detailed instructions on extracting TrackVerse from the JSONL files, refer to the [download guide](doc/download.md).

## Variance-Aware Contrastive Learning
Coming soon.

## Maintenance
For support or inquiries, please open a [GitHub issue](https://github.com/MMPLab/TrackVerse/issues). If you have questions about technical details or need further assistance, feel free to reach out to us directly.

## License
All code and data in this repo are available under the [MIT License](LICENSE) for research purposes only.
