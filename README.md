# TrackVerse: A Large-scale Dataset of Object Tracks

<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-black?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_TrackVerse_A_Large-Scale_Object-Centric_Video_Dataset_for_Image-Level_Representation_Learning_ICCV_2025_paper.pdf">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<!-- <a target="_blank" href="https://tiger-ai-lab.github.io/VLM2Vec/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-red?style=flat"></a> -->
<a target="_blank" href="https://huggingface.co/datasets/yibingwei/TrackVerse/">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>

This repository provides the data, tools, and code to download, explore, and utilize the TrackVerse dataset.

<img src="./doc/figs/banner.gif" alt="trackverse" width="80%"/>

The TrackVerse dataset is a large-scale collection of 31.9 million object tracks, each capturing the motion and appearance of an object over time. These tracks are automatically extracted from YouTube videos using state-of-the-art object detection ([DETIC](https://github.com/facebookresearch/Detic)) and tracking ([ByteTrack](https://github.com/ifzhang/ByteTrack)) algorithms. The dataset spans 1203 object categories from the [LVIS](https://www.lvisdataset.org) ontology, ensuring a diverse and long-tailed distribution of object classes.

TrackVerse is designed to ensure object-centricity, class diversity, and rich object motions and states.  Each track is enriched with metadata, including bounding boxes, timestamps, and prediction labels, making it a valuable resource for research in object-centric representation learning, video analysis, and robotics.

In our paper, we explore the use of TrackVerse for learning unsupervised image representations. By introducing natural temporal augmentations—i.e., viewing an object across time and motion—TrackVerse enables models to learn fine-grained, state-aware representations that are more sensitive to object transformations and behaviors (See paper and for details). 

🎁 Bonus: Our fully automated object track collection pipeline can be easily scaled up without any manual annotation. You can also create your own [customized dataset of object tracks](#create-customized-trackverse-dataset) using different vocabularies, source videos, or curation strategies.

## 🚀 News
- **[Oct 2025]** Our fully automated object track collection pipeline is now publicly released!
- **[July 2025]** TrackVerse dataset and download scripts are now publicly released!
- **[June 2025]** 🎉 Our paper TrackVerse has been accepted to ICCV 2025 🌺

Stay tuned for future updates and improvements!


## Table of Contents
- [Download TrackVerse](#download-trackverse)
- [Create Customized TrackVerse Dataset](#create-customized-trackverse-dataset)
- [Maintenance](#maintenance)
- [License](#license)
- [Citation](#citation)

## Download TrackVerse

TrackVerse is released as a collection of object track metadata stored in **JSONL files**, where each line represents a single track with the following fields:
<details>
<summary>metadata keys</summary>

* `track_id`: Unique ID for the track
* `track_ts`: Start and end timestamps of the track (seconds) in the original video
* `frame_ts`: Timestamps for each frame in the track (seconds) in the original video
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
| 82K-CB100      | 82K    | 100                  | [🤗 Link](https://huggingface.co/datasets/yibingwei/TrackVerse/resolve/main/TrackVerseLVIS-CB100-82K.jsonl.gzip) |
| 184K-CB300     | 184K   | 300                  | [🤗 Link](https://huggingface.co/datasets/yibingwei/TrackVerse/resolve/main/TrackVerseLVIS-CB300-184K.jsonl.gzip) |
| 259K-CB500     | 259K   | 500                  | [🤗 Link](https://huggingface.co/datasets/yibingwei/TrackVerse/resolve/main/TrackVerseLVIS-CB500-259K.jsonl.gzip) |
| 392K-CB1000    | 392K   | 1000                 | [🤗 Link](https://huggingface.co/datasets/yibingwei/TrackVerse/resolve/main/TrackVerseLVIS-CB1000-392K.jsonl.gzip) |
| 1121K-CB2500    | 1.1M   | 2500                 | [🤗 Link](https://huggingface.co/datasets/yibingwei/TrackVerse/resolve/main/TrackVerse-1121K-cls1171CB2500.jsonl.gzip) |
| 3778K-CB8000    | 3.8M   | 8000                 | [🤗 Link](https://huggingface.co/datasets/yibingwei/TrackVerse/resolve/main/TrackVerse-3778K-cls1182CB8000.jsonl.gzip) |
| Validation Set | 4188   | 6                    | [Link](assets/trackverse-verified-6perclass.txt) |



For detailed instructions on extracting TrackVerse from the JSONL files, refer to the [download guide](doc/download.md).


## Create Customized TrackVerse Dataset
You can also create your own customized dataset of object tracks, for example, using different vocabulary, different source videos or different curation strategies.

1) **Set Up the Environment:** Refer to the [install guidelines](doc/env.md) for detailed instructions.
2) **Clone the Repository:** `git clone --recurse-submodules https://github.com/MMPLab/TrackVerse.git`
3) **Follow the Pipeline:** Follow the detailed steps outlined in our [pipeline documentation](doc/pipeline.md).
## Maintenance
For support or inquiries, please open a [GitHub issue](https://github.com/MMPLab/TrackVerse/issues). If you have questions about technical details or need further assistance, feel free to reach out to us directly.

## License
All code and data in this repo are available under the [MIT License](LICENSE) for research purposes only.

## Citation
Please consider giving a star ⭐ and citing our paper if you find this repo useful:
```bib
@InProceedings{Wei_2025_ICCV,
    author    = {Wei, Yibing and Church, Samuel and Suciu, Victor and Lin, Jinhong and Wu, Cheng-En and Morgado, Pedro},
    title     = {TrackVerse: A Large-Scale Object-Centric Video Dataset for Image-Level Representation Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {11153-11163}
}
```