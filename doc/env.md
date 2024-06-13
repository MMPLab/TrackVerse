# Environment
Create conda environment with youtube-dl and ffmpeg 
```bash
conda create -n db -y python=3.8  # python must be 3.8 (won't work with >=3.9 or <3.7) 
conda activate db
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 'Pillow<10' -c pytorch -c conda-forge
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install yt-dlp hydra-core submitit cython_bbox matplotlib tqdm av scenedetect open-clip-torch
cd detic && pip install -r requirements.txt fasttext==0.9.1 && cd ..
```

- Install [pytorch](https://pytorch.org/) and [detectron2](https://detectron2.readthedocs.io/tutorials/install.html)
Note that detectron2 does not work with latest pytorch. We use pytorch==1.10 with cuda==11.3.
- Follow DETIC install [instructions](object_tracks_db/detic/docs/INSTALL.md). 
Default fasttext did not work for us. We used fasttext==0.9.1.
- Follow ByteTrack install [instructions](https://github.com/ifzhang/ByteTrack#readme).
- Download one of the detection models checkpoints and place it under `detic/models/`.

```bash
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```