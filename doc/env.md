# Environment
Create conda environment with youtube-dl, av, pytorch, detectron2, detic and bytetrack.

> [!WARNING]  
> Please follow these instructions carefully, including version numbers of the various packages, 
> to avoid conflicts between pytorch, detectron2 and detic.


> [!NOTE]  
> We are redistributing detic and bytetrack codebases because small modifications had to be made to get
> them to work together.

```bash
conda create -n trackverse -y python=3.8  # python must be 3.8 (won't work with >=3.9 or <3.7) 
conda activate trackverse

# detectron2 does not work with latest pytorch. We use pytorch==1.10 with cuda==11.3.
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Install detic requirements
# Default fasttext did not work for us. We used fasttext==0.9.1.
cd detic && pip install -r requirements.txt fasttext==0.9.1 && cd ..

# Download a DETIC checkpoint and place it under `detic/models/`.
mkdir detic/models && wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

# Install other tools
pip install yt-dlp hydra-core submitit lap cython_bbox matplotlib tqdm av scenedetect jupyter open-clip-torch 'Pillow<10'
```

- Install [pytorch](https://pytorch.org/) and [detectron2](https://detectron2.readthedocs.io/tutorials/install.html)
- Follow DETIC install [instructions](object_tracks_db/detic/docs/INSTALL.md).
- Follow ByteTrack install [instructions](https://github.com/ifzhang/ByteTrack#readme).
