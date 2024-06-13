import sys
import os
import torch
import av

sys.path.insert(0, '../detic')
sys.path.insert(0, '../detic/third_party/CenterNet2/')

from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detectron2.config import get_cfg

from detic.predictor import MetadataCatalog, get_clip_embeddings, reset_cls_test
from detic.predictor import BUILDIN_METADATA_PATH, BUILDIN_CLASSIFIER
from detic.modeling.text.text_encoder import build_text_encoder
from detectron2.engine.defaults import DefaultPredictor


def setup_detic_cfg(config_file, model_weights, confidence_threshold=0.5, nms_threshold=0, frame_size=None, cpu=False, gpu_id=0):
    cfg = get_cfg()
    if cpu:
        cfg.MODEL.DEVICE = "cpu"
    else:
        cfg.MODEL.DEVICE = f'cuda:{gpu_id}'
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_file)

    opts = [
        'MODEL.WEIGHTS', model_weights,
        'MODEL.MASK_ON', False,
        'MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH', 'detic/datasets/metadata/lvis_v1_train_cat_info.json',
        'MODEL.ROI_HEADS.NMS_THRESH_TEST', nms_threshold,
    ]
    if frame_size is not None:
        opts.extend(['INPUT.MIN_SIZE_TEST', frame_size, 'INPUT.MAX_SIZE_TEST', frame_size])
    cfg.merge_from_list(opts)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    cfg.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP = True
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


class BatchPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def __call__(self, batch):
        # Apply pre-processing to image.
        single_img = batch.ndim == 3
        if single_img:
            batch = batch[None]

        height, width = batch[0].shape[:2]
        transform = self.aug.get_transform(batch[0])
        batch = [transform.apply_image(img) for img in batch]
        batch = [torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) for img in batch]
        inputs = [{"image": img, "height": height, "width": width} for img in batch]
        preds = self.model.inference(inputs)
        return preds[0] if single_img else preds


class DETIC(object):
    def __init__(self, detic_cfg, vocabulary, custom=None):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        if vocabulary == 'custom':
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = custom
            # classifier = get_clip_embeddings(self.metadata.thing_classes)

            text_encoder = build_text_encoder(pretrain=True)
            text_encoder.eval()
            text_encoder = text_encoder.to(detic_cfg.MODEL.DEVICE)
            with torch.no_grad():
                prompts = [f'a {x}' for x in self.metadata.thing_classes]
                classifier = text_encoder(prompts).permute(1, 0)

        else:
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[vocabulary])
            classifier = BUILDIN_CLASSIFIER[vocabulary]

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")

        self.predictor = BatchPredictor(detic_cfg)
        reset_cls_test(self.predictor.model, classifier, num_classes)


def build_detic(class_desc, args, gpu_id=0):
    # build detic model
    model_cfg = 'detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
    model_weights = 'detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'

    dictionary = 'custom'
    cfg = setup_detic_cfg(
        config_file=model_cfg,
        model_weights=model_weights,
        confidence_threshold=args.conf,
        nms_threshold=args.nms,
        frame_size=args.frame_size,
        gpu_id=gpu_id
    )
    return DETIC(cfg, dictionary, custom=class_desc)