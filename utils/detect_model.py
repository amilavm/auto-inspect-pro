# import some common libraries
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from app import app
import cv2

class DetectAndSegment():

    def __init__(self, config_file, model_file, thresh_score, device, num_class):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_file)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_score
        self.cfg.MODEL.WEIGHTS = model_file
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class
        self.cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(self.cfg)


    def get_predictions(self, image):
        outputs = self.predictor(image)
        v = Visualizer(image[:, :, ::-1],
                        MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), 
                        scale=0.8, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )

        vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        indexes_instances = outputs["instances"].pred_classes.tolist()
        return outputs, indexes_instances, vis


