from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode,Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np


class Detector:
    def __init__(self, model_type = "OD"):
        self.cfg = get_cfg()
        self.model_type = model_type

        # load model config and pretrained model
        if model_type == "OD": # object detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        elif model_type == "IS": # instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "KP": # keypoint detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "LVIS": # LVIS segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif model_type == "PS": # Panoptic Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")


        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath, saveName):
        image = cv2.imread(imagePath)
        if self.model_type != "PS":
            predictions = self.predictor(image)
            viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                            instance_mode=ColorMode.IMAGE)
            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        else:
            predictions, segmenntInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmenntInfo)

        cv2.imwrite("output/"+saveName + ".jpg", output.get_image())
        # cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.waitKey(0)
    
    def onVideo(self, videoPath,saveName):
        cap = cv2.VideoCapture(videoPath)
        if(cap.isOpened() == False):
            print("Error opening the file ...")
            return
        (success, image) = cap.read()
        i = 0
        while success:
            i += 1
            if self.model_type != "PS":
                predictions = self.predictor(image)
                viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                instance_mode=ColorMode.IMAGE)
                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
            else:
                predictions, segmenntInfo = self.predictor(image)["panoptic_seg"]
                viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmenntInfo)

            # cv2.imshow("Result", output.get_image()[:,:,::-1])
            cv2.imwrite("output/"+saveName + ("%05d" % i) + ".jpg", output.get_image())


            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                break
            (success, image) = cap.read()

