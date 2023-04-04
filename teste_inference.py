#!/usr/bin/env python3
import numpy as np
import torch
import cv2

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
import glob
import time

img_path = "./fig2.jpg"
threshold = 0.45

cfg = LazyConfig.load("./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py")
# cfg = LazyConfig.load("./projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_b_100ep.py")

# edit the config to utilize common Batch Norm
cfg.model.backbone.norm = "BN"
cfg.model.roi_heads.num_classes = 2
cfg.train.device = "cuda"

model = instantiate(cfg.model)

# DetectionCheckpointer(model).load("./model_final_61ccd1.pkl")  # load a file, usually from cfg.MODEL.WEIGHTS
model.to(cfg.train.device)
model = create_ddp_model(model)
DetectionCheckpointer(model).load("./model_final_mask_rcnn.pth")  # load a file, usually from cfg.MODEL.WEIGHTS
# DetectionCheckpointer(model).load("./model_final_cascade.pth")  # load a file, usually from cfg.MODEL.WEIGHTS
register_coco_instances("ball_robot", {},"./Telstar_Mechta/train/_annotations.coco.json", "./Telstar_Mechta/train")
MetadataCatalog.get("ball_robot").thing_classes = ['ball', 'robot']

# read image for inference input
# use PIL, to be consistent with evaluation
img = cv2.imread(img_path)
#img = cv2.resize(img, (1024, 1024))
img = torch.from_numpy(np.ascontiguousarray(img))
img = img.permute(2, 0, 1)  # HWC -> CHW
if torch.cuda.is_available():
    img = img.cuda()
inputs = [{"image": img}]

# run the model
model.eval()
with torch.no_grad():
    start = time.time()
    predictions_ls = model(inputs)
    print(f'Time: {time.time() - start}')
predictions = predictions_ls[0]

print(predictions)
im = cv2.imread(img_path)
#im = cv2.resize(im, (1024, 1024))
color = (255, 0, 0)

indices = predictions['instances'].get_fields()['pred_classes'].to("cpu").numpy()
classes = MetadataCatalog.get("ball_robot").thing_classes
labels = [classes[idx] for idx in indices] 
print(labels)

print(predictions)

for i in range(len(predictions["instances"])):
	#print(predictions["instances"][i])
	#print(predictions["instances"][0].get("pred_boxes"))
	score = predictions["instances"][i].get("scores").item()
	if (score>threshold):
		box = predictions["instances"][i].get("pred_boxes")
		box_tensor = box.tensor[0]
		x_inicial = box_tensor.data[0].item()
		y_inicial = box_tensor.data[1].item()
		x_final = box_tensor.data[2].item()
		y_final = box_tensor.data[3].item()
		start = (int(x_inicial), int(y_inicial))
		final = (int(x_final), int(y_final))
		cv2.rectangle(im, start, final, (255, 0, 0), 1)
		im = cv2.putText(im, labels[i], (int(x_inicial), int(y_inicial)-4), cv2.FONT_HERSHEY_SIMPLEX, 
			0.5, (255, 0, 0), 1, cv2.LINE_AA)
		im = cv2.putText(im, str(round(score,3)), (int(x_inicial), int(y_final)+14), cv2.FONT_HERSHEY_SIMPLEX, 
			0.5, (255, 0, 0), 1, cv2.LINE_AA)
cv2.imshow("RoboFEI",im)
cv2.waitKey(0)





