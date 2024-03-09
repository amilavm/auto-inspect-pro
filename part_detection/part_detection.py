import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger #the log of det2 betterthan tf2
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from pathlib import Path

import pickle

from app import app

with open(app.config['PART_MODEL_ARTEFACTS_PATH'] + 'vehicle_parts1.pkl', 'rb') as f:
        thing_classes = pickle.load(f)

def get_part_detection(dmg_found_file_names, claim_folder_path, model):
    
    for image_name in dmg_found_file_names:
        # extension = os.path.splitext(image_name)[1]
        filename = Path(image_name).stem
        im = cv2.imread(app.config['UPLOAD_FOLDER'] + image_name)
        outputs, part_indexes_instances, vp =  model.get_predictions(im)

        # save the predicted part instances image
        cv2.imwrite(claim_folder_path + '/' + filename + '/processed_images/' + filename + '_predicted_parts.png', vp.get_image()[:, :, ::-1])

        part_instances = [thing_classes[i] for i in part_indexes_instances]
        uniquify_part_instances = [v + str(part_instances[:i].count(v) + 1) if part_instances.count(v) > 1 else v for i, v in enumerate(part_instances)]
        mask_array = outputs['instances'].pred_masks.numpy()
        mask_array = np.moveaxis(mask_array, 0, -1)
        mask_array_instance = []
        output = np.zeros_like(im) #black
        output.fill(255)

        for i in range(len(uniquify_part_instances)):
            part_uname = uniquify_part_instances[i]
            print('part name:',part_uname)
            mask_array_instance.append(mask_array[:, :, i:(i+1)])
            # mask_array1 = mask_array[:, :, i:(i+1)]
            # print('mask_array_instance', mask_array_instance)
            output = np.zeros_like(im)
            output.fill(255)
            output = np.where(mask_array_instance[i] == True, 155, output) #change this color to 0 for black in damage images
            # mask_array_instance =[]

            cv2.imwrite(claim_folder_path + '/' + filename + '/parts/' + part_uname + '.png', output)


