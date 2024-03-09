from statistics import mode
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
from utils.base_utils import create_directory

with open(app.config['DAMAGE_MODEL_ARTEFACTS_PATH'] + 'thing_classes.pkl', 'rb') as f:
        thing_classes = pickle.load(f)


def get_damage_prediction(file_names, claim_files_path, model):
    
    dmg_found_images = []
    for img in file_names:
        filename = Path(img).stem
        im = cv2.imread(app.config['UPLOAD_FOLDER'] + img)
        outputs, instance_idx, v =  model.get_predictions(im)

        if instance_idx: # if damages are found, proceed next steps
            dmg_found_images.append(img)
            # path to the folder for current image
            # (claim_files_path - path to the unique_claim_reg_num folder)
            parent_image_path = claim_files_path + '/' + filename

            #### create root image folder ####
            img_root_folder_name = filename
            # create folder
            create_directory(claim_files_path, filename)

            #### create processed_images folder ####
            processed_img_folder_name = 'processed_images'
            # processed_images folder Path
            processed_images_path = os.path.join(parent_image_path, processed_img_folder_name)
            # create folder
            create_directory(parent_image_path, processed_img_folder_name)

            #### create damages images folder ####
            damage_folder_name = 'damages'
            # damages folder Path
            damaged_images_path = os.path.join(parent_image_path, damage_folder_name)
            # create folder
            create_directory(parent_image_path, damage_folder_name)

            #### create parts images folder ####
            part_folder_name = 'parts'
            # damages folder Path
            part_images_path = os.path.join(parent_image_path, part_folder_name)
            # create folder
            create_directory(parent_image_path, part_folder_name)

            # save the predicted damage instances image
            cv2.imwrite(processed_images_path + '/' + filename + '_predicted_damages.png', v.get_image()[:, :, ::-1])

            dmg_instances = [thing_classes[i] for i in instance_idx]
            uniquify_dmg_instances = [v + str(dmg_instances[:i].count(v) + 1) if dmg_instances.count(v) > 1 else v for i, v in enumerate(dmg_instances)]
            mask_array = outputs['instances'].pred_masks.numpy()
            mask_array = np.moveaxis(mask_array, 0, -1)
            mask_array_instance = []
            output = np.zeros_like(im) #black
            output.fill(255)
            for i in range(len(uniquify_dmg_instances)):
                damage_uname = uniquify_dmg_instances[i]
                print('damage uname:',damage_uname)
                mask_array_instance.append(mask_array[:, :, i:(i+1)])
                # mask_array1 = mask_array[:, :, i:(i+1)]
                # print('mask_array_instance', mask_array_instance)
                output = np.zeros_like(im)
                output.fill(255)
                output = np.where(mask_array_instance[i] == True, 155, output) #change this color to 0 for black in damage images
                # mask_array_instance =[]

                cv2.imwrite(damaged_images_path+ '/' + damage_uname + '.png', output)

            #### get parts detection ####
            # get_part_detection(img, processed_images_path, part_images_path)
        else:
            # if no damages found, just skip this image and continue looking for others
            continue

    return dmg_found_images

# image = 'server/damage_detection/model_artifacts/tr_img_130.jpg'
# get_predict(image)
