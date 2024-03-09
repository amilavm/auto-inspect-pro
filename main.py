from unittest import result

from damage_detection.damage_detection import get_damage_prediction
from part_detection.part_detection import get_part_detection
from partwise_damage_assessment.calculate_partwise_damage_percentage import get_final_assessment
from utils.base_utils import create_directory, get_response_image

from utils.detect_model import DetectAndSegment
import os
from pathlib import Path
from app import app
from werkzeug.utils import secure_filename
from flask import flash, request
import json
import time

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Model Configuration
damage_config = {
	'config': app.config['DAMAGE_MODEL_ARTEFACTS_PATH'] + 'mask_rcnn_X_101_32x8d_FPN_3x_config.yml', 
	'weights': app.config['DAMAGE_MODEL_ARTEFACTS_PATH'] + 'model_final.pth', 
	'threshold': 0.5, 
	'device': 'cpu', 
	'num_classes': 4
	}

part_config = {
	'config': app.config['PART_MODEL_ARTEFACTS_PATH'] + 'mask_rcnn_X_101_32x8d_FPN_3x_config.yml',
	'weights': app.config['PART_MODEL_ARTEFACTS_PATH'] + 'model_final.pth', 
	'threshold': 0.5, 
	'device':'cpu', 
	'num_classes': 26
	}

# Model Declaration
damage_model = DetectAndSegment(
	damage_config['config'], damage_config['weights'], damage_config['threshold'], damage_config['device'], damage_config['num_classes']
	)

part_model = DetectAndSegment(
	part_config['config'], part_config['weights'], part_config['threshold'], part_config['device'], part_config['num_classes']
	)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# API Endpoint for Index
@app.route('/')
def index():
	"""
	This is the index endpoint.
	---
	tags:  [index]	
	produces:  [application/json]
	responses:
		200:  {'message': 'Welcome to the Damage Detection API'}

	"""
	return 'Welcome! to the EngenFix AI Engine',200

# API Endpoint for Damage Assessment
@app.route('/assessment', methods=['POST'])
def make_assessment_report():
	"""
	This is the assessment endpoint.
	---
	tags:  [assessment]	
	produces:
        - application/json
	parameters:
		- name: files
		  in: formData
		  type: file
		  required: true
		  description: Image to be assessed
		- name: reg_num
		  in: formData
		  type: string
		  required: true
		  description: Registration Number of the vehicle
	responses:
		200:
			description: Successful response
		400: 
			description: Bad Request
		500: 
			description: Internal Server Error
	"""
	#### read form input data ####
	if request.method == 'POST':
		if (request.form):
			# owner_name = str(request.form['owner_name'])
			# print(owner_name)
			# dl_num = str(request.form['dl_num'])
			reg_num = str(request.form['reg_num'])
			# policy_id = str(request.form['policy_id'])

	if 'files' not in request.files:
		flash('No file Selected')
		# return redirect(request.url)
		return "No files selected!", 400

	# start_globle = time.time()
	files = request.files.getlist('files')
	file_names = []
	for file in files:
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file_names.append(filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			# v_type, v_orient = image_prep(file)
		else:
			flash('Allowed image types are either png, jpg, jpeg')
			# return redirect(request.url)
			return "Allowed image types are either png, jpg, jpeg", 400
			# prediction = 0
			
	# print(file_names)
	
	#### create new claim file path ####
	unique_claim_folder_name = reg_num
	root_claim_path = app.config['CLAIM_FOLDER']
    # New folder Path
	new_claim_path = os.path.join(root_claim_path, unique_claim_folder_name)

    # Create the root directory
	create_directory(root_claim_path, unique_claim_folder_name)

	# start = time.time()
	#### damage detection ####
	dmg_found_files = get_damage_prediction(file_names, new_claim_path, damage_model)
	# print("time for damage recognition process: ", time.time() - start)

	#### part detection ####
	get_part_detection(dmg_found_files, new_claim_path, part_model)
	# print("time for part recognition process: ", time.time() - start)

	#### final assessment ####
	result = get_final_assessment(dmg_found_files, new_claim_path)
	# print("time for part wise assessment process: ", time.time() - start)

	# final_dict = {'assessment': [], 'images': []}
	# final_dict['assessment'].append(result)
	# add damage detected images to display
	out_img_list = []
	for file in dmg_found_files:
		output_image = {}
		file_path = new_claim_path + '/' + Path(file).stem + '/processed_images/' + Path(file).stem + '_predicted_damages.png'
		img_data = get_response_image(file_path)
		output_image['image'] = 'data:image/png;base64,'+ img_data.decode()
		out_img_list.append(output_image)
	

	# final_dict['images'].append(out_img_list)
	final_dict = {'assessment': result, 'images': out_img_list}
	# print("total time taken for the assessment api: ", time.time()-start_globle)

	# return json.dumps(result)
	return json.dumps(final_dict), 200


if __name__ == "__main__":
    app.run()

# flask run