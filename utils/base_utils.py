import os
import io
import base64
from PIL import Image

def create_directory(parent_dir_path, new_folder_name):
    try:
        os.makedirs(os.path.join(parent_dir_path, new_folder_name), exist_ok = True)
        print("Directory '%s' created successfully" % new_folder_name)
    except OSError as error:
        print("Directory '%s' can not be created" % new_folder_name)

def get_response_image(image_path):
	pil_img = Image.open(image_path, mode='r') # reads the PIL image
	byte_arr = io.BytesIO()
	pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
	byte_arr.seek(0)
	# encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
	encoded_img = base64.b64encode(byte_arr.read())
	return encoded_img