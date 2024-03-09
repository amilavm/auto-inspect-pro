from flask import Flask
# from flasgger import Swagger

UPLOAD_FOLDER = 'assets/inputs/'
CLAIM_FOLDER = 'claims/'
DAMAGE_DETECTION_MODULE_PATH = 'damage_detection/'
PART_DETECTION_MODULE_PATH = 'part_detection/'

DAMAGE_MODEL_ARTEFACTS_PATH = 'damage_detection/model_artifacts/'
PART_MODEL_ARTEFACTS_PATH = 'part_detection/model_artifacts/'

app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CLAIM_FOLDER'] = CLAIM_FOLDER
app.config['DAMAGE_DETECTION_MODULE_PATH'] = DAMAGE_DETECTION_MODULE_PATH
app.config['PART_DETECTION_MODULE_PATH'] = PART_DETECTION_MODULE_PATH

app.config['DAMAGE_MODEL_ARTEFACTS_PATH'] = DAMAGE_MODEL_ARTEFACTS_PATH
app.config['PART_MODEL_ARTEFACTS_PATH'] = PART_MODEL_ARTEFACTS_PATH

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
