
# AutoInspectPro

A Flask based Server Application for **Vehicle Damage Assessment** using AI

[![](https://img.shields.io/badge/python-3.7-blue.svg)]()
[![](https://img.shields.io/badge/Detectron2-0.6-brightgreen.svg)]()
[![](https://img.shields.io/badge/torch-1.11.0-red.svg)]()
[![](https://img.shields.io/badge/Made_with-Flask-important.svg)]()
[![](https://img.shields.io/badge/Product-AutoInspectPro-1f425f.svg)]()
<!-- [![](https://img.shields.io/badge/Powered_by-Engenuity_Ai-yellow.svg)]() -->


<!-- <br /> -->

## Objectives

    The project shows the AI capabilities of the AutoInspectPro product for Automating the Damage Inspections. This concept can be used for various domains which power the object detection and segmentation capabilities.

## Demo

This is a sample [demo](https://drive.google.com/file/d/1Csj1Q6oQPkw9a0qZLNkifSWSAdj4m0Mi/view?usp=sharing) video of the project.



## ✨ Code-base structure

The project code base structure is as below:

```bash
< PROJECT ROOT >
   |
   |-- assets/
   |    |-- inputs/                                         # Folder to store input images
   |    
   |-- claims/                                              # Folder to keep all the processed files for each claim
   |    
   |    
   |-- damage_detection/                                    # Module for Damage Detection
   |    |-- model_artifacts/                                # Folder containing all the model artefacts for part detection and segmentation
   |    |    |-- mask_rcnn_X_101_32x8d_FPN_3x_config.yml    # Model configuration file
   |    |    |-- model_final.pth                            # Model weights
   |    |    |-- thing_classes.pkl                          # Pickle file specifying all damage classes
   |    |    
   |    |-- __init__.py                                     # Module initialization
   |    |-- damage_detection.py                             # Damage detection operations
   |
   |    
   |-- part_detection/                                      # Module for part Detection
   |    |-- model_artifacts/                                # Folder containing all the model artefacts for damage detection and segmentation
   |    |    |-- mask_rcnn_X_101_32x8d_FPN_3x_config.yml    # Model configuration file
   |    |    |-- model_final.pth                            # Model weights
   |    |    |-- vehicle_parts1.pkl                         # Pickle file specifying all vahicle parts
   |    |    
   |    |-- __init__.py                                     # Module initialization
   |    |-- part_detection.py                               # Part detection operations
   |
   |
   |-- partwise_damage_assessment/                          # Module for assessing part wise damages
   |    |-- __init__.py                                     # Module initialization
   |    |-- calculate_partwise_damage_percentage.py         # Calculate part wise damage assessment
   |
   |
   |-- utils/                                               # Support files
   |    |-- __init__.py                                     # Module initialization
   |    |-- base_utils.py                                   # Basic helping functions
   |    |-- detect_model.py                                 # Model class
   |
   |
   |-- requirements.txt                                     # Requirements & dependencies
   |-- .flaskenv                                            # Flask environment configurations
   |
   |-- app.py                                               # Setup app
   |-- main.py                                              # Start the app - WSGI gateway
   |
   |-- ************************************************************************
```
<!-- <br /> -->

## Setup

1. Set up new virtual environment:

    ```bash
    $ conda create --name <env-name> python=3.7
    $ conda activate <env-name>
    ```

2. Install Requirements: 

    ```bash
    $ pip install -r requirements.txt
    ```

3. To run the app on localhost:

    ```bash
    $ flask run
    ```

4. app will be running at: 
<http://127.0.0.1:5000>
<!-- ```bash 
$ # can also set host and port manually
$ # flask run --host=0.0.0.0 --port=80
``` -->

<br />

<!-- ## ✨ Quick Start with `Docker` -->

<!-- #### Dataset Acquisition:

<pre>

Sample data to test : Chest X-Ray Images (Pneumonia)

</pre> -->
