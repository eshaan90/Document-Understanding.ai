import os
import streamlit as st
import configparser
import shutil
from pathlib import Path
#import json

thisfolder = os.path.dirname(os.path.abspath(__file__))
initfile = os.path.join(thisfolder, 'config.ini')

config = configparser.ConfigParser()
config.read(initfile)


DATAPATH=config.get('input', 'DATAPATH')
INPUT_FOLDERNAME=config.get('input', 'INPUT_FOLDERNAME')
IMAGES=config.get('input', 'IMAGES')
SAMPLES_FOLDER=config.get('input', 'SAMPLES_FOLDER')

RESULTS_FOLDERNAME=config.get('output', 'RESULTS_FOLDERNAME')
OUTPUT_JSON_FOLDER_NAME=config.get('output', 'OUTPUT_JSON_FOLDER_NAME')
PAGE_BLOCK_IMAGE_FOLDERNAME=config.get('output', 'PAGE_BLOCK_IMAGE_FOLDERNAME')
TABLES_DET_IMAGE_FOLDERNAME=config.get('output', 'TABLES_DET_IMAGE_FOLDERNAME')
TABLE_BLOBS_FOLDERNAME=config.get('output', 'TABLE_BLOBS_FOLDERNAME')


MODEL_TAG=config.get('model', 'MODEL_TAG')


sample_images_filepath=os.path.join(DATAPATH,SAMPLES_FOLDER)
results_path=os.path.join(DATAPATH,RESULTS_FOLDERNAME)
results_json_filepath=os.path.join(DATAPATH, RESULTS_FOLDERNAME, OUTPUT_JSON_FOLDER_NAME)
json_file_path=os.path.join(results_json_filepath,'table_results.json')

page_blocks_image_filepath=os.path.join(DATAPATH, RESULTS_FOLDERNAME, PAGE_BLOCK_IMAGE_FOLDERNAME)
table_images_filepath=os.path.join(DATAPATH,RESULTS_FOLDERNAME,TABLES_DET_IMAGE_FOLDERNAME)
table_blobs_filepath=os.path.join(DATAPATH, RESULTS_FOLDERNAME, TABLE_BLOBS_FOLDERNAME)

def make_output_dirs():
    os.makedirs(results_json_filepath, exist_ok=True)
    os.makedirs(page_blocks_image_filepath, exist_ok=True)
    os.makedirs(table_images_filepath, exist_ok=True)
    os.makedirs(table_blobs_filepath, exist_ok=True)

def delete_output_dirs():
    try:
        shutil.rmtree(results_path)
        # st.write('Folder and its content removed')
    except:
        print('Output folders not deleted')