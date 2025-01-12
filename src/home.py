import streamlit as st
import os 
import time
import math
import copy
import json
import shutil
from collections import defaultdict
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
import pypdfium2 as pdfium
import cv2
import random
import numpy as np

#local
from process_image import (grayscale, noise_removal, remove_borders,
                            getSkewAngle, rotateImage, display_image,deskew,
                            image_converter_openCV_to_pil, image_converter_pil_to_openCV,
                            table_recognition_from_images)
from load_config import (MODEL_TAG, results_json_filepath, layouts_image_filepath,
                         layout_result_img_path,
                        page_blocks_image_filepath, json_file_path,
                        table_images_filepath, table_blobs_filepath, 
                        make_output_dirs, delete_output_dirs,sample_images_filepath)
from ocr import get_image_countours, get_text_from_image
from utils import save_to_file, check_if_any_file_exists, paginator
from schema import COLOR_MAP, text_labels
from models import load_models, get_text_detections, get_layout_detection
from PIL import Image, ImageDraw


st.set_page_config(page_title='Home',layout="wide")

st.title('Machine Document Understanding')
# st.write('Visualize the extraction process of layout and text from PDF documents using AI tools')
placeholder = st.empty()

extraction_completed=False

# Declare variable.
if 'pdf_ref' not in ss:
    ss.pdf_ref = None



st.sidebar.title('1. Data')
data_container=st.sidebar.expander('Data',expanded=True)
# Access the uploaded ref via a key.
data_container.file_uploader('Upload PDF file', type=['pdf'], key='pdf', accept_multiple_files=False)

# st.write(f'pdf:{ss.pdf},\npdf_ref:{ss.pdf_ref}')
if not ss.pdf:
    placeholder.info('Visualize the process of extracting layout and text from PDF documents using AI \n\nLoad a pdf from the sidebar. The below containers will populate with the loaded file')
    ss.pdf_ref=None
    delete_output_dirs()
    if 'img_path_mapper' in ss:
        del ss['img_path_mapper']
else:
    ss.pdf_ref = ss.pdf  # backup
    if 'img_path_mapper' not in ss:
        ss.img_path_mapper= {
                            'blocks':{},
                            'table_blobs':{}
                            }
# st.write(f'pdf:{ss.pdf},\npdf_ref:{ss.pdf_ref}')


mainTab1, mainTab2, mainTab3 = st.tabs(["Document Analysis", "OCR Analysis", "Samples"])
with mainTab1:
    col1,col2=st.columns(2,border=True)

    col1.header('A. PDF File')
    col1_placeholder=col1.empty()
    col1_placeholder.info('Load a PDF file from the sidebar to visualize here.')
    
    col2.header('B. Selected Image')
    col2_placeholder=col2.empty()
    col2_placeholder.info('Image of the page selected from PDF file is displayed here.')

    if ss.pdf_ref:
        col1_placeholder.empty()
        col2_placeholder.empty()
        make_output_dirs()

        pdf = pdfium.PdfDocument(ss.pdf_ref)
        input_filename=ss.pdf_ref.name
        n_pages = len(pdf)  # get the number of pages in the document
        page_number=data_container.number_input('Select page number to load', min_value=0, max_value=n_pages-1, value=0, step=1, format="%d")
        
        page = pdf[page_number]  # load a page
        bitmap = page.render(
            scale = 1,    # 72dpi resolution
            rotation = 0, # no additional rotation
            # ... further rendering options
        )
        img = bitmap.to_pil()
        placeholder.info(f'Document loaded successfully!')

        
        with col1:
            st.write(f'Contains {n_pages} pages')
            binary_data = ss.pdf_ref.getvalue()
            pdf_viewer(input=binary_data, scroll_to_page=page_number, 
                        width=900,height=700,
                        pages_vertical_spacing=3)
        
        col2.write(f'Page {page_number} selected')
        col2.image(img, use_container_width=True)

        # col3.write('I am column 3')    


    col3,col4=st.columns(2, border=True)
    col3.header('C. Visualize the Processed Image')
    col3_placeholder=col3.empty()
    col3_placeholder.info('Processed Image is displayed here. \n\n Load a document from the sidebar and process the image using the sidebar tools.')
    
    col4.header('D.Output(Detection & Recognition)')
    col4_placeholder=col4.empty()
    col4_placeholder.info('Image with layout data is presented here. \n\n Load a document and select Extract from the sidebar.')
    
    st.sidebar.title('2. Pre-process Image (Optional)')
    run_preprocessing=st.sidebar.checkbox('Process Image')
    # Logic for Processing Images
    if ss.pdf_ref and run_preprocessing:
        col3_placeholder.empty()
        with st.sidebar.expander('Tools',expanded=True):
            run_binarization = st.checkbox('Binarization')
            if run_binarization:
                thresh = st.number_input('Threshold for binarization',value=230,format='%d')
                max_val = st.number_input('Max value for binarization',value=255,format='%d')
            st.divider()
            run_noise_removal = st.checkbox('Noise Removal')
            if run_noise_removal:
                dilation_kernel_size = st.number_input('Dilation Kernel Size (1-10)',value=1,format='%d')
                erosion_kernel_size = st.number_input('Erosion Kernel Size (1-10)',value=1,format='%d')
                iterations = st.number_input('Number of Iterations for noise removal',value=1,format='%d')
            st.divider()
            run_remove_borders = st.checkbox('Remove Borders')
            st.divider()
            run_deskew = st.checkbox('Deskew/Rotate image')
            # st.divider()
            # run_dilate=st.checkbox('Dilate')
            # st.divider()
            # run_erosion=st.checkbox('Erosion')

        #     process = st.button('Process', key='1b')


        # if process:
        #     # Code to process the image

        cv_img = image_converter_pil_to_openCV(img)
        if run_binarization:
            cv_img = grayscale(cv_img)
            thresh, cv_img = cv2.threshold(cv_img, thresh, max_val, cv2.THRESH_BINARY)

        if run_noise_removal:
            cv_img = noise_removal(cv_img,erosion_kernel_size,dilation_kernel_size,iterations)

        if run_remove_borders: 
            cv_img = remove_borders(cv_img)

        if run_deskew:
            cv_img = deskew(cv_img)

        col3_placeholder.empty()
        col3.image(cv_img, use_container_width=True)


    st.sidebar.title('3. Layout and Text Extraction')
    detect_tables_checkbox = st.sidebar.checkbox('Detect Table Structure')
    run_extraction = st.sidebar.button('Extract')

    #Logic to extract layout and text from Images
    if ss.pdf_ref and run_extraction:
        with st.spinner('Loading Models'):
            layout_model,layout_processor, det_model, det_processor, table_rec_model, tables_processor = load_models(model_tag=MODEL_TAG)
        placeholder.success('Models loaded!')

        if run_preprocessing:
            pil_image=image_converter_openCV_to_pil(cv_img)
            input_image=pil_image.copy()
        else:
            input_image=img.copy()
        input_image_copy=input_image.copy()

        with st.spinner('Detecting Layout and Text'):
            start_time=time.time()
            line_predictions=get_text_detections([input_image],det_model, det_processor)
            layout_predictions=get_layout_detection([input_image],layout_model, layout_processor, line_predictions)        
            end_time=time.time()
        placeholder.success('Inference Completed!')

        draw = ImageDraw.Draw(input_image_copy)
        for bbox in layout_predictions[0].bboxes:
            draw.rectangle(tuple(bbox.bbox), outline=COLOR_MAP[bbox.label], width=2)
            draw.text([bbox.polygon[0][0],bbox.polygon[0][1]-10],text=f'{bbox.position}:{bbox.label}',fill='green')#, font=ImageFont.truetype("font_path123"))

        col4_placeholder.empty()
        col4.image(input_image_copy, caption=f'Took {math.ceil(end_time-start_time)} secs', use_container_width=True)
        # input_image_copy.save(os.path.join(sample_images_filepath,f'{input_filename}_result.png'))
        input_image_copy.save(layout_result_img_path)

        structure={}
        if detect_tables_checkbox:
            with st.spinner('Detecting Table Structure'):
                try:
                    table_predictions=table_recognition_from_images(input_filename, input_image, layout_predictions, table_images_filepath,model_tag='surya')
                    structure['table_data']=table_predictions
                    # save_to_file(table_predictions, json_file_path, output_file_type='json')
                except Exception as e:
                    st.write(f'Error encountered while table recognition: {e}')
            placeholder.success('Table Structure Detection Complete!')

        
        cv_img=image_converter_pil_to_openCV(input_image)

        for bbox in layout_predictions[0].bboxes:
            if bbox.label not in structure:
                structure[bbox.label]={}
        
            structure[bbox.label][bbox.position]={}
            structure[bbox.label][bbox.position]['bbox']=bbox.bbox
            structure[bbox.label][bbox.position]['polygon']=bbox.polygon
            structure[bbox.label][bbox.position]['confidence']=bbox.confidence

            bounding_box = (int(bbox.polygon[0][0]), int(bbox.polygon[0][1]), int(bbox.width), int(bbox.height))
            image_box = get_image_countours(cv_img,bounding_box)
            block_filename=f'{bbox.position}:{bbox.label}.png'
            block_filepath=os.path.join(page_blocks_image_filepath, block_filename)
            cv2.imwrite(block_filepath,image_box)

            ocr_result=None
            #extract text if its a text label
            if bbox.label in text_labels:
                ocr_result = get_text_from_image(image_box)
                # if not ocr_result or not ocr_result.isalnum():
                #     ocr_result=None
                structure[bbox.label][bbox.position]['text']=ocr_result
            
            ss['img_path_mapper']['blocks'][block_filename]=(block_filepath,ocr_result)
        
        
        extraction_completed=True
        st.write('Here is the page structure:')
        page_structure=st.container(height=600,border=True)
        page_structure.write(structure)

        save_to_file(structure, json_file_path, output_file_type='json')

with mainTab2:
    col11,col12 = st.columns(2,border=True)

    col11.header('Layout Detection')
    col12.header('Visualize Blocks')
    col11_placeholder=col11.empty()
    col12_placeholder=col12.empty()

    col11_placeholder.info('Upload and extract a page to visualize results here')
    col12_placeholder.info('Upload and extract a page to visualize individual block OCR results here')

    if os.path.exists(layout_result_img_path):
        col11_placeholder.image(layout_result_img_path, use_container_width=True)

    if os.path.isdir(page_blocks_image_filepath):
        
        # image_iterator = paginator("Select page", 
        #                             img_blocks_locmapper['blocks'], 
        #                             items_per_page=5,
        #                             on_sidebar=False)
        # indices_on_page, images_with_captions = map(list, zip(*image_iterator))
        # images_on_page, captions=zip(*images_with_captions)
        # st.write(len(images_on_page)==len(captions))
        
         if 'img_path_mapper' in ss:
            block_options = sorted(ss['img_path_mapper']['blocks'].keys())
            # st.write(block_options)
            # st.write(ss['img_path_mapper'])
            selected_block = col12_placeholder.selectbox("Select image blocks to visualize", block_options)
            if selected_block:
                block_img_path=ss['img_path_mapper']['blocks'][selected_block][0]
                if os.path.exists(block_img_path):
                    col12.image(block_img_path, use_container_width=True)
                    if ss['img_path_mapper']['blocks'][selected_block][1]!=None:
                        col12.write(ss['img_path_mapper']['blocks'][selected_block][1])
    


with mainTab3:
    col5,col6,col7 = st.columns(3,border=True)
    col8,col9,col10 = st.columns(3,border=True)
    samples=[x for x in os.listdir(sample_images_filepath) if x.endswith(".png")]
    sample_imagenames=random.sample(samples,6)

    img_filepath=os.path.join(sample_images_filepath,sample_imagenames[0])
    img=cv2.imread(img_filepath)
    col5.image(img,use_container_width=True)


    img_filepath=os.path.join(sample_images_filepath,sample_imagenames[1])
    img=cv2.imread(img_filepath)
    col6.image(img,use_container_width=True)


    img_filepath=os.path.join(sample_images_filepath,sample_imagenames[2])
    img=cv2.imread(img_filepath)
    col7.image(img,use_container_width=True)


    img_filepath=os.path.join(sample_images_filepath,sample_imagenames[3])
    img=cv2.imread(img_filepath)
    col8.image(img,use_container_width=True)


    img_filepath=os.path.join(sample_images_filepath,sample_imagenames[4])
    img=cv2.imread(img_filepath)
    col9.image(img,use_container_width=True)


    img_filepath=os.path.join(sample_images_filepath,sample_imagenames[5])
    img=cv2.imread(img_filepath)
    col10.image(img,use_container_width=True)


