import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
from models import load_models
import copy
import streamlit as st
# import json
from collections import defaultdict
from surya.postprocessing.heatmap import draw_bboxes_on_image
from surya.detection import batch_text_detection
from surya.tables import batch_table_recognition

from utils import get_name_from_path

def display_image(image,cmap='gray'):
    dpi=80
    height,width=image.shape[:2]
    figsize=(width/float(dpi),height/float(dpi))
    fig=plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')
    ax=fig.add_subplot(111)
    ax.imshow(image,cmap=cmap)
    # plt.show()
    
    return fig


def grayscale(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def threshold_image(image,threshold,max_value):
    return cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def invert_image(image):
    return cv2.bitwise_not(image)

# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE


def dilate_image(image,iterations,val=0,kernel_size=(3,3)):
    '''
    image: array
    iterations: int
    val: int
    kernel_size: size 
    '''
    kernel_shape=morph_shape(val)
    kernal = cv2.getStructuringElement(kernel_shape, kernel_size)
    return cv2.dilate(image, kernel, iterations=iterations)

def noise_removal(image,dilation_kernel,erosion_kernel,iterations):
    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    image = cv2.dilate(image, kernel, iterations=iterations) #expansion of pixels

    kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8)
    image = cv2.erode(image, kernel, iterations=iterations) #thinning of pixels
    
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    if len(newImage.shape)>2:
        newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(newImage, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

    
# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)


def remove_borders(image):
    if len(image.shape)>2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)

def image_converter_pil_to_openCV(pil_image):
    cv_img = np.asarray(pil_image).copy()
    return cv_img

def image_converter_openCV_to_pil(cv2_image):
    # Notice the COLOR_BGR2RGB which means that the color is 
    # converted from BGR to RGB 
    color_coverted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)     
    pil_image = Image.fromarray(color_coverted)

    return pil_image




def rescale_bbox_with_buffer(bbox, buffer_size=2):
    # page_width, page_height = processor_size

    # img_width, img_height = image_size
    # width_scaler = img_width / page_width
    # height_scaler = img_height / page_height

    new_bbox = copy.deepcopy(bbox)
    # new_bbox[0] = int(new_bbox[0] * width_scaler-buffer_space)
    # new_bbox[1] = int(new_bbox[1] * height_scaler-buffer_space)
    # new_bbox[2] = int(new_bbox[2] * width_scaler+buffer_space)
    # new_bbox[3] = int(new_bbox[3] * height_scaler+buffer_space)

    new_bbox[0] = int(new_bbox[0] -buffer_size)
    new_bbox[1] = int(new_bbox[1] -buffer_size)
    new_bbox[2] = int(new_bbox[2] +buffer_size)
    new_bbox[3] = int(new_bbox[3] +buffer_size)
    return new_bbox


def draw_table_bounding_boxes(table_img, pred):
    '''
    Draws rows and cell bounding boxes on the table image. Returns the two images
    '''

    rows = [l.bbox for l in pred.rows]
    cols = [l.bbox for l in pred.cols]
    row_labels = [f"Row {l.row_id}" for l in pred.rows]
    col_labels = [f"Col {l.col_id}" for l in pred.cols]
    cells = [l.bbox for l in pred.cells]

    rc_image = copy.deepcopy(table_img)
    rc_image = draw_bboxes_on_image(rows, rc_image, labels=row_labels, label_font_size=20, color="blue")
    rc_image = draw_bboxes_on_image(cols, rc_image, labels=col_labels, label_font_size=20, color="red")

    cell_image = copy.deepcopy(table_img)
    cell_image = draw_bboxes_on_image(cells, cell_image, color="green")

    return rc_image, cell_image


def save_image_to_file(image, filepath):
    image.save(filepath)


def table_recognition_from_images(input_filepath, pil_image, layout_predictions, result_path, model_tag='surya'):
    
    _,_, det_model, det_processor, table_rec_model, tables_processor = load_models(model_tag)
    
    names=[get_name_from_path(input_filepath)]
    pnums = []
    prev_name = None
    for i, name in enumerate(names):
        if prev_name is None or prev_name != name:
            pnums.append(0)
        else:
            pnums.append(pnums[-1] + 1)

        prev_name = name
    
    table_imgs=[]
    table_counts=[]
    table_cells=[]
    buffer_size=3
    images, highres_images = [pil_image.copy()], [pil_image.copy()]
    
    # crop table snaps from each layout object
    for layout_pred, img, highres_img in zip(layout_predictions, images, highres_images):
        # The bbox for the entire table
        bbox = [l.bbox for l in layout_pred.bboxes if l.label in {'Table','TableOfContents'}]
        # Number of tables per page
        table_counts.append(len(bbox))

        if len(bbox) == 0:
          continue
        
        page_table_imgs = []
        highres_bbox = []
        for bb in bbox:
            highres_bb = rescale_bbox_with_buffer(bb,buffer_size)
            page_table_imgs.append(highres_img.crop(highres_bb))
            highres_bbox.append(highres_bb)

        table_imgs.extend(page_table_imgs)
        det_results = batch_text_detection(page_table_imgs, det_model, det_processor)
        cell_bboxes = [[{"bbox": tb.bbox, "text": None} for tb in det_result.bboxes] for det_result in det_results]
        table_cells.extend(cell_bboxes)

    table_preds = batch_table_recognition(table_imgs, table_cells, table_rec_model, tables_processor)
    os.makedirs(result_path, exist_ok=True)

    img_idx = 0
    prev_count = 0
    table_predictions = defaultdict(list)
    save_images=True

    for i in range(sum(table_counts)):
        while i >= prev_count + table_counts[img_idx]:
            prev_count += table_counts[img_idx]
            img_idx += 1

        pred = table_preds[i]
        orig_name = names[img_idx]
        pnum = pnums[img_idx]
        table_img = table_imgs[i]

        out_pred = pred.model_dump()
        out_pred["page"] = pnum + 1
        table_idx = i - prev_count
        out_pred["table_idx"] = table_idx
        table_predictions[orig_name].append(out_pred)

        if save_images:
            rc_image,cell_image=draw_table_bounding_boxes(table_img, pred)
            row_imagename= os.path.join(result_path, f"{name}_page{pnum + 1}_table{table_idx}_rc.png")
            # st.write(row_imagename)
            # st.write(os.getcwd())
            save_image_to_file(rc_image,row_imagename)
            save_image_to_file(cell_image, os.path.join(result_path, f"{name}_page{pnum + 1}_table{table_idx}_cells.png"))

    return table_predictions

    