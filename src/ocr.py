import cv2
import pytesseract

def get_text_from_image(image):
    ocr_result = pytesseract.image_to_string(image)
    return ocr_result

def get_image_countours(image, bounding_box, pad_size=30, border_type=cv2.BORDER_CONSTANT, buffer_space=5):
    height,width,depth=image.shape
    x, y, w, h=bounding_box
    roi = image[max(0,y-buffer_space):min(y+h+buffer_space,height), 
                max(0,x-buffer_space):min(x+w+buffer_space,width)]
      
    constant= cv2.copyMakeBorder(roi.copy(),pad_size,pad_size,pad_size,pad_size,border_type,value=[255,255,255])

    return constant