import streamlit as st

from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.layout.model import load_model as load_layout_model
from surya.model.layout.processor import load_processor as load_layout_processor
from surya.tables import batch_table_recognition
# from surya.postprocessing.util import rescale_bboxes, rescale_bbox
from surya.model.table_rec.model import load_model as load_model
from surya.model.table_rec.processor import load_processor


@st.cache_resource
def load_models(model_tag='surya'):
    if model_tag=='surya':
        layout_model = load_layout_model()
        layout_processor = load_layout_processor()
        det_model = load_det_model()
        det_processor = load_det_processor()

        table_rec_model = load_model()
        tables_processor = load_processor()


        return (layout_model, layout_processor, det_model, 
                det_processor, table_rec_model, tables_processor)
    return None

@st.cache_resource
def get_text_detections(images: list, _det_model, _det_processor):
    line_predictions = batch_text_detection(images, _det_model, _det_processor)
    return line_predictions


@st.cache_resource
def get_layout_detection(images: list, _layout_model, _layout_processor, _line_predictions):
    layout_prediction = batch_layout_detection(images, _layout_model, _layout_processor, _line_predictions)
    return layout_prediction
