# Document Understanding
 
This repo contains code for an interactive Streamlit based app for extracting data from PDF documents. 

For this task, have utilised OCR and neural network based open-source tools such as Pytesseract, Surya, and OpenCV. 

The inputs are pdf files and the extracted data is saved in json format.

My experimental work is contained in the Jupyter notebook in nbs folder.

Here are some samples:

<img src="assets/samples/d.pdf_result.png" width="400" height="300"/>
<!-- ![Sample images!](/imgs/country_page.png) -->


<img src="assets/samples/e.pdf_result.png" width="400" height="300" />
<!-- ![Screenshot of the World Bank API page!](/imgs/world_bank_page.png) -->
<img src="assets/samples/g_page1_table0_cells.png" width="400" height="300" />


### Run:
The app is build using streamlit and python. Deployed on streamlit cloud and can be accessed here: [Document Understanding](https://document-understanding.streamlit.app/)

NOTE: If the app is on sleep mode. Press the button on the screen to bring it back on. App takes about 4-5mins to re-load.

**To run it locally:**
1. Install the python packages mentioned in the `requirements.txt` file. 
2. Then run the following command from the console: `streamlit run src/home.py`

The app should open in your web browser.

