# Fluorescent Cell Detection App for Local Browser

This application runs the object detector YOLOv5 on images to locate immunofluorescently-labeled cells in tissue photographed in darkfield. It outputs the coordinates of all detected cells as an SVG file for download. You may also upload a series of single-channel images for colocalization analysis. To run analyses of the peptides pERK, DBH, and ChAT, input the single-channel image for each peptide. The program will run the object detection algorithm on each image and use the detection coordinates to estimate colocalization of pERK-DBH, pERK-ChAT, DBH-ChAT, and pERK-DBH-ChAT, outputting an SVG for alignment with your merged images.

Please refer to my [project blog](https://sites.google.com/view/project-blogs/blogs/cell-detection?authuser=0) for some implementation details.

## Usage

Clone repo to local machine
```
git clone https://github.com/alexarnal/web-app-fluorescent-cell-detection.git
```
Create new virtual environment in repo/folder
```
cd web-app-fluorescent-cell-detection
python3 -m venv env
source env/bin/activate
```
Install requirements in virtual enviroment
```
pip install --upgrade pip
pip install -r requirements.txt
```
Once you download the repo and install the dependencies, run `python app.py` and a browser window will open with the app. 

## Hardware & Software

MacBook Pro (16-inch, 2019) - macOS 11.7
Chrome Version 107.0.5304.110 (Official Build) (x86_64)
Safari Version 16.0 (16614.1.25.9.10, 16614)
Python Version 3.9.12

## To Do

Validate colocalization method 
Add online functionality
