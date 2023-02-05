# Data-centric-AI-vector-selection
An example on how to use CLIP to select vectors from a dataset that are most similar to a given image or text.

## About
Selecting vectors from a dataset that are most similar to a given image or text is a common task in data-centric AI. This repository contains an example on how to use CLIP to do this. The example is based on the [CLIP](https://github.com/openai/CLIP) repository by OpenAI.

## Requirements
Install the requirements using the following command:
```
pip install -r requirements.txt
```

## Usage
Step 1: Download the BDD100K dataset and unzip it. 
Step 2: Run the notebook Export CLIP embeddings.ipynb to export the CLIP embeddings for the BDD100K dataset. This gets saved as data.csv. 
Step 3: Run the file data_centric_ai.py using the following command:
```
streamlit run data_centric_ai.py
```

## Example
You can type things such as "ambulance" to get images from ambulances. If there is a specific ambulance you want to see more of you can click "find simlar to X" and it will find the most similar images to the one you clicked on. 

