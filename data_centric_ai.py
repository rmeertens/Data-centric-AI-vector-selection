import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache 
def get_model(): 
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

@st.cache(suppress_st_warning=True)
def knn(path): 
    st.write("Loading data")
    df = pd.read_csv(path)
    a = df['vector'].to_numpy()

    st.write("eval vector")
    all_vecs = [eval(vec) for vec in a[:]]
    all_vecs = np.array(all_vecs)

    st.write("fitting knn")
    nbrs = NearestNeighbors(n_neighbors=26, algorithm='auto', metric='cosine').fit(all_vecs)

    return df, all_vecs, nbrs
    
model, preprocess = get_model()    
df, all_vecs, nbrs = knn('data.csv')
st.write(f"Indexed {len(all_vecs)} images, at 1FPS this would be {len(all_vecs)/(60*60):.2f} hours of data")


to_show_vector = np.random.rand(512)
if 'source' not in st.session_state:
    st.session_state['source'] = 'random'
if 'similar_to' not in st.session_state: 
    st.session_state['similar_to'] = 0
if 'last_txt' not in st.session_state: 
    st.session_state['last_txt'] = ''

    
class SimilarShower: 
    def __init__(self, idx): 
        self.idx = idx
        
    def show_similar(self): 
        st.session_state['source'] = 'similar'
        st.session_state['similar_to'] = self.idx

in_txt = st.text_input('text_input', value="")
if in_txt: 
    if st.session_state['last_txt'] != in_txt: 
        st.session_state['last_txt'] = in_txt
        st.session_state['source'] = 'in_txt'

        
# pick what source to choose from        
if st.session_state['source'] == 'in_txt':        
    text = clip.tokenize([st.session_state['last_txt']]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features.numpy()[0]
    to_show_vector = text_features
elif st.session_state['source'] == 'similar': 
    to_show_vector = all_vecs[st.session_state['similar_to']]

    
# Do the actual showing    
distances, indices = nbrs.kneighbors([to_show_vector])
distances, indices = distances[0], indices[0]

idx = 0 
while idx < len(indices):
    try: 
        cols = st.columns(2) 

        path = df.iloc[indices[idx]].path
        image = Image.open(path)
        cols[0].image(image, caption=path)
        image_val = int(indices[idx])
        s = SimilarShower(image_val)
        cols[1].button('Find similar to ' + str(image_val), key=str(image_val), on_click=s.show_similar)

        cols[1].write(path)
        cols[1].write(f"Distance: {distances[idx]:.2f}")
        idx += 1
    except Exception as e: 
        st.write("tried to access", idx, len(indices))

