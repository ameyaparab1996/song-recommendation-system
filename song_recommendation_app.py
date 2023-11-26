import streamlit as st
import pandas as pd
import nltk
import re
import spotipy
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
from pages import generate_recommendations

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

st.set_page_config(
    page_title="Song Recommender",
    page_icon="🎵",
)


def intro():

    st.write("# Personalized Spotify Song Recommender")
    
    st.divider()
    st.markdown(
        """
        _“🎵 Music, once admitted to the soul, becomes a sort of spirit, and never dies.”_ — Edward Bulwer Lytton
        
        The songs we listen to can evoke nostalgia, lift our spirits, or simply provide the soundtrack to our daily routines. In the vast ocean of music, finding the perfect song can be like searching for hidden treasure. That's where personalized song recommendation systems come into play, offering a playlist of top songs that match the user's prompt and mood.
        
        This web application provides an interface for the user to input a prompt describing the type of songs or the mood they want to experience. Based on the user's input, the application employs NLP-based similarity algorithms to compare the lyrics of all the available songs in the database with the prompt, along with other features of the songs, to recommend suitable songs and build a playlist for the users. 
    """
    )

if "checkbox" not in st.session_state:
    st.session_state.checkbox = False
    
st.sidebar.success("Write a prompt to generate recommendations")
positive_prompt = st.sidebar.text_area('How do you want your songs to be?', 'Songs about long lost love that capture the complex emotions associated with the theme of love lost, nostalgia, and reflection', help="Positive prompt describing elements or mood of the songs you looking for.")
negative_prompt = st.sidebar.text_area('What should the songs be not like?', 'Breakup because of distance', help="Negative prompt describing how you don't want the songs to be.")
n = st.sidebar.number_input('Number of Songs to generate', min_value=5, max_value=50, value ="min", step=1)
if st.sidebar.button("Generate Playlist", type="primary") or st.session_state.checkbox:
    generate_recommendations(positive_prompt, negative_prompt, n)
else:
    intro()
    
