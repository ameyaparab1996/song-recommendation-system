# Setup code
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import time
import base64
import gzip
import gdown
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spotipy
import logging     
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
from urllib.parse import urlparse, parse_qs
from PIL import Image

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Song Recommender",
    page_icon="🎵",
)

# Function to display the Home Page
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


# Function to dowload files stored in drive
def download_data():
    
    data = '1-FS40zGGXThB9msiXEUE5UCkTe8MN39F'
    url = f'https://drive.google.com/uc?id={data}'
    output = 'filtered_songs.csv.gz'
    gdown.download(url, output, quiet=False)
    
    # Decompress the gzip file
    with gzip.open(output, 'rt') as f:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(f)
        
    logger.info("Data downloaded")
    
    return df

# Function to authenticate to spotify developer account
def authenticate_spotify(auth_scope):
    
    cid = '551b554ed7e14fafa21c5118bbba81fe'
    secret = 'baad9d3c05244d5fbfda7d5b9e8ebecb'

    if auth_scope == 'playlist-modify-public':
        # Authentication method to create and update playlist
        auth_manager= SpotifyOAuth(client_id=cid,
                                   client_secret=secret,
                                   redirect_uri='https://song-recommendation-system.streamlit.app/',
                                   scope=auth_scope,
                                   open_browser=True)
        return auth_manager
        
    else:
        # Authentication method to get songs from Spotify
        client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        return spotipy.Spotify(client_credentials_manager = client_credentials_manager)


# Function to Create Playlist
def create_playlist(sp_oauth, redirected_url, track_uri, username, playlist_name, playlist_description):
    
        parsed_url = urlparse(redirected_url)
        query_params = parse_qs(parsed_url.query)
        code = query_params.get('code', [None])[0]
        token_info = sp_oauth.get_access_token(code)
    
        if token_info:
            sp = spotipy.Spotify(auth=token_info["access_token"])
            logger.info("Successfully authenticated with Spotify!")
            playlist_info = sp.user_playlist_create(user=username, name=playlist_name, public=True, description=playlist_description)
            playlist_id = playlist_info['id']
            sp.playlist_add_items(playlist_id, track_uri)
            logger.info("Playlist Created")

            # Clear all session variables
            for key in st.session_state.keys():
                del st.session_state[key]

            # Display message on UI
            st.toast("Your Playlist '" + playlist_name + "' was created successfully", icon='✅')
            time.sleep(4)
            st.rerun()
            
        else:
            # Display Error Message
            st.error("Failed to authenticate. Please try again.")


# Function to create a dataframe of recommendations from the Genius dataset
def get_recommendations(songs_df, similar_doc):
    
  recommendation = pd.DataFrame(columns = ['id','title','tag','artist','lyrics','score'])
  count = 0
    
  for doc_tag, score in similar_doc:
    recommendation.at[count, 'id'] = songs_df.loc[int(doc_tag)].id
    recommendation.at[count, 'title'] = songs_df.loc[int(doc_tag)].title
    recommendation.at[count, 'tag'] = songs_df.loc[int(doc_tag)].tag
    recommendation.at[count, 'artist'] = songs_df.loc[int(doc_tag)].artist
    recommendation.at[count, 'lyrics'] = songs_df.loc[int(doc_tag)].lyrics
    recommendation.at[count, 'score'] = score
    count += 1
      
  return recommendation
    

# Function to process text document (prompts)
def normalize_document(doc, prompt = False):
    
    stop_words = nltk.corpus.stopwords.words('english')
    
    # Remove all strings inside [ ]
    doc = re.sub(r'\[.*?\]', '', doc)
    
    # Lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    doc = doc.replace("nbsp","")
    
    # Tokenize document
    tokens = nltk.word_tokenize(doc)

     # Filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]

    if prompt:
        return filtered_tokens
    else:
        return ' '.join(filtered_tokens)

# Function to generate word cloud of lyrics
def generate_wordcloud(text):
    wordcloud = WordCloud(background_color="black", mode="RGBA", colormap='Greens', repeat=False).generate(text)
    fig = plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    image_path = "data/wordcloud.png"
    image_path1 = "data/wordcloud1.png"
    fig.savefig(image_path, bbox_inches='tight', transparent=True, pad_inches= 0)
    image = Image.open(image_path)
    image_rgba = image.convert("RGBA")
    image_array = np.array(image_rgba)
    image_array[:, :, 3] = 30
    image_rgba = Image.fromarray(image_array, 'RGBA')
    #image_rgb = image_rgba.convert('RGB')
    image_rgba.save("data/wordcloud1.png")
    return image_path1
    


def set_background(main_bg):

    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: contain
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    

# Function to generate recommendations from the prompts
def generate_recommendations(positive_prompt, negative_prompt, n):
    
    st.markdown("# Spotify Song Recommendations")
    st.markdown("###### Here are the songs that best match your prompt")

    if (st.session_state.checkbox == False and st.session_state.create == False) or st.session_state.prompt_update:
        
        progress_text = "Fetching Songs 🎶. Please wait ⌛."
        my_bar = st.progress(0, text=progress_text)

        # Use model to find similar songs
        model = Doc2Vec.load("data/d2v_test.model")
        positive_vector = model.infer_vector(doc_words=normalize_document(positive_prompt, True), alpha=0.025)
        negative_vector = model.infer_vector(doc_words=normalize_document(negative_prompt, True), alpha=0.025)
        similar_doc = model.docvecs.most_similar(positive=[positive_vector], negative=[negative_vector], topn = n*10)
        
        df = pd.read_csv("data/sampled_songs.csv", index_col ="Unnamed: 0")
        #test_df = download_data()
            
        recommendations_df = get_recommendations(df, similar_doc)
        sp = authenticate_spotify('fetch_songs')
        spotify_df = pd.DataFrame(columns=['track_id', 'track_name', 'album_name', 'artists', 'album_image', 'preview_url', 'track_uri', 'lyrics'])

        # Search the recommended songs in Spotify and create a dataframe of songs
        for i in range(0,len(recommendations_df)):
            
            track_name = recommendations_df.iloc[i, 1]
            artist_name = recommendations_df.iloc[i, 3]

            # Search using exact track and artist name
            results = sp.search(q=f'track:{track_name} artist:{artist_name}', type='track', limit=1)
            if len(results['tracks']['items']) > 0:
                track = results['tracks']['items'][0]
                spotify_data = {'track_id': track['id'],
                                'track_name': track['name'],
                                'album_name': track['album']['name'],
                                'artists': [artist['name'] for artist in track['artists'] if 'name' in artist],
                                'album_image': track['album']['images'][0]['url'],
                                'preview_url': track['preview_url'],
                                'track_uri': track['uri'],
                                'lyrics': recommendations_df.iloc[i, 4]}
                spotify_df = pd.concat([spotify_df,pd.DataFrame([spotify_data])])

            # Search using track name and then match artists from Genius data
            elif len(results['tracks']['items']) == 0:
                results_new = sp.search(q=f'track:{track_name}', type='track', limit=1)
                
                if len(results_new['tracks']['items']) > 0 and re.sub(r'[^a-zA-Z0-9\s]', '', results_new['tracks']['items'][0]['name'].lower()) == re.sub(r'[^a-zA-Z0-9\s]', '', recommendations_df.iloc[i,1].lower()):
                    track = results_new['tracks']['items'][0]
                    spotify_data = {'track_id': track['id'],
                                    'track_name': track['name'],
                                    'album_name': track['album']['name'],
                                    'artists': [artist['name'] for artist in track['artists'] if 'name' in artist],
                                    'album_image': track['album']['images'][0]['url'],
                                    'preview_url': track['preview_url'],
                                    'track_uri': track['uri'],
                                    'lyrics': recommendations_df.iloc[i, 4]}
                    spotify_df = pd.concat([spotify_df,pd.DataFrame([spotify_data])])
                    
                else:
                    print(f"No track found with the name '{track_name}'")
                    
            else:
                print(f"No track found with the name '{track_name}'")

            # Update progress bar %
            if st.session_state.checkbox == False:
                my_bar.progress(int(len(spotify_df)*100/n), text=progress_text)

            # Break when required number of songs are fetched
            if(len(spotify_df) == n):
                break
                
        st.session_state.spotify_df = spotify_df
        my_bar.empty()
    
    display_recommendations(st.session_state.spotify_df, positive_prompt)


# Function to display the recommended songs
def display_recommendations(spotify_df, positive_prompt):

    # Display wordcloud
    if st.session_state.create == False or st.session_state.prompt_update:
        processed_lyrics = spotify_df['lyrics'].apply(normalize_document)
        combined_lyrics = " ".join(processed_lyrics)
        image = generate_wordcloud(combined_lyrics)
        set_background(generate_wordcloud(combined_lyrics))
        
    css = f'''
    <style>
        
        .stMarkdown p, [data-testid="stCheckbox"], [data-testid="stStyledFullScreenFrame"] {{
            height: 140px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }}

        .stAudio {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            height: 140px !important;
            padding-bottom: 50px !important;
        }}

        h3 {{
            text-align: center !important;
        }}
        
        p {{
            text-align: center !important;
        }}
    </style>
    '''
    st.write(css, unsafe_allow_html=True)
    
    # Display table headers
    album_image_col, track_name_col, artists_col, preview_col, playlist_col = st.columns([1,1,1,3,1])
    album_image_col.subheader("Album", divider='green')
    track_name_col.subheader("Track", divider='green')
    artists_col.subheader("Artists", divider='green')
    preview_col.subheader("Preview", divider='green')
    playlist_col.subheader("Add", divider='green')

    if st.session_state.checkbox == False or !(False in st.session_state.include.contains):
        # Include all songs by default
        logger.info("checkbox" + str(st.session_state.checkbox))
        logger.info("prompt" + str(st.session_state.prompt_update))
        st.session_state.include = [True] * (len(spotify_df))
        st.session_state.sp_oauth = authenticate_spotify('playlist-modify-public')
        
    def update_include():
        # Update flag when checkbox value is changed
        logger.info("checkbox" + str(st.session_state.checkbox))
        logger.info("prompt" + str(st.session_state.prompt_update))
        st.session_state.checkbox = True
        
    spotify_df['include'] = st.session_state.include

    if st.session_state.create == False or st.session_state.prompt_update:

        for j in range(0, len(spotify_df)):
            album_image_col.image(spotify_df.iloc[j, 4], caption=spotify_df.iloc[j, 2])
            track_name_col.markdown('<p>' + spotify_df.iloc[j, 1] + '</p>', unsafe_allow_html=True)
            artists_col.markdown('<p>' + ', '.join(spotify_df.iloc[j, 3]) + '</p>', unsafe_allow_html=True)
            preview_col.audio(spotify_df.iloc[j, 5], format="audio/mp3")
            logger.info("before checkbox" + str(st.session_state.create))
            st.session_state.include[j] = playlist_col.checkbox("",key=j, value=spotify_df.iloc[j, 7], label_visibility="collapsed", on_change=update_include())
            if st.session_state.include[j] != spotify_df.iloc[j, 7]:
                #update_include()
                spotify_df.iloc[j, 7] = st.session_state.include[j]

        # Display wordcloud
       #st.image(generate_wordcloud(combined_lyrics), use_column_width=True, output_format="png", channels="RGBA")
        
        with st.form(key='playlist_form'):
            st.session_state.username = st.text_input('Spotify Username' ,help="To find your username go to Settings and privacy > Account")
            auth_url = st.session_state.sp_oauth.get_authorize_url()
            st.markdown(f"[Login with Spotify]({auth_url})")
            st.session_state.redirected_url = st.text_input("Enter the redirected URL after login:", help="URL where you will be redirected after clicking above link")
            st.session_state.playlist_name = st.text_input('Playlist Name', help="Give a suitable name to your playlist which will appear in your library")
            logger.info("Before form submission")
            st.session_state.submit_button = st.form_submit_button(label='Create Playlist')

            # Update flag when button is clicked
            if st.session_state.submit_button:
                st.session_state.create = True
                logger.info("inside form" + str(st.session_state.create))
       
        st.dataframe(spotify_df[spotify_df['include'] == True])
        # Update flag once all the results are displayed again after changing prompt
        st.session_state.prompt_update = False

    # Call the function to create playlist when the flag of the button is True
    if st.session_state.create:
        logger.info("outside form" + str(st.session_state.create))
        create_playlist( st.session_state.sp_oauth,  st.session_state.redirected_url, list(spotify_df.loc[spotify_df['include'] == True, 'track_uri']), st.session_state.username, st.session_state.playlist_name, positive_prompt)

# Function to update flag when the prompts are changed
def prompt_update():
    st.session_state.prompt_update = True

# Initialize session state
if "checkbox" not in st.session_state:
    st.session_state.checkbox = False

if "create" not in st.session_state:
    st.session_state.create = False

if "prompt_update" not in st.session_state:
    st.session_state.prompt_update = False

if "spotify_df" not in st.session_state:
    st.session_state.spotify_df = ""

if "include" not in st.session_state:
    st.session_state.include = []

if "username" not in st.session_state:
    st.session_state.username = ""

if "playlist_name" not in st.session_state:
    st.session_state.playlist_name = ""

if "generate_button" not in st.session_state:
    st.session_state.generate_button = False

if "submit_button" not in st.session_state:
    st.session_state.submit_button = False

if 'redirected_url' not in st.session_state:
    st.session_state.redirected_url = "https://song-recommendation-system.streamlit.app/"

if 'sp_oauth' not in st.session_state:
    st.session_state.sp_oauth = ""

# Sidebar content    
st.sidebar.write("#### Write a prompt to generate recommendations")
positive_prompt = st.sidebar.text_area('How do you want your songs to be?', 'Songs about long lost love that capture the complex emotions associated with the theme of love lost, nostalgia, and reflection', help="Positive prompt describing elements or mood of the songs you looking for", on_change=prompt_update())
negative_prompt = st.sidebar.text_area('What should the songs be not like?', 'Breakup because of distance', help="Negative prompt describing how you don't want the songs to be", on_change=prompt_update())
n = st.sidebar.number_input('Number of Songs to generate', min_value=5, max_value=50, value ="min", step=1, on_change=prompt_update())

st.session_state.generate_button = False
st.session_state.generate_button = st.sidebar.button("Generate Playlist", type="primary")

# Condition to display appropriate page based on the flags
if (st.session_state.generate_button and st.session_state.prompt_update) or st.session_state.checkbox or st.session_state.create:
    generate_recommendations(positive_prompt, negative_prompt, n)
else:
    intro()
