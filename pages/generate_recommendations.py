
def authenticate_spotify(auth_scope):
    
    cid = '551b554ed7e14fafa21c5118bbba81fe'
    secret = 'baad9d3c05244d5fbfda7d5b9e8ebecb'
    
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, 
                                                          client_secret=secret)

    if auth_scope == 'playlist-modify-public':
        auth_manager= SpotifyOAuth(client_id=cid,
                                   client_secret=secret,
                                   redirect_uri='https://song-recommendation-system.streamlit.app/',
                                   scope=auth_scope,
                                   open_browser=True)
        return spotipy.Spotify(auth_manager = auth_manager)
    else:
        return spotipy.Spotify(client_credentials_manager = client_credentials_manager)

  
def get_recommendations(songs_df, similar_doc):
    
  recommendation = pd.DataFrame(columns = ['id','title','tag','artist','score'])
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
    
def normalize_document(doc):
    stop_words = nltk.corpus.stopwords.words('english')
    
    # remove all strings inside [ ]
    doc = re.sub(r'\[.*?\]', '', doc)
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    doc = doc.replace("nbsp","")
    # tokenize document
    tokens = nltk.word_tokenize(doc)

     # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]

    return filtered_tokens
    
def generate_recommendations(positive_prompt, negative_prompt, n):
    
    st.markdown("# Spotify Song Recommendations")
    st.markdown("###### Here are the songs that best match your prompt:")

    if st.session_state.checkbox == False:
        progress_text = "Fetching Songs 🎶. Please wait ⌛."
        my_bar = st.progress(0, text=progress_text)
        
    model = Doc2Vec.load("data/d2v_test.model")
    positive_vector = model.infer_vector(doc_words=normalize_document(positive_prompt), alpha=0.025)
    negative_vector = model.infer_vector(doc_words=normalize_document(negative_prompt), alpha=0.025)
    similar_doc = model.docvecs.most_similar(positive=[positive_vector], negative=[negative_vector], topn = n*10)
    
    sampled_df = pd.read_csv("data/sampled_songs.csv", index_col ="Unnamed: 0")
    recommendations_df = get_recommendations(sampled_df, similar_doc)
    
    sp = authenticate_spotify('fetch_songs')
    
    spotify_df = pd.DataFrame(columns=['track_id', 'track_name', 'album_name', 'artists', 'album_image', 'preview_url', 'track_uri'])
    
    for i in range(0,len(recommendations_df)):
        track_name = recommendations_df.iloc[i, 1]
        artist_name = recommendations_df.iloc[i, 3]
        results = sp.search(q=f'track:{track_name} artist:{artist_name}', type='track', limit=1)
        if len(results['tracks']['items']) > 0:
            track = results['tracks']['items'][0]
            spotify_data = {'track_id': track['id'],
                            'track_name': track['name'],
                            'album_name': track['album']['name'],
                            'artists': [artist['name'] for artist in track['artists'] if 'name' in artist],
                            'album_image': track['album']['images'][0]['url'],
                            'preview_url': track['preview_url'],
                            'track_uri': track['uri']}
            spotify_df = pd.concat([spotify_df,pd.DataFrame([spotify_data])])
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
                                'track_uri': track['uri']}
                spotify_df = pd.concat([spotify_df,pd.DataFrame([spotify_data])])
            else:
                print(f"No track found with the name '{track_name}'")
        else:
            print(f"No track found with the name '{track_name}'")
        if st.session_state.checkbox == False:
            my_bar.progress(int(len(spotify_df)*100/n), text=progress_text)
        if(len(spotify_df) == n):
            break

    if st.session_state.checkbox == False:
        my_bar.empty()

    display_recommendations(spotify_df, positive_prompt)


def display_recommendations(spotify_df, positive_prompt):
    css = '''
    <style>
        .stMarkdown p, [data-testid="stCheckbox"] {
            height: 140px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .stAudio {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            height: 140px !important;
            padding-bottom: 30px !important;
        }

        h3 {
            text-align: center !important;
            height: 100px !important;
        } 
        
        p {
            text-align: center !important;
        }
    </style>
    '''
    
    st.write(css, unsafe_allow_html=True)
    album_image_col, track_name_col, artists_col, preview_col, playlist_col = st.columns([1,1,1,3,1])
    album_image_col.subheader("Album Cover", divider='green')
    track_name_col.subheader("Track       ", divider='green')
    artists_col.subheader("Artists       ", divider='green')
    preview_col.subheader("Preview       ", divider='green')
    playlist_col.subheader("Add to Playlist", divider='green')
    
    include = [True] * (len(spotify_df))
    spotify_df['include'] = [True] * (len(spotify_df))
    
    def update_include():
        spotify_df['include'] = include
        st.session_state.checkbox = True
    
    with st.form(key='my_form'):
        for j in range(0, len(spotify_df)):
            album_image_col.image(spotify_df.iloc[j, 4], caption=spotify_df.iloc[j, 2])
            track_name_col.markdown('<p>' + spotify_df.iloc[j, 1] + '</p>', unsafe_allow_html=True)
            artists_col.markdown('<p>' + ', '.join(spotify_df.iloc[j, 3]) + '</p>', unsafe_allow_html=True)
            preview_col.audio(spotify_df.iloc[j, 5], format="audio/mp3")
            include[j] = playlist_col.checkbox("",key=j, value=spotify_df.iloc[j, 7], label_visibility="collapsed")
        username = st.text_input('Spotify Username', help="To find your username go to Settings and privacy > Account")
        playlist_name = st.text_input('Playlist Name', help="Give a name to your playlist which will appear in your library")
        create_button = st.form_submit_button(label='Create Playlist', on_click=update_include())

    #st.dataframe(spotify_df[spotify_df['include']])
        
    if create_button:
        create_playlist(list(spotify_df.loc[spotify_df['include'] == True, 'track_uri']), username, playlist_name, positive_prompt)
        

def create_playlist(track_uri, username, playlist_name, playlist_description):
    #sp = authenticate_spotify('playlist-modify-public')
    cid = '551b554ed7e14fafa21c5118bbba81fe'
    secret = 'baad9d3c05244d5fbfda7d5b9e8ebecb'
    redirect_uri='https://song-recommendation-system.streamlit.app/'

    sp_oauth = SpotifyOAuth(cid, secret, redirect_uri, scope='playlist-modify-public')
    auth_url = sp_oauth.get_authorize_url()
    st.markdown(f"[Login with Spotify]({auth_url})")
    redirected_url = st.text_input("Enter the redirected URL after login:")
    if redirected_url:
        token_info = sp_oauth.get_access_token(redirected_url)
        if token_info:
            sp = spotipy.Spotify(auth=token_info["access_token"])
            st.success("Successfully authenticated with Spotify!")
        else:
            st.error("Failed to authenticate. Please try again.")
    
    #playlist_info = sp.user_playlist_create(user=username, name=playlist_name, public=True, description=playlist_description)
    #playlist_id = playlist_info['id']
    #sp.playlist_add_items(playlist_id, track_uri)
    #st.toast("Your Playlist '" + playlist_name + "' was created successfully", icon='✅')
    