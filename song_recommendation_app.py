import streamlit as st

def intro():
    import streamlit as st

    st.write("# Personalized Spotify Song Recommender")
    st.sidebar.success("Write a prompt to generate recommendations")
    
    st.divider()
    st.markdown(
        """
        _“Music, once admitted to the soul, becomes a sort of spirit, and never dies.”_ — Edward Bulwer Lytton
        
        The songs we listen to can evoke nostalgia, lift our spirits, or simply provide the soundtrack to our daily routines. In the vast ocean of music, finding the perfect song can be like searching for hidden treasure. That's where personalized song recommendation systems come into play, offering a playlist of top songs that match the user's prompt and mood.
        
        This web application provides an interface for the user to input a prompt describing the type of songs or the mood they want to experience. Based on the user's input, the application employs NLP-based similarity algorithms to compare the lyrics of all the available songs in the database with the prompt, along with other features of the songs, to recommend suitable songs and build a playlist for the users. 
    """
    )

def authenticate_spotify():
    from spotipy.oauth2 import SpotifyOAuth
    cid = '551b554ed7e14fafa21c5118bbba81fe'
    secret = 'baad9d3c05244d5fbfda7d5b9e8ebecb'
    return spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid,
                                               client_secret=secret,
                                               redirect_uri='http://localhost',
                                               scope='playlist-modify-public',
                                               open_browser=False))

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

def generate_recommendations(positive_prompt, negative_prompt, n):
    import streamlit as st
    import pandas as pd
    import spotipy
    import gensim
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    model = Doc2Vec.load("d2v_test.model")
    positive_vector = model.infer_vector(doc_words=normalize_document(positive_prompt), alpha=0.025)
    negative_vector = model.infer_vector(doc_words=normalize_document(negative_prompt), alpha=0.025)
    similar_doc = model.docvecs.most_similar(positive=[positive_vector], negative=[negative_vector], topn = n*10)

    sampled_df = pd.read_csv("https://drive.google.com/file/d/1uUQLH3IHQ6eN127jEzEKGUQQL6X1ZGcU/view?usp=sharing")
    recommendations_df = get_recommendations(sampled_df, similar_doc)

    sp = authenticate_spotify()

    spotify_df = pd.DataFrame(columns=['track_id', 'track_name', 'album_name', 'artists', 'album_image', 'preview_url'])

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
        spotify_df = spotify_df.append(spotify_data, ignore_index = True)
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
          spotify_df = spotify_df.append(spotify_data, ignore_index = True)
        else:
          print(f"No track found with the name '{track_name}'")
      else:
        print(f"No track found with the name '{track_name}'")

    st.table(spotify_df[:n])
    
def data_frame_demo():
    import streamlit as st
    import pandas as pd
    import altair as alt

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo shows how to use `st.write` to visualize Pandas DataFrames.

(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
"""
    )

    @st.cache_data
    def get_UN_data():
        AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
        df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
        return df.set_index("Region")

    try:
        df = get_UN_data()
        countries = st.multiselect(
            "Choose countries", list(df.index), ["China", "United States of America"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write("### Gross Agricultural Production ($B)", data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

page_names_to_funcs = {
    "—": intro,
    "DataFrame Demo": data_frame_demo
}

positive_prompt = st.sidebar.text_area('How do you want your songs to be?', 'Songs about long lost love that capture the complex emotions associated with the theme of love lost, nostalgia, and reflection')
negative_prompt = st.sidebar.text_area('Movie title', 'Breakup because of distance')
st.number_input('Number of Songs to generate', min_value=5, max_value=50, value ="min", step=1)
if st.sidebar.button("Generate Playlist", type="primary"):
    data_frame_demo()
else:
    intro()
#demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
#page_names_to_funcs[demo_name]()
    
