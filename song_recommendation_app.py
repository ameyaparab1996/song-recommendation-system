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

positive_prompt = st.sidebar.text_input('How do you want your songs to be?', 'Songs about long lost love that capture the complex emotions associated with the theme of love lost, nostalgia, and reflection')
negative_prompt = st.sidebar.text_input('Movie title', 'Breakup because of distance')
if st.sidebar.button("Generate Playlist", type="primary"):
    data_frame_demo()
#demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
#page_names_to_funcs[demo_name]()
    
