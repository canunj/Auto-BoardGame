import streamlit as st

st.set_page_config(page_title='Auto-BG: The Game Concept Generator', layout='wide')

def application():
    ###Imports
    import streamlit as st
    import pandas as pd
    import numpy as np
    import re
    import nltk
    import urllib
    import pickle
    from nltk.corpus import stopwords
    from gensim.parsing import preprocess_string, strip_tags, strip_numeric, strip_multiple_whitespaces, stem_text, strip_punctuation, remove_stopwords
    import spacy, en_core_web_sm
    import torch
    from torch import nn
    import torch.nn.functional as F
    from transformers import T5ForConditionalGeneration,T5Tokenizer
    from title_generator import Title_Generator
    import gzip
    import io
    from spacy.tokens import DocBin
    from description_generator import input_manager, model_control
    import  Model_Constants as mc

    #non-ui helper functions
    def reader(url):
        url_file = io.BytesIO(urllib.request.urlopen(url).read())
        f = gzip.GzipFile(fileobj=url_file)
        data = f.read()
        obj = pickle.loads(data)
        f.close()
        return obj
            
    def token_expand(url):
        nlp = spacy.blank("en")
        url_file = urllib.request.urlopen(url)
        f = gzip.GzipFile(fileobj=url_file)
        data = f.read()
        obj = pickle.loads(data)
        f.close()
        doc_bin = DocBin().from_bytes(obj)
        docs = list(doc_bin.get_docs(nlp.vocab))
        return (docs[1:9],docs[9:192],docs[192:276],docs[276:3901])

    def revert_cats(gt, mec, cat, fam, coop):
        gt = ["game_type_" + x for x in gt]
        mec = ["mechanic_" + x for x in mec]
        cat = ["category_" + x for x in cat]
        fam = ["family_" + x for x in fam if x != "Game: [redacted]"]
        if coop == 1:
            co  = ["cooperative", "mechanic_Cooperative Game"]
        else:
            co  = []

        final_list = [gt,mec,cat,fam, co]
        return [item for sublist in final_list for item in sublist]

    ###Variables

    ###Data
    slim_df = pd.read_parquet('https://github.com/canunj/GameDescGenerator/blob/main/Model_Step_Data/slim_df.parquet.gzip?raw=true')
    search_tokens = token_expand("https://github.com/canunj/GameDescGenerator/blob/main/Persistent%20Objects/token_search.gz?raw=true")
    vector_df = pd.read_parquet('https://github.com/canunj/GameDescGenerator/blob/main/Model_Step_Data/vector_df.parquet.gzip?raw=true')
    category_keys = reader("https://github.com/canunj/GameDescGenerator/blob/main/Persistent%20Objects/current_keys.gz?raw=true")
    coop = [1,0]

    #Is this needed????
    ex_check = ["[Ee]verquest","[Cc]ivilization [Ii][IiVv]","[Cc]ivilization(?=:)","[Cc]ivilization [Ii][Ii]",
            "[Cc]ivilization [Ii][Ii][Ii]","[Aa]ge [Oo]f [Ee]mpires [Ii][Ii2]([Ii]|\b)", "[Rr]avenloft|[Cc]astle [Rr]avenloft",
            "[Ss]cythe(?=:|\b)","[Dd]ungeons [&Aa][ n][Dd ][ Ddr][Ddra][rg][oa][gn][os](ns|\b)",
            "[Aa]ge [Oo]f [Ee]mpires [Ii][Ii]: [Tt]he [Aa]ge [Oo]f [Kk]ings","[Aa]ge [Oo]f [Ee]mpires 2: [Tt]he [Aa]ge [Oo]f [Kk]ings",
            "[Aa]ge [Oo]f [Ee]mpires"] 

    ###Models
    Tgen = Title_Generator('./t5_model', slim_df)
    iman = input_manager(vector_df, slim_df, search_tokens)

    #UI Variables
    input = []
    output_dict = {}
    Title_v = False
    Desc_v = False

    #UI

    #Intro
    st.title("""Auto-BG: The Game Concept Generator!""")

    with st.expander("How to use", expanded=True):
        st.write(
            """
            Discover the concept for your next favorite game!
            
            Take your ideas and turn them into a full-fledged tabletop game concept through the power of deep learning!
            
            How do you use Auto-BG?
            Pick any set of choices from four selectors below: Family, Game, Mechanic, and Category.
            Then, if you are looking to lose together, activate the toggle for a cooperative game!
            
            Need more detail utilize the ? icons for more information.
            
            Need a guided walkthrough or want to see how Auto-BG performs on a known game?
            Open the menu in the top left corner!
            """
        )
    
    st.sidebar.success("Select a demo below.")

    with st.expander('Results', expanded=True):
        title_placeholder = st.container()
        desc_placeholder = st.container()
    
    #Form
    with st.form("Auto-BG"):

        col1, col2 = st.columns(2)
    
        with col1:
            Family_v = st.multiselect("Family", options=pd.Series(category_keys[4]), key='Family', max_selections=6, help='Descriptive niches for groupings of games.\n Maximum of six choices.')
    
        with col2:
            Game_v = st.multiselect("Game", options=pd.Series(category_keys[1]), key='Game', max_selections=2, help='Top level genres - Family, Strategy, etc.\n Maximum of two choices.')
        
        col3, col4 = st.columns(2)

        with col3:
            Category_v = st.multiselect("Category", options=pd.Series(category_keys[3]), key='Category', max_selections=3, help='The primary genres.\n Maximum of three choices.')
        
        with col4:
            Mechanics_v = st.multiselect("Mechanics", options=pd.Series(category_keys[2]), key='Mechanic', max_selections=5, help='Game rules!\n Maximum of five choices.')

        Cooperative_v = st.checkbox('Cooperative?', value=coop, key='CoopCheck')

        run = st.form_submit_button("Run Model")

        if run:
            if input  == revert_cats(Game_v, Mechanics_v, Category_v, Family_v, Cooperative_v):
                st.write('Already  Ran')
            else:
                desc_next = 0
                input = revert_cats(Game_v, Mechanics_v, Category_v, Family_v, Cooperative_v)
                ks = iman.input_parser(iman.set_input(input))
                mctrl = model_control(mc.SEND_KEY())
                mctrl.prompt_formatter(ks)
                desc = mctrl.call_api(status=desc_next)
                clean_desc = mctrl.resp_cleanup(desc)
                inter_pair = Tgen.candidate_generator(clean_desc)
                Tgen.candidate_score(inter_pair,ex_check)
                output = Tgen.title_check()
                output_dict[desc_next] = output
                print(output)
                title_placeholder.write(
                    """
                    #### Title:
                    """)
                title_placeholder.write(output_dict[desc_next][0])
                desc_placeholder.write(
                    """
                    ####  Description:
                    """)
                desc_placeholder.write(output_dict[desc_next][1])

page_names_to_funcs = {
    "Application": application
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

