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
    
    def builder(ip):
        ks = iman.input_parser(iman.set_input(ip))
        mctrl.prompt_formatter(ks)
        descs = []
        for status in np.arange(0,3):
            desc = mctrl.call_api(status=status)
            clean_desc = mctrl.resp_cleanup(desc)
            inter_pair = Tgen.candidate_generator(clean_desc)
            out = Tgen.candidate_score(inter_pair,ex_check)
            descs.append(out)
            st.sidebar.success("Prompt " +str(status+1)+ " generated!")
        st.session_state.output_dict = {0:descs[0],1:descs[1],2:descs[2]}


    def title_check(next=1):
        if next==1:
            if st.session_state.title_iter == (len(st.session_state.output_dict[st.session_state.desc_iter]['titles'])-1):
                st.session_state.title_iter = 0
            else:
                st.session_state.title_iter+=1
        elif next==-1:
            if st.session_state.title_iter == 0:
                st.session_state.title_iter = (len(st.session_state.output_dict[st.session_state.desc_iter]['titles'])-1)
            else:
                st.session_state.title_iter-=1
        cur_title = st.session_state.output_dict[st.session_state.desc_iter]['titles'][st.session_state.title_iter][0]
        desc = re.sub(re.compile("__"),cur_title,st.session_state.output_dict[st.session_state.desc_iter]['text'])  

        return (cur_title, desc.lstrip())

    ###Variables

    ###Data
    @st.cache_resource
    def fetch_data():
        slim_df = pd.read_parquet('https://github.com/canunj/GameDescGenerator/blob/main/Model_Step_Data/slim_df.parquet.gzip?raw=true')
        search_tokens = token_expand("https://github.com/canunj/GameDescGenerator/blob/main/Persistent%20Objects/token_search.gz?raw=true")
        vector_df = pd.read_parquet('https://github.com/canunj/GameDescGenerator/blob/main/Model_Step_Data/vector_df.parquet.gzip?raw=true')
        category_keys = reader("https://github.com/canunj/GameDescGenerator/blob/main/Persistent%20Objects/current_keys.gz?raw=true")
        coop = [1,0]
        st.sidebar.success("Fetched Data!")
        return slim_df, search_tokens, vector_df, category_keys, coop
    
    slim_df, search_tokens, vector_df, category_keys, coop = fetch_data()

    #Is this needed????
    ex_check = ["[Ee]verquest","[Cc]ivilization [Ii][IiVv]","[Cc]ivilization(?=:)","[Cc]ivilization [Ii][Ii]",
            "[Cc]ivilization [Ii][Ii][Ii]","[Cc]ivilization V","[Aa]ge [Oo]f [Ee]mpires [Ii][Ii2]([Ii]|\b)", "[Rr]avenloft|[Cc]astle [Rr]avenloft",
            "[Ss]cythe(?=:|\b)","[Dd]ungeons [&Aa][ n][Dd ][ Ddr][Ddra][rg][oa][gn][os](ns|\b)",
            "[Aa]ge [Oo]f [Ee]mpires [Ii][Ii]: [Tt]he [Aa]ge [Oo]f [Kk]ings","[Aa]ge [Oo]f [Ee]mpires 2: [Tt]he [Aa]ge [Oo]f [Kk]ings",
            "[Aa]ge [Oo]f [Ee]mpires"] 

    ###Models
    @st.cache_resource
    def setup_models():
        return Title_Generator('./t5_model', slim_df), input_manager(vector_df, slim_df, search_tokens),  model_control(mc.SEND_KEY())

    Tgen, iman, mctrl = setup_models()
    
    #UI Variables
    if 'desc_iter' not in st.session_state:
        st.session_state.desc_iter = 0
    if 'title_iter' not in st.session_state:
        st.session_state.title_iter = -1
    if 'output_dict' not in st.session_state:
        st.session_state.output_dict = {}
    if 'inputs' not in st.session_state:
        st.session_state.inputs = []
    if 'cur_pair' not in st.session_state:
        st.session_state.cur_pair = ("","Run me!")
    if 'f_d' not in st.session_state:
        st.session_state.f_d = None
    if 'g_d' not in st.session_state:
        st.session_state.g_d = None
    if 'm_d' not in st.session_state:
        st.session_state.m_d = None
    if 'c_d' not in st.session_state:
        st.session_state.c_d = None
    if 'coop_d' not in st.session_state:
        st.session_state.coop_d = 0

    Title_v = False
    Desc_v = False

    ###ui helper functions
    def show_title(val): 
        out = title_check(next=val)
        st.session_state.cur_pair = out

    #UI

    #Intro
    st.title("""Auto-BG: The Game Concept Generator!""")

    with st.expander("How to use", expanded=True):
        st.write(
            """
            Discover the concept for your next favorite game!
                        
            How do you use Auto-BG?
            Pick any set of choices from four selectors below: Family, Game, Mechanic, and Category.
            If you are looking to lose together - activate the cooperative toggle.
            
            See ? icons for detailed information.
            
            Want to see how Auto-BG performs on a known game? Select any of the three pre-configured demos below!
            """
        )
    
    results = st.empty()

    with st.expander('Demos'):

        st.write("""Below are some buttons to run Auto-BG on some real games you might have heard of.
                 Press the button, and the corresponding attribute types will be placed into the drop-down fields below.
                 Then press run to see what Auto-BG comes up with!""")

        b1, b2, b3 =  st.columns(3)

    with b1:
        SoC = st.button('Settlers of Catan')
        if SoC:
            st.session_state.f_d = [
                'Animals: Sheep',
                'Components: Hexagonal Tiles',
                'Components: Wooden pieces & boards'
                ]
            st.session_state.g_d = ['Family Game', 'Strategy Game']
            st.session_state.m_d = [
                'Hexagon Grid',
                'Network and Route Building',
                'Random Production',
                'Trading',
                'Variable Set-up'
                ]
            st.session_state.c_d = [
                'Economic',
                'Negotiation'
                ]
            st.session_state.coop_d = 0

    with b2:
        TtR = st.button('Ticket to Ride')
        if TtR:
            st.session_state.f_d = [
                'Components: Map (Continental / National scale)',
                'Continents: North America',
                'Country: USA'
                ]
            st.session_state.g_d = ['Family Game']
            st.session_state.m_d = [
                'Contracts',
                'End Game Bonuses',
                'Network and Route Building',
                'Push Your Luck',
                'Set Collection'
                ]
            st.session_state.c_d = [
                'Trains'
                ]
            st.session_state.coop_d = 0
        
    with b3:
        P = st.button('Pandemic')
        if P:
            st.session_state.f_d = [
                'Components: Map (Global Scale)',
                'Components: Multi-Use Cards',
                'Medical: Diseases',
                'Region: The World',
                'Theme: Science'
                ]
            st.session_state.g_d = ['Family Game', 'Strategy Game']
            st.session_state.m_d = [
                'Action Points',
                'Cooperative Game',
                'Point to Point Movement',
                'Trading',
                'Variable Player Powers'
                ]
            st.session_state.c_d = [
                'Medical'
                ]
            st.session_state.coop_d = 1

    #Form
    with st.expander("Auto-BG", expanded=True):

        col1, col2 = st.columns(2)
    
        with col1:
            Family_v = st.multiselect("Family", options=pd.Series(category_keys[4]), key='Family', default=st.session_state.f_d, max_selections=6, help='Descriptive niches for groupings of games.\n Maximum of six choices.')
    
        with col2:
            Game_v = st.multiselect("Game", options=pd.Series(category_keys[1]), key='Game', default=st.session_state.g_d, max_selections=2, help='Top level genres - Family, Strategy, etc.\n Maximum of two choices.')
        
        col3, col4 = st.columns(2)

        with col3:
            Category_v = st.multiselect("Category", options=pd.Series(category_keys[3]), key='Category', default=st.session_state.c_d, max_selections=3, help='The primary genres.\n Maximum of three choices.')
        
        with col4:
            Mechanics_v = st.multiselect("Mechanics", options=pd.Series(category_keys[2]), key='Mechanic', default=st.session_state.m_d, max_selections=5, help='Game rules!\n Maximum of five choices.')

        Cooperative_v = st.checkbox('Cooperative?', value=st.session_state.coop_d, key='CoopCheck')

        run = st.button("Run Model")

        if run:
            if st.session_state.inputs  == revert_cats(Game_v, Mechanics_v, Category_v, Family_v, Cooperative_v):
                st.write('Inputs did not change, results currently loaded.')
            else:
                st.session_state.output_dict = {}
                st.session_state.title_iter = -1
                st.session_state.desc_iter = 0
                st.session_state.inputs  = revert_cats(Game_v, Mechanics_v, Category_v, Family_v, Cooperative_v)
                builder(st.session_state.inputs)
                st.session_state.cur_pair = title_check()

    if st.session_state.output_dict == {}:
        results.empty()
    else:
        with results.expander('Results', expanded=True):
        
            
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                if st.button("See Previous Title"):
                    show_title(-1)

            with t_col2:
                if st.button("See Next Title"):
                    show_title(1)
            
            
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                if st.button("See Previous Description"):
                    if st.session_state.desc_iter == 0:
                        st.session_state.desc_iter = 2
                        st.session_state.title_iter = -1
                    else:
                        st.session_state.desc_iter -= 1
                        st.session_state.title_iter = -1
                    show_title(1)

            with d_col2:
                if st.button("See Next Description"):
                    if st.session_state.desc_iter == 2:
                        st.session_state.desc_iter = 0
                        st.session_state.title_iter = -1
                    else:
                        st.session_state.desc_iter += 1
                        st.session_state.title_iter = -1
                    show_title(1)

            st.write(
                """
                #### Title:
                """)
            st.write(st.session_state.cur_pair[0])

            st.write(
                """
                ####  Description:
                """)
            st.write(st.session_state.cur_pair[1])



def demo():
    st.text('This is demo, wow more changes')

page_names_to_funcs = {
    "Application": application,
    "Demo": demo
}

demo_name = st.sidebar.selectbox("Choose a page:", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

