import streamlit as st

st.set_page_config(page_title='Auto-BG: The Game Concept Generator', layout='wide')

def application():
    ###Imports
    import pandas as pd
    import numpy as np
    import re
    import urllib
    import pickle
    import spacy
    from spacy.tokens import DocBin
    from title_generator import Title_Generator
    import gzip
    import io
    from description_generator import input_manager, model_control

    #UI Session Variables
    if 'desc_iter' not in st.session_state:
        st.session_state.desc_iter = 0
    if 'title_iter' not in st.session_state:
        st.session_state.title_iter = 0
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

    #non-ui helper functions
    #reader code extended from https://gist.github.com/thearn/5424244 for alternate load format
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



    def title_check(next=0):
        if next==1:
            if st.session_state.title_iter == (len(st.session_state.output_dict[st.session_state.desc_iter]['titles'])-1):
                st.session_state.title_iter = 0
            else:
                st.session_state.title_iter +=1
        elif next==-1:
            if st.session_state.title_iter == 0:
                st.session_state.title_iter = (len(st.session_state.output_dict[st.session_state.desc_iter]['titles'])-1)
            else:
                st.session_state.title_iter -=1
        else:
            st.session_state.title_iter = 0

        cur_title = st.session_state.output_dict[st.session_state.desc_iter]['titles'][st.session_state.title_iter][0]
        desc = re.sub(re.compile("__"),cur_title,st.session_state.output_dict[st.session_state.desc_iter]['text'])  

        return (cur_title, desc.lstrip())

    def show_title(val): 
        out = title_check(next=val)
        st.session_state.cur_pair = out

    def PT_button_clicked():
        show_title(-1)

    def NT_button_clicked():
        show_title(1)

    def PD_button_clicked():
        if st.session_state.desc_iter == 0:
            st.session_state.desc_iter = 2
            st.session_state.title_iter = 0
        else:
            st.session_state.desc_iter -= 1
            st.session_state.title_iter = 0
        show_title(0)

    def ND_button_clicked():
        if st.session_state.desc_iter == 2:
                st.session_state.desc_iter = 0
                st.session_state.title_iter = 0
        else:
            st.session_state.desc_iter += 1
            st.session_state.title_iter = 0
        show_title(0)


        
    ###Variables

    ###Data
    @st.cache_data
    def fetch_data():
        slim_df = pd.read_parquet('https://github.com/canunj/Auto-BoardGame/blob/main/Model_Step_Data/slim_df.parquet.gzip?raw=true')
        search_tokens = token_expand("https://github.com/canunj/Auto-BoardGame/blob/main/Persistent%20Objects/token_search.gz?raw=true")
        vector_df = pd.read_parquet('https://github.com/canunj/Auto-BoardGame/blob/main/Model_Step_Data/vector_df.parquet.gzip?raw=true')
        category_keys = reader("https://github.com/canunj/Auto-BoardGame/blob/main/Persistent%20Objects/current_keys.gz?raw=true")
        st.sidebar.success("Fetched Data!")
        return slim_df, search_tokens, vector_df, category_keys
    
    slim_df, search_tokens, vector_df, category_keys = fetch_data()
    
    ex_check = ["[Ee]verquest","[Cc]ivilization [Ii][IiVv]","[Cc]ivilization(?=:)","[Cc]ivilization [Ii][Ii]",
            "[Cc]ivilization [Ii][Ii][Ii]","[Cc]ivilization V","[Aa]ge [Oo]f [Ee]mpires [Ii][Ii2]([Ii]|\b)", "[Rr]avenloft|[Cc]astle [Rr]avenloft",
            "[Ss]cythe(?=:|\b)","[Dd]ungeons [&Aa][ n][Dd ][ Ddr][Ddra][rg][oa][gn][os](ns|\b)",
            "[Aa]ge [Oo]f [Ee]mpires [Ii][Ii]: [Tt]he [Aa]ge [Oo]f [Kk]ings","[Aa]ge [Oo]f [Ee]mpires 2: [Tt]he [Aa]ge [Oo]f [Kk]ings",
            "[Aa]ge [Oo]f [Ee]mpires","Doctor Who"] 

    ###Models
    @st.cache_resource
    def setup_models():
        spacy.cli.download("en_core_web_md")
        return Title_Generator('./t5_model', slim_df), input_manager(vector_df, slim_df, search_tokens),  model_control(apikey=st.secrets.key,model_id=st.secrets.model)

    Tgen, iman, mctrl = setup_models()
    


    #UI

    #Application

    ###Intro
    st.title("""Auto-BG: The Game Concept Generator""")

    with st.expander("How to use", expanded=True):
        st.write(
            """
            Discover the concept for your next favorite game!
                        
            How do you use Auto-BG?

            Pick any set of tags from four selectors below: Family, Game, Mechanic, and Category.
            If you are looking to lose together - activate the cooperative toggle.
            
            See ? icons for detailed information on each type of tag.
            
            Select any pre-configured demo below to see how Auto-BG works on the tag set for a popular board game. 
            """
        )
    
    results = st.empty()

    ###Demo
    with st.expander('Demos'):

        st.write("""These buttons run Auto-BG on the tag set for real games you might be familiar with,
                 choose a button and the corresponding tags automatically fill the selectors below.
                 Press run and see how Auto-BG creates an alternate concept for these hit titles!
                 """)

        b1, b2, b3 =  st.columns(3)

    with b1:
        SoC = st.button('Catan', use_container_width=True)
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
        TtR = st.button('Ticket to Ride', use_container_width=True)
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
        P = st.button('Pandemic', use_container_width=True)
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
                'Point to Point Movement',
                'Trading',
                'Variable Player Powers'
                ]
            st.session_state.c_d = [
                'Medical'
                ]
            st.session_state.coop_d = 1

    ###Form
    with st.expander("Auto-BG", expanded=True):

        col1, col2 = st.columns(2)
    
        with col1:
            Family_v = st.multiselect("Family", options=pd.Series(category_keys[4][8:]), key='Family', default=st.session_state.f_d, max_selections=6, help='Descriptive niches for groupings of games.\n Maximum of six choices.')
    
        with col2:
            Game_v = st.multiselect("Game", options=pd.Series(category_keys[1]), key='Game', default=st.session_state.g_d, max_selections=2, help='Top level genres - Family, Strategy, etc.\n Maximum of two choices.')
        
        col3, col4 = st.columns(2)

        with col3:
            Category_v = st.multiselect("Category", options=pd.Series(category_keys[3]), key='Category', default=st.session_state.c_d, max_selections=3, help='Expanded genre tags.\n Maximum of three choices.')
        
        with col4:
            Mechanics_v = st.multiselect("Mechanics", options=pd.Series([x for x in category_keys[2] if x != "Cooperative Game"]), key='Mechanic', default=st.session_state.m_d, max_selections=5, help='Game rules!\n Maximum of five choices.')

        Cooperative_v = st.checkbox('Cooperative?', value=st.session_state.coop_d, key='CoopCheck')
        
        run = st.button("Run Model", use_container_width=True)

        if run:
            if st.session_state.inputs  == revert_cats(Game_v, Mechanics_v, Category_v, Family_v, Cooperative_v):
                st.write('Inputs did not change, results currently loaded.')
            else:
                
                st.session_state.desc_iter = 0
                st.session_state.title_iter = 0
                st.session_state.output_dict = {}

                if Cooperative_v == True:
                    Mechanics_v.append('Cooperative Game')

                st.session_state.inputs  = revert_cats(Game_v, Mechanics_v, Category_v, Family_v, Cooperative_v)
                builder(st.session_state.inputs)
                st.session_state.cur_pair = title_check()

    if st.session_state.output_dict == {}:
        results.empty()
    else:
        with results.expander('Results', expanded=True):

            st.write(
                """
                #### Title:
                """)
            
            
            
            st.write(st.session_state.cur_pair[0])


            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.button("See Previous Title", on_click=PT_button_clicked, use_container_width=True)

            with t_col2:
                st.button("See Next Title", on_click=NT_button_clicked, use_container_width=True)

            st.write(
                """
                ####  Description:
                """)
            st.write(st.session_state.cur_pair[1].replace('$','\$'))
            
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.button("See Previous Description", on_click=PD_button_clicked, use_container_width=True)

            with d_col2:
                st.button("See Next Description", on_click=ND_button_clicked, use_container_width=True)
                    
def blog():
    """
    Blog describing the Auto-BG project
    """
    with open("BGG Blog MD.md", 'r') as blog_md:
        blog_text = blog_md.read() 
    st.markdown(blog_text)
    st.sidebar.subheader('Auto-BG: The Board Game Concept Generator')
    st.sidebar.write("*This application attempts to augment one step, early in that journey, when the seeds of an idea combine and sprout into a holistic concept.\
                      By interpreting disparate mechanical and descriptive tags to identify a game concept, Auto-BG uses a custom pipeline of GPT3 and T5 models to create a new description and proposed titles for a game that doesn't exist today.\
                      These descriptions support designers-to-be as alternatives to current concepts, seeds for future concepts, or any user as, hopefully, an entertaining thought experiment.*")

def about_us():
    """
    About us page describing creators of Auto-BG
    """
    st.title("About Us")
    st.sidebar.subheader('Creators of Auto-BG')
    st.sidebar.write('*With a shared love of data science and board games, we came together and created Auto-BG as part of our Capstone project\
                     for our "Master of Applied Data Science" program at the University of Michigan.\
                     We hope you enjoy!*')

    # Columns containing information on each of the creators
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        st.image('./NC.jfif', use_column_width=True)
        st.subheader('Nick Canu')
        st.write("""
        **University of Michigan**\n
        *MADS (Master of Applied Data Science)*\n
        """)
    
    with col2:
        st.image('./TD.jfif', use_column_width=True)
        st.subheader('Taylor Druhot')
        st.write("""
        **University of Michigan**\n
        *MADS (Master of Applied Data Science)*\n
        """)

    with col3:
        st.image('./SC.jfif', use_column_width=True)
        st.subheader('Sebastian Capp')
        st.write("""
        **University of Michigan**\n
        *MADS (Master of Applied Data Science)*\n
        """)

page_names_to_funcs = {
    "Application": application,
    "Blog": blog,
    "About Us": about_us, 
}

demo_name = st.sidebar.selectbox("Choose a page:", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

