import PySimpleGUI as sg
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

def var_saver(v_x, n_val, n_var):
    if v_x < len(n_var):
            try:
                n_var[v_x] = values[n_val+str(v_x)]
            except:
                exit
    else:
        n_var.append(values[n_val+str(v_x)])


def state_saver(c_fam, c_gam, c_cat, c_mec, coop):
    for x in range(c_fam+1):
        var_saver(x, 'fam', family_v)
    for x in range(c_gam+1):
        var_saver(x, 'game', game_v)
    for x in range(c_cat+1):
        var_saver(x, 'cat', cat_v)
    for x in range(c_mec+1):
        var_saver(x, 'mech', mech_v)
    coop = values['CoopCheck']
        
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


sg.theme('Reddit')  # Let's set our own color theme

#demo
slim_df = pd.read_parquet('https://github.com/canunj/Auto-BoardGame/blob/main/Model_Step_Data/slim_df.parquet.gzip?raw=true')
search_tokens = token_expand("https://github.com/canunj/Auto-BoardGame/blob/main/Persistent%20Objects/token_search.gz?raw=true")
vector_df = pd.read_parquet('https://github.com/canunj/Auto-BoardGame/blob/main/Model_Step_Data/vector_df.parquet.gzip?raw=true')
exp = 'The sequel to Scythe sends players on a new adventure into Siberia, where a massive meteorite crashed near the Tunguska River, awakening ancient corruption. An expedition led by Dr. Tarkovsky ventures into the taiga to learn about the meteorite and its impact on the land. Itching for adventure, heroes from the war privately fund their own expeditions to Siberia, hoping to find artifacts, overcome challenges, and ultimately achieve glory. Expeditions is a competitive, card-driven, engine-building game of exploration. Play cards to gain power, guile, and unique worker abilities; move your mech to mysterious locations and gain cards found among the tiles; use workers, items, meteorites, and quests to enhance your mech; and use power and guile to vanquish corruption.'
san = 'Sankokushin: Five Sacrifices is a fully cooperative campaign game set in a fictional version of medieval Japan. Take control of several heroes arriving to assist the city-state of Yamashiro, a land destroyed by the quest for the Unseen, and help the city survive its grim destiny. Experience an adventure in which you’re free to decide what activities the heroes pursue, where they explore, who they meet and how they interact with their opponents; all of which organically generates a unique storyline inside of the main plot that moves the destiny of Yamashiro. Fight monsters and human foes in psycologically-intense encounters where personality and tactical manuevering are paramount. Intimidate or provoke your enemies to scare them or force them to attack carelessly. Use your diplomatic abilities to calm dangerous, psychopathic entities until you are able to backstab them or plead on the behalf of honorable opponents. Manage the stress of battle, seek to understand your opponents’ perverse intelligence and use that knowledge to destroy them or to find a way to save them from destroying themselves and everyone you hold dear. Welcome to Yamashiro. Campaign The city of Yamashiro is an immense micro-universe where the lives of the citizens, heroes and enemies are tightly intertwined. Experience a unique adventure that unfolds over the course of 12 months during which you decide what activities the heroes pursue, where they explore and who they meet. Craft equipment, revel in festivals, expand the city, take advantage of the amenities of its many facilities, and explore personal relationships with the town’s inhabitants as well as its assailants. Sankokushin does not require a heavy, multi-player commitment to reach the end of a campaign and it will not abruptly end if things go poorly. However, only those who travel across time and memory will gain the knowledge necessary to free Yamashiro from the burden of its tragic fate. Procedural Narrative No walls of text. No “Yes/No” choices. Sankokushin instead wraps you in its unique story as you experience Yamashiro without confining you to a guided plotline. Instead of presenting you with only narrative choices, the storyline is organically generated by the heroes’ actions during both the combat encounters and the city phase. The assailants of Yamashiro and its inhabitants are all intimately connected by a dark fate and only the actions of the heroes can alter the course of their destiny. These interrelationships have profound consequences in game and they are the heart of Sankokushin’s narrative and it is impossible to experience them all in a single playthrough as they are a reflection of your unique playstyle so different gaming groups will have different experiences. Intelligent Enemies Each of the 10 bosses included in Sankokushin has a unique personality and story arc that is closely linked with the overall narrative. The way you deal with each of them will reverberate throughout the course of the campaign. You will learn to love and hate their twisted personalities and will see that the methods used to fight them will deeply influence their future behavior. They learn from your strategies and change their attitude towards you (and the way they fight!) according to the way you psychologically approach them thanks to the unique communication system. Sankokushin is designed by a small team whose knowledge in the fields of psychology and psychoanalysis have made possible bosses with realistic, uncanny, and perverse personalities that will show you the darkest sides of human nature.'
sky = "The Mayor has enlisted some of humanity's greatest visionaries to help build Skyrise: a magnificent city in the sky, dedicated to art, science, and beauty. But only one artisan can be remembered as the greatest! Prove your genius by using a brilliant spatial auction system to win sites to build in, earning favor across Islands and factions, scoring secret and public objectives, courting mysterious Patrons, and building your own unique Wonder. Skyrise tells an interactive story of a growing cityscape, hard decisions, scarce resources, and ever-rising stakes that will keep you enthralled until the final scores are revealed. Skyrise is a brand new game inspired by Sebastien Pauchon's 2008 Spiel Des Jahres recommended Metropolys."
Tgen = Title_Generator('./t5_model', slim_df)

category_keys = reader("https://github.com/canunj/Auto-BoardGame/blob/main/Persistent%20Objects/current_keys.gz?raw=true")

ex_check = ["[Ee]verquest","[Cc]ivilization [Ii][IiVv]","[Cc]ivilization(?=:)","[Cc]ivilization [Ii][Ii]",
            "[Cc]ivilization [Ii][Ii][Ii]","[Aa]ge [Oo]f [Ee]mpires [Ii][Ii2]([Ii]|\b)", "[Rr]avenloft|[Cc]astle [Rr]avenloft",
            "[Ss]cythe(?=:|\b)","[Dd]ungeons [&Aa][ n][Dd ][ Ddr][Ddra][rg][oa][gn][os](ns|\b)",
            "[Aa]ge [Oo]f [Ee]mpires [Ii][Ii]: [Tt]he [Aa]ge [Oo]f [Kk]ings","[Aa]ge [Oo]f [Ee]mpires 2: [Tt]he [Aa]ge [Oo]f [Kk]ings",
            "[Aa]ge [Oo]f [Ee]mpires"] 

iman = input_manager(vector_df, slim_df, search_tokens)

#cooperative?, Gametype, Mechanic, Category, family
family_c = 0
game_c = 0
cat_c = 0
mech_c = 0

family_v = ['Choose a Type!']
game_v = ['Choose a Type!']
cat_v = ['Choose a Type!']
mech_v = ['Choose a Type!']
coop = 0

input = []

title = 'Auto-BG: The Game Concept Generator'
main_text = "Discover the concept for your next favorite game!\n\nTake your ideas and turn them into a full-fledged tabletop game concept through the power of deep learning! \n\nHow do you use Auto-BG? \nPick any set of choices from four selectors below: Family, Game, Mechanic, and Category.\nThen, if you are looking to lose together, activate the toggle for a cooperative game!\n\n Need more detail?\n\nFamily: Descriptive niches for grouping games\nGame Type: Top level genres - Family, Strategy, etc.\nMechanic: Game rules!\nCategory: The primary genres\n\nNeed a guided walkthrough or want to see how Auto-BG performs on a known game?\nPress the lifepreserver button in the top left corner!" 

#Step 0 additive/subtractive layout

def FamilyBox(x):
    print(x, family_c, family_v)
    if x == family_c and family_c == 0:
        a = [sg.Button('+', key='+family', size=(1,1)), sg.Combo(category_keys[4], default_value=family_v[x], key='fam'+str(x), size=(58,1))]
    elif x == 6 and family_c == 6:
        a = [sg.Button('-', key='-family', size=(58,1))]
    elif x == family_c:
        a = [sg.Button('+', key='+family', size=(1,1)), sg.Combo(category_keys[4], default_value=family_v[x], key='fam'+str(x), size=(58,5)), sg.Button('-', key='-family', size=(1,1))]
    else:
        a = [sg.Combo(category_keys[4], default_value=family_v[x], key='fam'+str(x), size=(65,5))]
    return a

def GameBox(x):
    if x == game_c and game_c == 0:
        a = [sg.Button('+', key='+game',size=(1,1)), sg.Combo(category_keys[1], default_value=game_v[x], key='game'+str(x), size=(58,1))]
    elif x == 2 and game_c == 2:
        a = [sg.Button('-', key='-game', size=(58,1))]
    elif x == game_c:
        a = [sg.Button('+', key='+game', size=(1,1)), sg.Combo(category_keys[1], default_value=game_v[x], key='game'+str(x), size=(58,5)), sg.Button('-', key='-game', size=(1,1))]
    else:
        a = [sg.Combo(category_keys[1], default_value=game_v[x], key='game'+str(x), size=(65,5))]
    return a

def CatBox(x):
    if x == cat_c and cat_c == 0:
        a = [sg.Button('+', key='+cat',size=(1,1)), sg.Combo(category_keys[3], default_value=cat_v[x], key='cat'+str(x), size=(58,1))]
    elif x == 3 and cat_c == 3:
        a = [sg.Button('-', key='-cat', size=(58,1))]
    elif x == cat_c:
        a = [sg.Button('+', key='+cat', size=(1,1)), sg.Combo(category_keys[3], default_value=cat_v[x], key='cat'+str(x), size=(58,5)), sg.Button('-', key='-cat', size=(1,1))]
    else:
        a = [sg.Combo(category_keys[3], default_value=cat_v[x], key='cat'+str(x), size=(65,5))]
    return a

def MechBox(x):
    if x == mech_c and mech_c == 0:
        a = [sg.Button('+', key='+mech', size=(1,1)), sg.Combo(category_keys[2], default_value=mech_v[x], key='mech'+str(x), size=(58,1))]
    elif x == 5 and mech_c == 5:
        a = [sg.Button('-', key='-mech', size=(58,1))]
    elif x == mech_c:
        a = [sg.Button('+', key='+mech', size=(1,1)), sg.Combo(category_keys[2], default_value=mech_v[x], key='mech'+str(x), size=(58,5)), sg.Button('-', key='-mech', size=(1,1))]
    else:
        a = [sg.Combo(category_keys[2], default_value=mech_v[x], key='mech'+str(x), size=(65,5))]
    return a

# STEP 1 define the layout

def make_window():

    layout_title = [sg.Text(title, expand_x=True, font='Helvetica 25', justification='c')]

    layout_main_text = [sg.Text(main_text, expand_x=True, font='Helvetica 10', justification='c')]

    layout_top_controls = sg.Frame('Inputs',  [
                    [sg.Col([FamilyBox(x) for x in range(0,family_c+1)], vertical_alignment='bottom'),
                    sg.Col([GameBox(x) for x in range(0,game_c+1)], vertical_alignment='bottom'),
                    sg.Col([CatBox(x) for x in range(0,cat_c+1)], vertical_alignment='bottom'),
                    sg.Col([MechBox(x) for x in range(0,mech_c+1)], vertical_alignment='bottom')],
                    [sg.Checkbox('Cooperative?', key='CoopCheck', default=coop)],
                    [sg.Button('Run')]

                ], border_width=2, font='Helvetica 18', expand_x=True, title_location='n', element_justification ='c')
    
    space0 = [sg.Text('_'*100, expand_x=True, font='Helvetica 10', justification='c')]

    gen_title =  sg.Frame('Title',
                          [
                    [sg.Text(key='-Title_OUTPUT-', expand_x=True, font='Helvetica 45', justification='c')],
                    [sg.Col([[sg.Button('Show Me The Previous Title'), sg.Text(' ' * 100), sg.Button('Show Me The Next Title')]], justification='c')]

                ], border_width=1, font='Helvetica 18', expand_x=True, title_location='n', element_justification ='c')

    gen_desc=  sg.Frame('Description',
                          [
                    [sg.Text(key='-Desc_OUTPUT-', expand_x=True, font='Helvetica 18', justification='c')],
                    [sg.Col([[sg.Button('Show Me The Previous Description'), sg.Text(' ' * 100), sg.Button('Show Me The Next Description')]], justification='c')]

                ], border_width=1, font='Helvetica 18', title_location='n', element_justification ='c')

    layout_bottom = [ 
                [sg.Text('Description Test')],
                [sg.Button('Get Title')],
                [sg.Input(key='-IN-')],
                [sg.Text('Description Defaults')],
                [sg.Button('Expeditions'), sg.Button('Sankokushin: Five Sacrifices'), sg.Button('Skyrise')], 
                [sg.Button('Exit')],
            ]

    layout = [
               [layout_title] + [space0] + [layout_main_text] + [[layout_top_controls]] + [[gen_title]] + [[gen_desc]]
            ]

    #STEP 2 - create the window
    window = sg.Window('Board Game Description and Title Generator', layout, margins=(1,1), element_justification='center')

    return window

# STEP 3 - Start Window

window = make_window()

# STEP 4 - the event loop
while True:

    event, values = window.read()   # Read the event that happened and the values dictionary

    state_saver(family_c, game_c, cat_c, mech_c, coop)

    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':     # If user closed window with X or if user clicked "Exit" button then exit
      break
    
    if  event == 'Run':
        if input  == revert_cats(game_v, mech_v, cat_v, family_v, coop):
            print('Already  Ran')
        else:
            title_next =  0
            input = revert_cats(game_v, mech_v, cat_v, family_v, coop)
            ks = iman.input_parser(iman.set_input(input))
            mctrl = model_control(mc.SEND_KEY())
            mctrl.prompt_formatter(ks)
            desc = mctrl.call_api(status='1st')
            clean_desc = mctrl.resp_cleanup(desc)
            inter_pair = Tgen.candidate_generator(clean_desc)
            Tgen.candidate_score(inter_pair,ex_check)
            output = Tgen.title_check()
            print(desc, title[title_next])
            window['-Title_OUTPUT-'].update(output[0])
            window['-Desc_OUTPUT-'].update(output[1])


    if event == 'Expeditions':
        next = 0
        title = Tgen.candidate_generator(exp)[0]
        window['-Title_OUTPUT-'].update(title[next])

    if event == 'Sankokushin: Five Sacrifices':
        next = 0
        title = Tgen.candidate_generator(san)[0]
        window['-Title_OUTPUT-'].update(title[next])

    if event == 'Skyrise':
        next = 0
        title = Tgen.candidate_generator(sky)[0]
        window['-Title_OUTPUT-'].update(title[next])

    if event == 'Show Me The Next Title':
        output = Tgen.title_check(next=1)
        window['-Title_OUTPUT-'].update(output[0])
        window['-Desc_OUTPUT-'].update(output[1])
    
    if event == 'Show Me The Previous Title':
        output = Tgen.title_check(next=-1)
        window['-Title_OUTPUT-'].update(output[0])
        window['-Desc_OUTPUT-'].update(output[1])

    if event == 'Get Title':
        next = 0
        title = Tgen.candidate_generator(values['-IN-'])[0]
        window['-Title_OUTPUT-'].update(title[next])

    if event == '+family':
        if family_c < 6:
            family_c += 1
            family_v.append('Choose a Type!')
        window.close()
        window = make_window()

    if event == '-family':
        family_c -= 1
        family_v.pop()
        window.close()
        window = make_window()

    if event == '+game':
        if game_c < 2:
            game_c += 1
            game_v.append('Choose a Type!')
        window.close()
        window = make_window()

    if event == '-game':
        game_c -= 1
        game_v.pop()
        window.close()
        window = make_window()

    if event == '+cat':
        if cat_c  < 3:
            cat_c += 1
            cat_v.append('Choose a Type!')
        window.close()
        window = make_window()

    if event == '-cat':
        cat_c -= 1
        cat_v.pop()
        window.close()
        window = make_window()

    if event == '+mech':
        if mech_c < 5:
            mech_c += 1
            mech_v.append('Choose a Type!')
        window.close()
        window = make_window()

    if event == '-mech':
        mech_c  -= 1
        mech_v.pop()
        window.close()
        window = make_window()

window.close()
