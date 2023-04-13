# Auto-BG
LLM-based text generation tool for creating board game concepts (description & title)

Auto-BG (Board Game) is a text generation tool for creating board game concepts. It utilizes multiple large-language models to generate board game titles and descriptions tailored from user-input tags based on BoardGameGeek.com. The models used in this project include a trained T5 sequence-to-sequence model for title generation and a fine-tuned Curie-base GPT3 model for description generation. 


The T5 model was initially presented by Raffel et al. in ["Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"](https://arxiv.org/pdf/1910.10683.pdf). The GPT3 model builds from Brown et al.'s work in ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165).


## Features
The main features of this application include:

- Intuitive text generation through an easy to use tag selector.
- Suggested title generation with multiple options for every prompt.
- Demos based on the tags for three popular games to orient users in Auto-BG.

The user-friendly interface for Auto-BG can be found at https://huggingface.co/spaces/AutoBG/Auto-BoardGame.

Auto-BG has confirmed compatability with Chrome, Firefox, & Edge along with their mobile equivalents. Other interfaces should work but aren't tested.

## Project Structure

### Required Files
Auto-BG's required files consist of:

3 Python files-

Home.py - The Streamlit app framework and runs all generation functions through a central script.
description_generator.py - The input manager and text generation model control classes.
title_generator.py - The title generation model control and title generation/cleanup classes.

t5_model folder containing all required configuration and model files to run our trained title generator.

Persistent_Data folder containing 4 data files -

slim_df.parquet.gzip - Game data including name, description, tokenized descriptions, and language tag.
vector_df.parquet.gzip - Game data including only the one-hot tag vectors.

current_keys.gz - List object containing an up-to-date ground truth key set for tags.
token_search.gz - SpaCy preprocessed tokens for all existing keys, required for matching unknown keys efficiently.

### Running Auto-BG
Must be loaded into a properly configured Streamlit space or a local repo with a secrets.toml file. Secrets must include a "key" variable with a valid OpenAI API key and a "model" variable with a valid OpenAI model reference value.

1. Run Home.py

That's it!

Home.py will open Streamlit in your browser, load all data files, and configure the model control classes. When tags are selected and "Run Model" is chosen, Home.py will call description_generator and title_generator in sequence then return the output dictionary with titles and descriptions in the UI.

## Customizing Auto-BG
NOTE: Auto-BG uses a fine-tuned GPT-3 Curie model that will be inaccessible without an organizational API key, 
the below instructions are for advanced users interested in remixing Auto-BG with a new generator model.

In order to run this application, you will need the following:
1. An OpenAI account and API key
2. All libraries specified in both the primary and data processing requirements.txt files
3. A raw stream JSON file of BoardGameGeek data, formatted to match output from the Recommend.Games scraper

To implement a new instance of Auto-BG, follow these steps:
1. Clone the repository onto your local machine
2. Install the required packages listed in both 'requirements.txt' files using pip
3. Download the trained T5 model or provide a path to an alternate T5 model.
4. Placing the JSON data file in Stream_to_Output, run GameCleaner.py - this provides all required data files.

5. Prepare training prompts - convert all active keys to period stopped tokens in a string for each game.
6. Fine-tune a selected model following the instructions at: https://platform.openai.com/docs/guides/fine-tuning
NOTE: Auto-BG uses a Curie model with a lowered learning rate running for fewer epochs.
---
7. Create a Model_Constants.py file with your personal API key and model instance based on the template above.

OR

8. Add your API Key and model reference to secrets on your preferred hosting platform, use the Streamlit secrets framework to access values.
---
9. You now have a customized instance of Auto-BG!

## Licensing
Auto-BG is licensed under CC BY-NC-SA 2.0. Data sourced from Recommend.Games scraper @GitLab, all game data is property of BoardGameGeek.com.

## About Us
N. Canu trained, evaluated, & implemented text and title generator models for Auto-BG. Their current board game obsession is Obsession by Kayenta Games.

S. Capp assisted with model research, initial text generator implementation, and tailoring of streamlit application elements. Immense fan of Catan, Coup, and any deduction games.

T. Druhot designed personas, user journeys, and wireframes for the user interface. Turned title model notebook into module and built UI front end and integrated model modules into end user functionality. Avid Magic the Gathering Limited player and deck/engine building board games.

