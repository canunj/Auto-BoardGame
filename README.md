[icon banner image placeholder]

# Auto-BG
LLM-based text generation tool for creating board game concepts (description & title)

The Auto-BG (Board Game) tool is a text generation tool for creating board game concepts. It utilizes multiple large-language models to generate board game titles and descriptions tailored from user-input tags based on BoardGameGeek.com. The models used in this project include a trained T5 sequence-to-sequence model, primarily for title generation, and a robust GPT3 model for board game description generation. The T5 model was initially presented by Raffel et al. in ["Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"](https://arxiv.org/pdf/1910.10683.pdf). The GPT3 model builds from Brown et al.'s work in ["Language Models are Few-Shot Learners"](https://arxiv.org/pdf/1910.10683.pdf).


## Table of Contents
- Features and Demo
- Examples
- Project Structure
- Customizing Auto-BG
- Citations and Licensing

## Features and Demo
The main features of this application include:

A user-friendly interface for Auto-BG can be found at (homepage).

## Examples

## Project Structure

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
7. 6a. Auto-BG uses a Curie model with a lowered learning rate running for fewer epochs.
8. Create a Model_Constants.py file with your personal API key and model instance based on the template above.
9. You now have a customized instance of Auto-BG!

## Citations and Licensing
Auto-BG is licensed under CC BY-NC-SA 2.0, original data sourced from Recommend.Games @GitLab
