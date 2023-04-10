*Auto-BG: The Board Game Concept Generator*

1.  [Introduction]{.underline}

    a.  [Goals Statement - design aid/improved user control over
        > generation]{.underline}

**A Gentle Introduction to Auto-BG & Board Game Data**

**What is Auto-BG?**

How does a board game transform from an idea to physically sitting on
your table?

This application attempts to augment one step, early in that journey,
when the seeds of an idea combine and sprout into a holistic concept. By
interpreting disparate mechanical and descriptive tags to identify a
game concept, Auto-BG uses a custom pipeline of GPT3 and T5 models to
create a new description and proposed titles for a game that doesn't
exist today. These descriptions support designers-to-be as alternatives
to current concepts, seeds for future concepts, or any user as,
hopefully, an entertaining thought experiment.

While, overall, ChatGPT solves this application case by generating
coherent descriptions from sentence-formed prompts, we believe this
design niche would be better served by a dedicated application
leveraging domain-specific transfer learning and granular control
through an extensive tag-prompt framework.

Before digging into the process of generating a board game concept,
let's ground some key terms:

-   *Mechanical* - The gameplay mechanics which, together, create a
    > distinct ruleset for a given game. Represented here by a class of
    > tags each capturing individual mechanics such as "Worker
    > Placement" or "Set Collection".

    -   *Cooperative* - Our application, and the dataset, singles out
        > this mechanic as being important enough to have its own tag
        > class. As an alternative to the domain-default of
        > *Competitive* gaming, *Cooperative* represents both an
        > individual mechanic and an entire design paradigm shift.

-   *Descriptive* - In this context, the narrative, production, or
    > family-connective elements of a game. This can include genre,
    > categorical niches, or any relational tag not captured by the
    > class of mechanical tags described above.

    a.  [Orientation to the data - understanding how assembled tags
        > define a game item]{.underline}

**Understanding the Data**

To train our models, we utilized a processed dataset of 20,769 ranked
board games scraped from the board game database & forum
[[BoardGameGeek.com]{.underline}](https://boardgamegeek.com/)^1^. Each
game includes its name, description, and 3,800 feature tags separated
into five classes including Cooperative, Game Type, Category, Mechanic,
and Family; when you select tags within Auto-BG, you choose and assign
them within these distinct classes. To be ranked, these games must have
received a lifetime minimum of thirty user ratings; non-standalone
expansion and compilation game items have also been removed to reduce
replication in the training data.

As previous studies^2^ identified a disproportionate bias toward english
language descriptions, our approach used a soft pass to remove games
identified as having non-english descriptions using langdetect^3^. We
also implemented a check on all remaining titles to remove any that used
non-english characters. In total, this purged 1,423 or \~6.4% of the
data.

The transformed data powering AutoBG was originally scraped via the BGG
XML API by \@mshepard on GitLab^4^ and customized for The Impact of
Crowdfunding on Board Games (2022)^5^ by N. Canu, A. Oldenkamp, and J.
Ellis & Deconstructing Game Design (2022)^2^ by N. Canu, K. Chen, and R.
Shetty. That work, and Auto-BG, are derivatively licensed under
[[Creative Commons
Attribution-NonCommercial-ShareAlike]{.underline}](http://creativecommons.org/licenses/by-nc-sa/3.0/).
Our GitHub repository includes a python script collating the processing
steps required to turn a raw scraper output file into the final data
constructs needed to run an instance of Auto-BG.

b.  [Ethical considerations in domain - inherited biases from both
    > generative model and BGG framework, role of deep learning as
    > supplement/replacement]{.underline}

**Ethical Considerations**

As a chaotician once said "your scientists were so preoccupied with
whether or not they could, they didn\'t stop to think if they
should"^6^ - but that's not what we're going to do. So before we
continue, a few notes regarding potential biases and concerns when
training or using Auto-BG.

First and foremost, Auto-BG inherits critical biases from the base GPT3
Curie model. Extensive work exists documenting the model family's biases
related to race^7^, gender^7,8^, religion^9^, and abstracted moral
frameworks on right and wrong^10^. A key vector for producing biased
output from these models comes from the conditioning context (i.e. user
input) which prompts a response^11^; while these inherited biases can't
be eliminated, AutoBG attempts to reduce their impact on output text
through strictly controlling the conditioning context by relating
unknown input to pre-approved keys.

Additionally, Auto-BG has inherited significant bias in generating
gender markers within descriptive text from the BoardGameGeek training
data. As seen in figure 1, game descriptions include a disproportionate
volume of male-associated gender markets in the source data. This
transfers through Auto-BG's pipeline, resulting in output that, as in
figure 2, retains a similar distribution. While outside the feasible
scope of this project, future iterations of Auto-BG should include
coreference cleaning on the training input to minimize inherited bias;
this approach would require a robust review of the source data through
the framework of coreference resolution to avoid further harms by making
inferences that fail to acknowledge the complexities of gender within
this data^12^.

Beyond these substantial issues, with any large language model (LLM)
project, we have to ask - are we harming human actors by deploying this
tool? In 2023, discussion of the prospective gains and potential danger
of LLMs has broken out of the data science domain and represents a
growing global conversation. Academics wrestle with the scope and
limitations of GPT models^13^ while crossover publications debate topics
such as the merits of ChatGPT as an author^14^ and which professions AI
models might replace in the near future^15^.

With these concerns in mind, we designed Auto-BG as a strictly
augmentative tool - a board game is more than its conceptual description
and while Auto-BG may suggest rule frameworks based on your input, it
can't generate any of the production elements such as full rulesets or
art needed to take the game from concept to table. In addition, we
strongly encourage the use of Auto-BG as a starting point, not a ground
truth generation. The model makes mistakes that a human actor may not;
while we have implemented catches to discourage it from replicating
existing board games, its GPT3 trained core has an extensive knowledge
of gaming in general. We recommend reviewing output text thoroughly
before committing to your new board games \[maybe insert pictures of it
generating these?\] *Age of Empires II: Age of Kings* or *Everquest^©^*.

c.  [Overall generator framework - birds eye graphic/discussion of
    > pipeline]{.underline}

**What's in a pipeline?**

Auto-BG relies on multiple LLMs to generate the full game concept from
your initial input. The entire process from turning your tags into a
prompt to presenting your finished concept is embedded in a pipeline, or
a framework of related code running sequentially, that enables these
distinct steps to work in unison. It all starts with an
idea...![](media/image1.png){width="0.2760422134733158in"
height="0.44166666666666665in"}

1.  The most complex participant in Auto-BG's pipeline (that's you)
    > translates different elements of a vague concept for a game into
    > defined tags through the Auto-BG interface.

2.  An input parser collates the tags, turning them into a game vector
    > (a binary, or one-hot, representation of a game where all selected
    > tags are valued at 1). This vector activates the internal keys
    > representing each tag selected in step 1 and passes them as a
    > prompt to Auto-BG's GPT3 model.

3.  A generator function calls the model through the OpenAI API,
    > returning a response using selected parameters for the input
    > prompt. This response is decoded and lightly cleaned to remove
    > trailing sentences for readability.

4.  The cleaned output is passed downstream to a local t5 trained
    > sequence-to-sequence model, it uses the output as a prompt to
    > translate to potential candidate titles.

5.  Because the GPT3 model has learned that games should include a
    > title, it may include an artificial title already. The title
    > generator runs once and if any candidate titles are identified in
    > the text, it strips them out for a placeholder before running
    > again.

6.  A title selector rejoins the initial and secondary generations,
    > removes duplicates, validates against existing game titles, then
    > scores the cosine similarity of each title with its reference
    > input. It attaches the highest scoring title to the output and
    > fills all placeholders, storing backups of the unchosen titles.

7.  The Auto-BG user interface returns the generated prompt with its
    > title for you to give feedback on, review alternate titles, or
    > save for future use.

At the end of the pipeline, you have a brand-new game concept customized
by your selected tags! And if you're not happy with the output, running
the generator again will produce a new description given the same tags.

To understand Auto-BG better, let's go through each step in detail,
looking at how an example prompt is transformed into its final output.

2.  [Pipeline Ride Along - How does "your" input become a
    > description/title pair?]{.underline}

**The Inner Workings of Auto-BG w/Examples**

a.  *The Input Step*

***Interpreting User Input***

i.  [User input as a "game" & input design - Building off 1b, cover
    > design choices like dividing feature classes, uncaptured features
    > in training, bridge to input selection.]{.underline}

*How do individual tags add up to a game profile?*

This key question guided every aspect of Auto-BGs design principles. The
complex series of transformations from input to generation required
understanding both how a human reader would interact and interpret these
tags but also how the GPT3 LLM at the heart of Auto-BG could best learn
them.

Auto-BG inherits tag classes from BoardGameGeek, if you go to any game
page, you'll see a collection of type, category, mechanic, and family
tags on the sidebar. We decided Auto-BG 1.0 would focus explicitly on
these four primary classes as they translated coherently to formatting
for a GPT3 prompt training structure, could be universally applied to an
unknown game, and don't rely on post-design information like user
recommendations or publisher data.

*What happens when you choose tags in Auto-BG?*

Inside Auto-BG, each tag has a hidden class prefix that denotes it as a
member of that class and tracks its affiliation throughout the pipeline.
This means when you select a new tag, even if it\'s semantically similar
to another overlapping tag in a different class, Auto-BG remembers its
associations; in this approach, Auto-BG effectively handles unknown
inputs by scanning the associated class, matching the unknown tag to its
closest existing sibling.

By grouping all of your selected tags, Auto-BG creates an approximate
profile of a game to generate. When generating your new concept, Auto-BG
tries to infer additional features based on the tags provided. Some
features like player count, age rating, and playtime may still show up
in your concept if Auto-BG thinks they should be included in the text.
By changing even a single tag out, Auto-BG will try to accommodate that
new design feature and generate a different concept!

With this sensitivity to the prompt selection, users must understand how
their choices impact downstream generation. Auto-BG needed an intuitive
user experience with coherent tutorials throughout to streamline this
process, alongside some targeted limits on profile creation.

ii. [UI design considerations - Making tags easily accessible w/volume
    > of options, input tutorial design and approach to creating min/max
    > tag caps.]{.underline}

##\# UI Input Discussion

iii. [Vectorizing user inputs - text key to vector transformation and
     > syncing with the ground truth key list. Approach to handling
     > unknown user input in tag selection.]{.underline}

     1.  Converting to final input - Need for a consistent prompt
         > structure and design considerations to improve GPT3 prompt
         > interpretation, and discuss autonomous generation from tags
         > vs free prompt RE user's end goal. \<- joining these together
         > for formatting.

Once you've created your new game profile, Auto-BG translates it into a
format that its LLM can understand for text generation. Ultimately,
Auto-BG tries to convert your profile tags following OpenAI's
recommended best practices for conditional generation prompts^16,17^.
This results in a text prompt version of your tag list with each tag
period stopped like so:\
\
##\# example here\
\
Fine-tune training has taught Auto-BGs model that this specific format
of text prompt should translate to a descriptive paragraph of the
associated tags. To create it, Auto-BG passes your tags through an input
manager that performs several steps in sequence before sending a
polished prompt to the generator.

First, Auto-BG establishes a ground truth key dictionary of all existing
tags. This format allows for iterative updating from future data, as new
tags appear on BoardGameGeek, Auto-BG can update to include them. This
python dictionary includes a key for every tag with the equivalent value
set to 0 - tracking that it does not appear. Eventually, it will change
the values for your selected tags to 1, telling Auto-BG that they appear
and should be added to the prompt, but first, it tries to correct for
unknown inputs.

Each tag not recognized as being present in the key dictionary moves to
a bucket matching its attached prefix. With these marked as unknown, the
input manager implements a within-class comparison between the unknown
tag and all known tags that share its prefix. Auto-BG estimates semantic
relevance between tags through token cosine similarity^18^, comparing
the unknown tag to each candidate with SpaCy's token similarity
method^19^. Once it evaluates this for all candidates, it adds the
highest scoring option for each unknown tag to the remaining input tags.

Auto-BGs underlying model inherits the characteristic sensitivity to
prompt design of LLMs; prompt design often requires significant human
effort, and many approaches have been suggested to address this
challenge. Our design philosophy focused on improving the quality and
control of prompts with goals of reliability, creativity, and coherence
to input, specifically inspired by concepts of metaprompting^20^ and
internal prompt generation within the LLM^21^. Auto-BG leverages
existing human tagging work done within the source data to provide the
generator model with a controllable abstracted prompt instead of a
potentially complex natural language sentence-structured prompt.

To pass the now cleaned input to the generator, Auto-BGs' input manager
converts the pooled tags into a text string as above, with each tag key
turned into a period-stopped sentence. The generator will send this
prompt to a remote model through the OpenAI API and return a matching
description!

b.  *The Generation Step*

***Generating a Description***

i.  Approach to model selection - comparative performance of GPT3 models
    > vs cost, tradeoff with controllable output and quality.

ii. Training methodology - literature discussion on optimizing fine tune
    > LLM training for generation relative to other tasks
    > (classification in particular).

iii. Initial output generation - limited key parameters in API
     > generation, tuning strategy, refining with validation set.

     1.  Challenges w/GPT3 - Discuss sensitivity to prompt construction
         > and changes in keys & difficulty of low volume references for
         > keys w/source data. Outside knowledge polluting output (i.e.
         > using video games as names in text). Testing for prompt
         > memorization (increasing chaos in output for learned
         > keysets).

iv. Output processing - Increased text cleanup for user output tasks,
    > approaches to mitigating potentially subpar generation from GPT 3
    > including embedded titles and excessive referencing from BGG
    > (incorrect publishers/designers included in text), challenge of
    > removing references again.

```{=html}
<!-- -->
```
c.  *The Title Step*

***A Fitting Title***

i.  Text to title relationship - Domain considerations w/title as a
    > product, distinct relationship to train on. Role of title in text
    > as a key point when running models in sequence.

To complete the game concept, Auto-BG provides a selection of titles
chosen to fit the newly generated description. But what defines a good
title? It depends on the domain and it depends on the product! A sense
of tension may fit a thriller novel^22^ and sticking to positive
associations enhances commercial products^23^, but where do board games
fit in? To create titles that both fit their description and matched
domain trends as a whole, we needed a model that could learn the unique
relationship between title and description in board gaming.

ii. Transformation approach - seq2seq advantages for this task & model
    > selection, advantages of extending transfer learning from domain
    > to domain (news headlines to game titles)

While our generator uses the prompt-based GPT paradigm, we decided to
take a different approach for title generation inside Auto-BG. T5 models
fit within a unified framework for transfer learning known as
sequence-to-sequence or text-to-text^24^; at their core, they're an
encoder-decoder model operating by treating all NLP operations as a
translation task. This means that given text as input, the model will
attempt to output a predicted target that best fits the training data.

Transfer learning, in machine learning, applies a model trained on one
task or domain and performs additional training to apply the model to a
different task. The total corpus of descriptions for BGG, approximately
20,000 descriptions, is inadequate for fresh training of a generative
NLP model; instead Auto-BG leverages two-stages of upstream transfer
learning through HuggingFace's Transformer library. The final
implementation extends a model^25^ already fine-tuned on an additional
500,000 news articles from t5-base.

1.  Relative model performance - approach to model selection based on
    > above including comparative metrics. Discuss aligning metrics
    > w/goal of tool + need for human review.

To scope the best generator for Auto-BG, we trained a sequence of models
on both t5-base^26^ and the headline-trained model. Beyond training
baseline models, we looked toward work by Tay, et al.^27^ on scaling and
fine-tune training for transformers to guide selecting training
parameters; the final round of testing utilized a higher learning rate
of .001 while introducing weight decay in the optimizer.

With several models trained

iii. Low-cost iterative generation - two-pass competitive scoring on
     > multi-output, using the title generator to select and replace
     > low-quality titles then scoring the maximum pool of two
     > generations.

     1.  Scoring approach - title output cleanup and scoring final title
         > to body text semantic similarity.

```{=html}
<!-- -->
```
d.  *Final Output*

    i.  Rerunning in real-time - approaches to adjusting output
        > including updating text references, finding text generator
        > default "profiles" through hyperparameter testing, and
        > implementation of update processes in the UI.

    ii. The value of user feedback - considerations for future
        > performance w/tuning generation settings, content reporting
        > for title cleanup or problematic text content.

```{=html}
<!-- -->
```
3.  [Discussion & Conclusion]{.underline}

    a.  What's the point? - Practical design tool / Interpreting how
        > "mechanical" changes interact with language descriptions /
        > Extending user interactivity beyond prompts

    b.  Discuss methodology - user interactivity through prompt control,
        > did it work & other potential approaches to same application
        > (reinforcement, different prompt structures)

    c.  Alternative applications - fine-tune human-written desc
        > (training on extended prompt + more deterministic output),
        > rules generation?

    d.  Future extensions/limitations - Expand on language scope, live
        > data + training, discarded feature classes / discussion on
        > approach not including these currently

    e.  What next? - Encourage readers to experiment with tool + give
        > feedback in app

4.  [Extras]{.underline}

    a.  Work Statement + Appendices + Citations

**References**

1.  "BoardGameGeek." Accessed March 19, 2023.
    > [[https://boardgamegeek.com/]{.underline}](https://boardgamegeek.com/).

2.  Canu, Nicholas, Kuan Chen, and Rhea Shetty. "Deconstructing Game
    > Design." Jupyter Notebook, October 20, 2022.
    > [[https://github.com/canunj/deconstructing_games]{.underline}](https://github.com/canunj/deconstructing_games).

3.  Danilák, Michal. "Langdetect." Python, March 15, 2023.
    > [[https://github.com/Mimino666/langdetect]{.underline}](https://github.com/Mimino666/langdetect).

4.  GitLab. "Recommend.Games / Board Game Scraper · GitLab," March
    > 10, 2023.
    > [[https://gitlab.com/recommend.games/board-game-scraper]{.underline}](https://gitlab.com/recommend.games/board-game-scraper).

5.  Canu, Nicholas, Jonathan Ellis, and Oldenkamp, Adam. "The Impact of
    > Crowdfunding on Board Games." Jupyter Notebook, May 20, 2022.
    > [[https://github.com/canunj/BGG_KS_Analysis]{.underline}](https://github.com/canunj/BGG_KS_Analysis/blob/92452d7e7b174bf45763e469b1ed4ce61a84b7ba/The%20Impact%20of%20Crowdfunding%20on%20Board%20Games.pdf).

6.  Spielberg, Steven dir. *Jurassic Park*. 1993; Universal City, CA:
    > Universal Studios, 2022. UHD Blu Ray.

7.  Chiu, Ke-Li, Annie Collins, and Rohan Alexander. "Detecting Hate
    > Speech with GPT-3." arXiv, March 24, 2022.
    > [[https://doi.org/10.48550/arXiv.2103.12407]{.underline}](https://doi.org/10.48550/arXiv.2103.12407).

8.  Lucy, Li, and David Bamman. "Gender and Representation Bias in GPT-3
    > Generated Stories." In *Proceedings of the Third Workshop on
    > Narrative Understanding*, 48--55. Virtual: Association for
    > Computational Linguistics, 2021.
    > [[https://doi.org/10.18653/v1/2021.nuse-1.5]{.underline}](https://doi.org/10.18653/v1/2021.nuse-1.5).

9.  Abid, Abubakar, Maheen Farooqi, and James Zou. "Persistent
    > Anti-Muslim Bias in Large Language Models." In *Proceedings of the
    > 2021 AAAI/ACM Conference on AI, Ethics, and Society*, 298--306.
    > AIES '21. New York, NY, USA: Association for Computing
    > Machinery, 2021.
    > [[https://doi.org/10.1145/3461702.3462624]{.underline}](https://doi.org/10.1145/3461702.3462624).

10. Schramowski, Patrick, Cigdem Turan, Nico Andersen, Constantin A.
    > Rothkopf, and Kristian Kersting. "Large Pre-Trained Language
    > Models Contain Human-like Biases of What Is Right and Wrong to
    > Do." *Nature Machine Intelligence* 4, no. 3 (March 2022): 258--68.
    > [[https://doi.org/10.1038/s42256-022-00458-8]{.underline}](https://doi.org/10.1038/s42256-022-00458-8).

11. Huang, Po-Sen, Huan Zhang, Ray Jiang, Robert Stanforth, Johannes
    > Welbl, Jack Rae, Vishal Maini, Dani Yogatama, and Pushmeet Kohli.
    > "Reducing Sentiment Bias in Language Models via Counterfactual
    > Evaluation." arXiv, October 8, 2020.
    > [[https://doi.org/10.48550/arXiv.1911.03064]{.underline}](https://doi.org/10.48550/arXiv.1911.03064).

12. Cao, Yang Trista, and Hal Daumé III. "Toward Gender-Inclusive
    > Coreference Resolution." In *Proceedings of the 58th Annual
    > Meeting of the Association for Computational Linguistics*,
    > 4568--95, 2020.
    > [[https://doi.org/10.18653/v1/2020.acl-main.418]{.underline}](https://doi.org/10.18653/v1/2020.acl-main.418).

13. Floridi, Luciano, and Massimo Chiriatti. "GPT-3: Its Nature, Scope,
    > Limits, and Consequences." *Minds and Machines* 30, no. 4
    > (December 1, 2020): 681--94.
    > [[https://doi.org/10.1007/s11023-020-09548-1]{.underline}](https://doi.org/10.1007/s11023-020-09548-1).

14. Thorp, H. Holden. "ChatGPT Is Fun, but Not an Author." *Science*
    > 379, no. 6630 (January 27, 2023): 313--313.
    > [[https://doi.org/10.1126/science.adg7879]{.underline}](https://doi.org/10.1126/science.adg7879).

15. Zinkula, Aaron Mok, Jacob. "ChatGPT May Be Coming for Our Jobs. Here
    > Are the 10 Roles That AI Is Most Likely to Replace." Business
    > Insider. Accessed March 19, 2023.
    > [[https://www.businessinsider.com/chatgpt-jobs-at-risk-replacement-artificial-intelligence-ai-labor-trends-2023-02]{.underline}](https://www.businessinsider.com/chatgpt-jobs-at-risk-replacement-artificial-intelligence-ai-labor-trends-2023-02).

16. "Fine-Tuning." Accessed March 23, 2023.
    > [[https://platform.openai.com/docs/guides/fine-tuning/conditional-generation]{.underline}](https://platform.openai.com/docs/guides/fine-tuning/conditional-generation).

17. "Best Practices for Prompt Engineering with OpenAI API \| OpenAI
    > Help Center." Accessed March 26, 2023.
    > [[https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api]{.underline}](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api).

18. Gunawan, D., C. A. Sembiring, and M. A. Budiman. "The Implementation
    > of Cosine Similarity to Calculate Text Relevance between Two
    > Documents." *Journal of Physics: Conference Series* 978, no. 1
    > (March 2018): 012120.
    > [[https://doi.org/10.1088/1742-6596/978/1/012120]{.underline}](https://doi.org/10.1088/1742-6596/978/1/012120).

19. Token. "Token · SpaCy API Documentation." Accessed March 26, 2023.
    > [[https://spacy.io/api/token#similarity]{.underline}](https://spacy.io/api/token#similarity).

20. Reynolds, Laria, and Kyle McDonell. "Prompt Programming for Large
    > Language Models: Beyond the Few-Shot Paradigm." arXiv, February
    > 15, 2021.
    > [[https://doi.org/10.48550/arXiv.2102.07350]{.underline}](https://doi.org/10.48550/arXiv.2102.07350).

21. Zhou, Yongchao, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster,
    > Silviu Pitis, Harris Chan, and Jimmy Ba. "Large Language Models
    > Are Human-Level Prompt Engineers." arXiv, March 10, 2023.
    > [[https://doi.org/10.48550/arXiv.2211.01910]{.underline}](https://doi.org/10.48550/arXiv.2211.01910).

22. MasterClass. "How to Write a Book Title in 7 Tips: Create the Best
    > Book Title - 2023." Accessed March 30, 2023.
    > [[https://www.masterclass.com/articles/how-to-write-a-book-title-in-7-tips-create-the-best-book-title]{.underline}](https://www.masterclass.com/articles/how-to-write-a-book-title-in-7-tips-create-the-best-book-title).

23. Qualtrics. "How to Find the Perfect Product Name in 2023." Accessed
    > March 30, 2023.
    > [[https://www.qualtrics.com/experience-management/product/product-naming/]{.underline}](https://www.qualtrics.com/experience-management/product/product-naming/).

24. Raffel, Colin, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    > Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu.
    > "Exploring the Limits of Transfer Learning with a Unified
    > Text-to-Text Transformer." arXiv, July 28, 2020.
    > [[http://arxiv.org/abs/1910.10683]{.underline}](http://arxiv.org/abs/1910.10683).

25. "Michau/T5-Base-En-Generate-Headline · Hugging Face." Accessed March
    > 30, 2023.
    > [[https://huggingface.co/Michau/t5-base-en-generate-headline]{.underline}](https://huggingface.co/Michau/t5-base-en-generate-headline).

26. "T5-Base · Hugging Face," November 3, 2022.
    > [[https://huggingface.co/t5-base]{.underline}](https://huggingface.co/t5-base).

27. Tay, Yi, Mostafa Dehghani, Jinfeng Rao, William Fedus, Samira Abnar,
    > Hyung Won Chung, Sharan Narang, Dani Yogatama, Ashish Vaswani, and
    > Donald Metzler. "Scale Efficiently: Insights from Pre-Training and
    > Fine-Tuning Transformers." arXiv, January 30, 2022.
    > [[https://doi.org/10.48550/arXiv.2109.10686]{.underline}](https://doi.org/10.48550/arXiv.2109.10686).

28. 
