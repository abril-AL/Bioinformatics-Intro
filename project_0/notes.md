# Project 0

# Chapter 1 - Transformer Models

## NLP and LLMs

NLP
- natural language processing
- goal: understand the context of words in a language


LLMs
- large language models
- scale, in-context learning

## Transformers 

Transformers
- used to solve nlp problems

`pipeline()` function
- connects a model with its necessary preoprocessing and postprocessing steps
- allows us to directly input any text and get an intelligibe answer

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")

[{'label': 'POSITIVE', 'score': 0.9598047137260437}]

# pass several sentences:
classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)

[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]
```
- this pipeline is selecting a pretrained model for sentiment analysis
- model is downloaded and cached when u create the `classifier` object, reruning uses cached model

Three main steps
1. Text preprocessed into format model understands
2. Passes inputs to model
3. Predictions post processed for our understanding 

Zero Shot Classification
- classify unlabeled text
- dont need to fine tune the model on your data to use it - can directly get the probabilities
- allows u to specify labels to use

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

{'sequence': 'This is a course about the Transformers library',
 'labels': ['education', 'business', 'politics'],
 'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}
```


Text Generation
- provide prompt, model auto-completes

```python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")

[{'generated_text': 'In this course, we will teach you how to understand and use '
                    'data flow and data interchange when handling user data. We '
                    'will be working with one or more of the most commonly used '
                    'data flows — data flows of various types, as seen by the '
                    'HTTP'}]
```

Using Any model from the Hub in a pipeline

ex. distilgpt2
```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

[{'generated_text': 'In this course, we will teach you how to manipulate the world and '
                    'move your mental and physical capabilities to your advantage.'},
 {'generated_text': 'In this course, we will teach you how to become an expert and '
                    'practice realtime, and with a hands on experience on both real '
                    'time and real'}]
```

Mask Filling
- fill in the blanks given text
- topk - how many possibilities you want displayed 

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)

[{'sequence': 'This course will teach you all about mathematical models.',
  'score': 0.19619831442832947,
  'token': 30412,
  'token_str': ' mathematical'},
 {'sequence': 'This course will teach you all about computational models.',
  'score': 0.04052725434303284,
  'token': 38163,
  'token_str': ' computational'}]
```

Named Entity Recognition
- model has to find which parts of the input text correspond to entities such as persons, locations, or organization

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

[{'entity_group': 'PER', 'score': 0.99816, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
 {'entity_group': 'ORG', 'score': 0.97960, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
 {'entity_group': 'LOC', 'score': 0.99321, 'word': 'Brooklyn', 'start': 49, 'end': 57}
]
```

Question Answering
- answers given info from a context

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)

{'score': 0.6385916471481323, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
```

Summarization
- reduce text and keep important aspects
```python
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering ...
"""
)

[{'summary_text': ' America has changed dramatically during recent years . The '
                  'number of engineering graduates in the U.S. has declined in '
                  'traditional engineering disciplines such as mechanical, civil '
                  ', ... }]
```

Translation
- provide language pair 

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")

[{'translation_text': 'This course is produced by Hugging Face.'}]
```

## How do Transformers Work

Transformers are language models
- trained on large amts of raw text in self-superviesd fashion

Big models - better performance by increasing model size and amt of data they're pretrained on

Transfer Learning
- Pretraining: training model from scratch, no prior knowledge
	- done on large amts of data
- Fine Tuning: perform additional traingng with a pretrained model, on data specific to the task
	- takes advantage of base knowledge aquired from pretraining
	- thus required less data

General Architecture
- Encoder
	- Recieves Input and builds a representation of its features - model optimized to aquire understanding from the input
- Decoder
	- Sues the encoders representation (features) along w other inputs to generate a target sequence - means the model is optimized for generating outputs

each can me used independently depending on the task

Attention Layers
- this layer tells the model to pay specific attention to certain words in the sentence it was passes when dealing with the representation of each word
- word itself has no meaning, but meaning affected by context

Architecture: skeleton of the model
Checkoints: weights loaded in
Model: both

## Encoder Models

use only the encoder of a transformer model
at each stage, attention layers can access all the words in the initial sentence
bi-directional attention -> auto-encoding models

pretraining revolves around corrupting a sentence and reconstructing it

best suited for tesks requiring an understanding of the full sentence - like classification or name entity recognition 

ex. BERT, DistilBERT, ELECTRA


## Decoder Models

use only the decoder of a TM
at each stage, attention layers can only access the words positioned *before* it in the sentence -> auto-regressive models

pretraining revolves around predicting the enxt word in the sentence

best suited for text generation

ex. GPT, GPT-2, CTRL

## Sequence to Sequence models Sequence to Sequence Models

Encoder-Decoder Models ^

use both parts of TM architecture

at each stage, attention layers of the encoder can access all words in the initial sentence

pretraining can be done sing the objectives of encoder or decoder models, usually more complez

best suited for tasks like generating new sentences depending on input, like summarization, translation, or generative question answering

ex. BART, Marian, T5

## Bias and Limitations

Limitations
- to anable pretraining on large amts of data, researchers will scrape as much content as possible, taking in whatever is available on the internet
- need to keep in mind the original model could have issues
- fine tuning wont make intrinsic bias disappear


## Summary

`pipelin()` function

Encoder: Sentence classification, named entity recognition, extractive question answering

Decoder: Text generation

Encoder-decoder: Summarization, translation, generative question answering

# Using Transformers 

## Intro

## Behind the pipeline

## Models

## Tokenizers

## Handling Multiple Sequences

## Putting it all together

## Basic Usage Completed
