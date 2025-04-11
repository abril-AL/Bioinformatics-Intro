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

`pipeline()` function

Encoder: Sentence classification, named entity recognition, extractive question answering

Decoder: Text generation

Encoder-decoder: Summarization, translation, generative question answering

# Using Transformers 

## Intro

Library created to manage large models for easy use. 

## Behind the pipeline

Tokenizer -> Model -> Post Processing

Raw Text -> Input IDs -> Logits -> Predictions

Preprocessing with a tokenizer: splits input into tokens, maps each to an integer, adds additional useful inputs. Once we have tokenizer, we ass it our sentence. 

Tensor: like NumPy arrays. can be scalar, vector, or more dimension. Transformer models *only* accept tensors.

Going through the model: for each model input we get back a high-dimensional vector representing the contextual understanding of that input by the transformer model. (hidden states)

Model Heads: making sense out of numbers
- takes high-dimensional vector and projects into different dimension 
- usually a few linear layers

## Models

'AutoModel' class
- simple wrappers over the wide variety of models available
- can automatically guess the appropriate model architecture for your checkpoint, and then instantiates a model with this architecture.

Creating a Transformer
```python
# init a BERT model - first load a config obj
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig() # attributes for building the model

# Building the model from the config
model = BertModel(config)
```

Different Loading Methods
- initialized with random values
```python
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)
```
- will ouput gibbereish, needs to be trained - needs time and data!
- instead will load a transformer that is already trained:
```python
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")
```

Saving Methods
- use the `save_pretrained()` method
- `model.save_pretrained("directory_on_my_computer")`
- saves files to disk: attributes config, and state dictionary (model weights)

Using a Transformer model for inference
- tokenizer does input to appropriate tensor
- then pass to model - just call it
	- `output = model(model_inputs)`
	- where `model_inputs = torch.tensor(encoded_seq)'

## Tokenizers

Purpose: translate text into data that can be processed by the model

Ex. Word Based, easy, decent results

Ex. Character-based, smaller vocabulary, less meaningful tokens

Subword Tokenization
- frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords.
- provide more semantic meaning

Loading and Saving
- simple to ld/sv tokenizers
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
```

Encoding
- translating text to numbers
- two steps: tokenization then conversion to input IDs
- that way we acn build a tensor out of them and feed into the model

Deconding: go from vocab indices to a string

##  Handling Multiple Sequences

Transformers models expect multiple sentences by default

Batching: act of sending multiple sentences through the model, all at once

Padding the inputs: use to make tensor have the right shape, add padding token

Attention masks
- tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s
- 1 - indicate the corresponding tokens should be attended to
- 0 - indicate the corresponding tokens should not be attended to (ignore)

Longer sequences
- limit to lengths of sequences
- solns: use diff model or truncate sequence

## Putting it all together

how it can handle multiple sequences (padding!), very long sequences (truncation!), and multiple types of tensors with its main API:
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```

## Basic Usage Completed
