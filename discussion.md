# Discussion

Aakarsh Anand

Friday, 10am-12pm, Pub Aff 2250

## Discussion 1

Office Hours - e6 294

Mon 10am-12pm

### Background Bio Knowledge

DNA -> RNA -> **Proteins**

ways to measure data of this bio process / stages

Omics Tech
- Genomics: study dna, genes, regulatory regions, mutations, variants
- Transcriptomics: study of RNA
- Protomics: study protein system

### Machine Learning Basics

Functions to generalize for unseen data

Unsupervised Learing: Learn without labeled data

Neural Networks
- learns to recognize patters by passing data through layers of interconnected nodes
- more expressive, cost of interpretability 

### Transformer Architecture

Architectire of LLMs composed of encoders and decoder components
- process input simultaneously
- ID relevant relationships b/w words
- capture long-range dependencies b/w inputs effectively 

#### BERT vs GPT models

BERT	
- classification, question answering, 
- only uses encoder (designed to learn embddings )
- bidirectional - model sees all words before and after - training detail 
- typically needs more training data

GPT
- only uses decoder ( for generating )
- text gen, auto-complete
- unidirectional - predicts next word - only sees words before
- more resource intensive

#### Transfer Learning and Fine-Tuning
- Pretraining: first train in large general text, objectiev to predict next word, learn the grammer and general language patterns
- Fine-Tuning: model can be adapted to specific tasks or domains, ex. biomedical QA or legal summarization
- Transfer Learning: leverage knowledge from pretraining to perform well on ne task with limited data (generalization?)
- Reinforcement Learning: further train from human feedback (RLHF)

### Project 0

Hugging Face
- quick NLP with `pipeline()`
- lot of prebuild pipelines in HF built for different tasks

Demo:

https://colab.research.google.com/drive/1p58QxpTV1FY5YQoqlWyMOAtfIEuSPy92?usp=sharing

Model Hub
- contains thousands of pretrained models
- search by task, framework or dataset

#### Behind the pipeline
- Tokenizer -> Model -> Post-Processing
- Raw Text -> Input Ids -> Logits -> Predictions

##### Tokenization
- preprocessing on inputs, diff ways to do it
- token should have meaning so model can learn the meaning associated with tokens
- ex. word tokenization (has itw own problems)
- ex. character tokenization (char itself wont mean alot on its own)
- ex. subword tokenization - extract more info from words, isolation meaning encapsulated in words
- then turned into number for model
- why its useful to use pretrained tokenizer 

`****.from_pretrained("****")` <- using weights from a model already trained

##### The Model

Transformer Network: lean how to create good hidden representation of input - how u use it is determined by the head
- Embeddings: projection on inputs into higher dimentional space
- Layers: 

Hidden States

Head - condense high dim vectors into lower dim nums to be interpreted in the end

##### Put it all together
- choose model and tokenizer
- load weights into both
- pass it into model

can pass in mult sequences at a time

### Why Learn NLP in c121

Parallels b/w language and biology
- tokenize amino acid seq?
- model structure-function relationships with transformers?
- uncover biologically meaningful patters using attention

Understanding general flow -> can switch data around
- learning how to go from text to bio via ML understanding
- how to tweak things for the set up

ESM - Evolutionary Scale Model
- came from Meta research
- predict protein secondary structures
- p1 or p2 (details later)




