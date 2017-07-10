# Neural coref

State-of-the-art coreference resolution library using neural nets and spaCy. [Try it online !](https://huggingface.co/coref/)
![Neuralcoref demo](https://huggingface.co/coref/assets/thumbnail-large.png)

This coreference resolution module is based on the super fast [spaCy](https://spacy.io/) parser and uses the scoring neural model described in [Deep Reinforcement Learning for Mention-Ranking Coreference Models](http://cs.stanford.edu/people/kevclark/resources/clark-manning-emnlp2016-deep.pdf) by Kevin Clark and Christopher D. Manning, EMNLP 2016.

Be sure to check out [our medium post](https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30) where we talk more about neuralcoref and coreference resolution.

## Installation
Clone the repo (the trained model weights are too large for PyPI)
````
cd neuralcoref
pip install -r requirements.txt
````


You will also need an english model for spaCy if you don't already have spaCy installed.
````
python -m spacy download 'en'
````

The mention extraction module strongly depends on the quality of the parsing so we recommand selecting a model with a higher accuray than usual.
Since the coreference algorithm don't make use of spaCy's word vectors, a medium size spaCy model like [spacy's 'en_depent_web_md' model](https://github.com/explosion/spacy-models/releases/en_depent_web_md-1.2.1) is a good balance between memory footprint and parsing accuracy.

To download and install it use
````
python -m spacy download 'en_depent_web_md'
python -m spacy link en_depent_web_md en
````

If you are an early user of [spacy 2 alpha](https://github.com/explosion/spaCy/releases/tag/v2.0.0-alpha), you can use `neuralcoref` with spacy 2 without any specific modification.

## Usage
### As a standalone server
`python -m neuralcoref.server` starts a wsgiref simple server.

You can retreive coreferences on a text or dialogue utterances by sending a GET request on port 8000.

Example
`curl -G "http://localhost:8000/" --data-urlencode "text=My sister has a dog. She loves him so much"`
### As a library
Please refer to the the source for details on the various functions and arguments.

The main class you can import is `neuralcoref.Coref`.
It offers two main functions for resolving coreference:
- `one_shot_coref()` for a single coreference resolution operation.
- `continuous_coref()` for continous coreference resolution (during an on-going dialogue for instance).

The source code contains details of the various arguments you can use. The code of `server.py` also provides a simple example.

Example:
````
from neuralcoref import Coref
coref = Coref()
clusters = coref.one_shot_coref(u"My sister has a dog and she loves him.")
print(clusters)
````
