# Neural coref

A state-of-the-art coreference resolution system based on neural nets.

This coreference resolution system is based on the super fast (spaCy parser)[https://spacy.io/] and uses the high quality scoring neural network described in [Deep Reinforcement Learning for Mention-Ranking Coreference Models](http://cs.stanford.edu/people/kevclark/resources/clark-manning-emnlp2016-deep.pdf) by Kevin Clark and Christopher D. Manning, EMNLP 2016.

To know more about coreference adn neuralcoref, check out (our medium post)[https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30].

## Installation
`pip install neuralcoref`


You also need to download an english model for Spacy if you don't already have one.
`python -m spacy download 'en'`


The mention extraction algorithm depends quite strongly on the quality of the parsing so we would recommand to use a model with a good accuray.
On the other hand, the coreference algorithm don't use spacy's model word vectors (although it could) so a good balance between parsing accuracy and model size is for example [spacy's 'en_depent_web_md' model](https://github.com/explosion/spacy-models/releases/en_depent_web_md-1.2.1) which has a parsing accuracy of 90.6% on Ontonotes 5.0.

You can get and use it like this
````
python -m spacy download 'en_depent_web_md'
python -m spacy link en_depent_web_md en_default --force
````

## Usage
### Standalone server
`python server.py` starts a wsgiref simple server on port 8000, endpoint `/coref/`.
You can retreive coreferences on a text or dialogue utterances by calling the server.
Example:
`curl http://localhost:8001/coref?text=She%20loves%20him%20so%20much&context=My%20sister%20has%20a%20dog.`
### Library
````
from neuralcoref import Coref
coref = Coref()
# retrieve all the coreference resolved clusters
clusters = coref.one_shot_coref(u"My sister has a dog and she loves him.")
print(clusters)
# Show a dictionnary of resolved coreferences with the most representative mention of each cluster
coreferences = coref.get_most_representative()
pritn(coreferences)
````
