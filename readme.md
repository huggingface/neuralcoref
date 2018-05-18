# Neural coref v2.0

State-of-the-art coreference resolution library using neural nets and spaCy. [Try it online !](https://huggingface.co/coref/)
![Neuralcoref demo](https://huggingface.co/coref/assets/thumbnail-large.png)

This coreference resolution module is based on the super fast [spaCy](https://spacy.io/) parser and uses the neural net scoring model described in [Deep Reinforcement Learning for Mention-Ranking Coreference Models](http://cs.stanford.edu/people/kevclark/resources/clark-manning-emnlp2016-deep.pdf) by Kevin Clark and Christopher D. Manning, EMNLP 2016.

With ✨Neuralcoref v2.0, you should now be able to train  the coreference resolution system on your own dataset — e.g., another language than English! — **provided you have an annotated dataset**. Be sure to check [our medium post detailing the release of v2.0 and how to train the model](https://medium.com/huggingface/how-to-train-a-neural-coreference-model-neuralcoref-2-7bb30c1abdfe)  as well as our [first medium post](https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30) in which we talk more about coreference resolution in general.

## Installation
Clone the repo and install using pip (the trained model weights are too large for PyPI)

```
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install .
```

The install script will install `spacy` and `falcon` (only used by the server).

You will also need an English model for spaCy if you don't already have one.
```
python -m spacy download en
```

The mention extraction module is strongly influenced by the quality of the parsing so we recommend selecting a model with a higher accuray than usual.

## Re-train the model / Extend to another language
If you want to retrain the model or train it on another language, see our detailed [training instructions](training.md) as well as our [detailed blog post](https://medium.com/huggingface/how-to-train-a-neural-coreference-model-neuralcoref-2-7bb30c1abdfe)

## Usage
### As a standalone server
`python -m neuralcoref.server` starts a wsgiref simple server.
You can retreive coreferences on a text or dialogue utterances by sending a GET request on port 8000.

*Example of server query*:
`curl -G "http://localhost:8000/" --data-urlencode "text=My sister has a dog. She loves him so much"`

### As a library
The main class to import is `neuralcoref.Coref`.
#### Resolving coreferences:
`neuralcoref.Coref` exposes two main functions for resolving coreference:
- `Coref.one_shot_coref()` for a single coreference resolution. Clear history, load a list of utterances/sentences and an optional context and run the coreference model on them.
        *Arguments*:
    - `utterances` : optional iterator or list of string corresponding to successive utterances (in a dialogue) or sentences. Can be a single string for non-dialogue text.
    - `utterances_speakers_id=None` : optional iterator or list of speaker id for each utterance (in the case of a dialogue).
        - if not provided, assume two speakers speaking alternatively.
        - if utterances and utterances_speaker are not of the same length padded with None
    - `context=None` : optional iterator or list of string corresponding to successive utterances  or sentences sent prior to `utterances`. Coreferences are not computed for the mentions identified in `context`. The mentions in `context` are only used as possible antecedents to mentions in `uterrance`. Reduce the computations when we are only interested in resolving coreference in the last sentences/utterances.
    - `context_speakers_id=None` : optional, same as `utterances_speakers_id` for `context`. 
    - `speakers_names=None` : optional dictionnary of list of acceptable speaker names (strings) for speaker_id in `utterances_speakers_id` and `context_speakers_id`. Help identify the speakers when they are mentioned in the utterances/sentences.

    *Return*: dictionnary of clusters of entities with coreference resolved
- `Coref.continuous_coref()` for continous coreference resolution (during an on-going dialogue for instance). What is the difference between `Coref.one_shot_coref()` and `Coref.continuous_coref()`: `Coref.one_shot_coref()` start from a blank page with a supplied context and supplied last utterances/sentences. `Coref.continuous_coref()` add the supplied utterance/sentence to the memory and compute coreferences on it.
        *Arguments*:
    - `utterances` : optional iterator or list of string corresponding to successive utterances (in a dialogue) or sentences. Can be a single string for non-dialogue text.
    - `utterances_speakers_id=None` : optional iterator or list of speaker id for each utterance (in the case of a dialogue).
        - if not provided, assume two speakers speaking alternatively.
        - if utterances and utterances_speaker are not of the same length padded with None
    - `speakers_names=None` : optional dictionnary of list of acceptable speaker names (strings) for speaker_id in `utterances_speakers_id` and `context_speakers_id`. Help identify the speakers when they are mentioned in the utterances/sentences.

    *Return*: dictionnary of clusters of entities with coreference resolved

#### Retreiving results of the coreference operation:
`neuralcoref.Coref` has several functions for retreiving the results of the coreference resolution:
- `Coref.get_utterances()`: retrieve the list utterances parsed by spaCy (list of spaCy Docs).
    Argument:
    - `last_utterances_added=True`: only send back the last utterances/sentences added.
- `Coref.get_mentions()`: retrieve the list of mentions identified during the coreference resolution.
- `Coref.get_scores()`: retrieve dictionnary of scores for single mentions and pair of mentions.
- `Coref.get_clusters()`: return the clusters computed during the coreference: dictionnary of list of mentions indexes. The mentions indexes are index in the list of mentions obtained by `Coref.get_mentions()`.
    Arguments:
    - `remove_singletons=True`: only send back cluster with more than one mention.
    - `blacklist=True`: don't send back mentions in a list of words for which coreference is not necessary/tricky (currently "I" and "you").
- `Coref.get_most_representative()`: return a dictionnary of coreference with a *representative coreference* for each mention which has an antecedent. A *representative coreference* is typically a proper noun if there is one or a noun chunk if possible.
    Arguments:
    - `last_utterances_added=True`: only send back representative mentions for the coreferences resolved in the last utterances added.
    - `blacklist=True`: don't send back representative mentions for a list of words for which coreference is not necessary/tricky (currently "I" and "you").

The source code contains details of the various arguments you can use.
The code of `server.py` also provides a simple example.

#### Example:
Here is a simple example of use of the coreference resolution system.

````python
from neuralcoref import Coref

coref = Coref()
clusters = coref.one_shot_coref(utterances=u"She loves him.", context=u"My sister has a dog.")
print(clusters)

mentions = coref.get_mentions()
print(mentions)

utterances = coref.get_utterances()
print(utterances)

resolved_utterance_text = coref.get_resolved_utterances()
print(resolved_utterance_text)
````
