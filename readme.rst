NeuralCoref: State-of-the-art Coreference Resolution using Neural Networks and spaCy.
******************************************************************************************

NeuralCoref is a pipeline extension for spaCy 2.0 to annotate and resolve coreference clusters in text data. NeuralCoref is based on a neural network model with the aim of making state-of-the-art coreference resolution production-ready, naturally integrated in a Python NLP pipeline and easily extensible to new training datasets. For a brief introduction to coreference resolution and NeuralCoref, please refere to our [blog post](https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30).
NeuralCoref is written in Python/Cython and comes with pre-trained statistical models for English. It can be extended to other languages. NeuralCoref is accompanied by a visualization client [NeuralCoref-Viz](https://github.com/huggingface/neuralcoref-viz), a web interface  powered by a REST server that can be [tried online](https://huggingface.co/coref/). NeuralCoref is commercial open-source software released under the MIT license.


✨ Version 3.0 out now! This new release is 100x faster and comes as pre-trained spaCy language models. Check it out here.

.. image:: https://img.shields.io/github/release/huggingface/neuralcoref.svg?style=flat-square
    :target: https://github.com/huggingface/neuralcoref/releases
    :alt: Current Release Version
.. image:: https://img.shields.io/badge/made%20with%20❤%20and-spaCy-09a3d5.svg
    :target: https://spacy.io
    :alt: spaCy

.. image:: https://huggingface.co/coref/assets/thumbnail-large.png
    :target: https://huggingface.co/coref/
    :alt: NeuralCoref online Demo


Install NeuralCoref as a pre-trained model
==========================================

==================== ===
**Operating system** macOS / OS X, Linux, Windows (Cygwin, MinGW, Visual Studio)
**Python version**   CPython 2.7, 3.4+. Only 64 bit.
==================== ===

The easiest way to install NeuralCoref if you don't want to train a model is as a spaCy model.

NeuralCoref is currently available in English in three versions that mirror `spaCy english models <https://spacy.io/models/en>`_. The larger the model, the higher the accuracy.

To install a model, copy the `MODEL_URL` of the release you are interested in from the following table

============== =================== ======== ====================================================
**Model Name** **MODEL_URL**       **Size** **Description**
`en_coref_sm`  `en_coref_sm_link`_ `35 Mo`  `A small English model based on spaCy en_core_web_sm`
`en_coref_md`  `en_coref_md_link`_ `180 Mo` `A medium English model based on spaCy en_core_web_md`
`en_coref_lg`  `en_coref_lg_link`_ `900 Mo` `A large English model based on spaCy en_core_web_lg`
============== =================== ======== ====================================================

.. _en_coref_sm_link: https://spacy.io/models
.. _en_coref_md_link: https://spacy.io/models
.. _en_coref_lg_link: https://spacy.io/models


You can then install the model as follows.

.. code:: bash

    pip install MODEL_URL

When using pip it is generally recommended to install packages in a virtual
environment to avoid modifying system state:

.. code:: bash

    venv .env
    source .env/bin/activate
    pip install MODEL_URL


Install NeuralCoref from source
===============================
Clone the repo and install using pip (the trained model weights are too large for PyPI)

.. code:: bash

	git clone https://github.com/huggingface/neuralcoref.git
	cd neuralcoref
	pip install -e .


The install script will install `spacy` and `falcon` (only used by the server).

You will also need a Language model for spaCy if you don't already have one. Please refer to [spaCy's models webpage](https://spacy.io/models/) to download and install a model.

The mention extraction module is strongly influenced by the quality of the parsing so we recommend selecting a model with a higher accuray than usual.

Usage
===============================
Loading NeuralCoref
-------------------
NeuralCoref is now integrated as a spaCy Pipeline Extension in the provided models.

To load NeuralCoref, simply load the model you dowloaded above using ``spacy.load()`` with the model's shortcut link and process your text. 

NeuralCoref will resolve the coreferences and add several `extension attributes <https://spacy.io/usage/processing-pipelines#custom-components-extensions>`_ to the spaCy ``Doc`` and ``Span`` objects like the usual spaCy annotations.

Here is a simple example before we list in greater detail NeuralCoref annotations.

.. code:: python

    import spacy
    nlp = spacy.load('en_coref_sm')
    doc = nlp(u'My sister has a dog. She loves him.')

    doc._.has_coref
    doc._.coref_clusters

You can also ``import`` NeuralCoref model directly and then call its ``load()`` method:

.. code:: python

    import spacy
    import en_coref_sm

    nlp = en_coref_sm.load()
    doc = nlp(u'My sister has a dog. She loves him.')

    doc._.has_coref
    doc._.coref_clusters

Attributes
----------
============================= ======================== ====================================================
**Attribute**                 **Type**                 **Description**
``doc._.has_coref``           boolean                  Has any coreference has been resolved in the Doc
``doc._.coref_mentions``      list of ``Span``         List all the mentions that have at least one corefering mention in the doc
``doc._.coref_clusters``      list of list of ``Span`` List clusters of corefering mentions in the doc. Each cluster is a list of mentions refering to the same entity.
``doc._.coref_main_mentions`` list of ``Span``         List for each cluster the mention that can be considered the `main` mention for this cluster.
``doc._.coref_resolved``      unicode                  Unicode representation of the doc where each corefering mention is replaced by the main mention in the associated cluster.
``span._.is_coref``           boolean                  Whether the span has at least one corefering mention
``span._.coref_cluster``      list of ``Span``         List the mentions that corefer with the span
``span._.coref_main_mention`` ``Span``                 Mention that can be considered the `main` mention for the cluster associated to the span.
============================= ======================== ====================================================

These attributes can thus be chained together to navigate in the coreference clusters.

Here is an example:

.. code:: python

    import spacy
    nlp = spacy.load('en_coref_sm')
    doc = nlp(u'My sister has a dog. She loves him.')

    doc._.coref_mentions[1]
    doc._.coref_mentions[1]._.is_coref
    doc._.coref_mentions[1]._.coref_main_mention
    doc._.coref_mentions[1]._.coref_main_mention._.coref_cluster

Using NeuralCoref as a server
-----------------------------

A simple example of server script for integrating NeuralCoref in a REST API is provided as an example in `examples/server.py <examples/server.py>`_.

There are many other ways you can manage and deploy NeuralCoref. Some examples can be found in `spaCy Universe <https://spacy.io/universe/>`_.

Re-train the model / Extend to another language
===============================================
If you want to retrain the model or train it on another language, see our detailed `training instructions <training.md>`_ as well as our `detailed blog post <https://medium.com/huggingface/how-to-train-a-neural-coreference-model-neuralcoref-2-7bb30c1abdfe>`_

