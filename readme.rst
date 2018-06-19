✨NeuralCoref: Coreference Resolution in spaCy with Neural Networks.
*******************************************************************

NeuralCoref is a pipeline extension for spaCy 2.0 that annotates and resolves coreference clusters using a neural network. NeuralCoref is production-ready, integrated in spaCy's NLP pipeline and easily extensible to new training datasets.

For a brief introduction to coreference resolution and NeuralCoref, please refer to our `blog post <https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30>`_.
NeuralCoref is written in Python/Cython and comes with pre-trained statistical models for English. It can be trained in other languages. NeuralCoref is accompanied by a visualization client `NeuralCoref-Viz <https://github.com/huggingface/neuralcoref-viz>`_, a web interface  powered by a REST server that can be `tried online <https://huggingface.co/coref/>`_. NeuralCoref is released under the MIT license.


✨ Version 3.0 out now! 100x faster and tightly integrated in spaCy pipeline.

.. image:: https://img.shields.io/github/release/huggingface/neuralcoref.svg?style=flat-square
    :target: https://github.com/huggingface/neuralcoref/releases
    :alt: Current Release Version
.. image:: https://img.shields.io/badge/made%20with%20❤%20and-spaCy-09a3d5.svg
    :target: https://spacy.io
    :alt: spaCy
.. image:: https://travis-ci.org/huggingface/neuralcoref.svg?branch=master
    :target: https://travis-ci.org/huggingface/neuralcoref
    :alt: Travis-CI

.. image:: https://huggingface.co/coref/assets/thumbnail-large.png
    :target: https://huggingface.co/coref/
    :alt: NeuralCoref online Demo


Install NeuralCoref
===================

As a pre-trained spaCy model
----------------------------

This is the easiest way to install NeuralCoref if you don't need to train the model on a new language or dataset.

==================== ===
**Operating system** macOS / OS X, Linux, Windows (Cygwin, MinGW, Visual Studio)
**Python version**   CPython 2.7, 3.4+. Only 64 bit.
==================== ===

NeuralCoref is currently available in English with three models of increasing accuracy that mirror `spaCy english models <https://spacy.io/models/en>`_. The larger the model, the higher the accuracy:

================== =================== =============== ====================================================
**Model Name**     **MODEL_URL**       **Size**        **Description**
en_coref_sm        `en_coref_sm`_      78 Mo           A *small* English model based on spaCy `en_core_web_sm-2.0.0`_
en_coref_md        `en_coref_md`_      161 Mo          [Recommended] A *medium* English model based on spaCy `en_core_web_md-2.0.0`_ 
en_coref_lg        `en_coref_lg`_      893 Mo          A *large* English model based on spaCy `en_core_web_lg-2.0.0`_
================== =================== =============== ====================================================

.. _en_core_web_sm-2.0.0: https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-2.0.0
.. _en_core_web_md-2.0.0: https://github.com/explosion/spacy-models/releases/tag/en_core_web_md-2.0.0
.. _en_core_web_lg-2.0.0: https://github.com/explosion/spacy-models/releases/tag/en_core_web_lg-2.0.0

.. _en_coref_sm: https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_sm-3.0.0/en_coref_sm-3.0.0.tar.gz
.. _en_coref_md: https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz
.. _en_coref_lg: https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_lg-3.0.0/en_coref_lg-3.0.0.tar.gz

To install a model, copy the **MODEL_URL** of the model you are interested in from the above table and type:

.. code:: bash

    pip install MODEL_URL

When using pip it is generally recommended to install packages in a virtual
environment to avoid modifying system state:

.. code:: bash

    venv .env
    source .env/bin/activate
    pip install MODEL_URL


Install NeuralCoref from source
-------------------------------
Clone the repo and install using pip.

.. code:: bash

	git clone https://github.com/huggingface/neuralcoref.git
	cd neuralcoref
	pip install -e .


Usage
===============================
Loading NeuralCoref
-------------------
NeuralCoref is integrated as a spaCy Pipeline Extension .

To load NeuralCoref, simply load the model you dowloaded above using ``spacy.load()`` with the model's name (e.g. `en_coref_md`) and process your text `as usual with spaCy <https://spacy.io/usage>`_.

NeuralCoref will resolve the coreferences and annotate them as `extension attributes <https://spacy.io/usage/processing-pipelines#custom-components-extensions>`_ in the spaCy ``Doc``,  ``Span`` and ``Token`` objects under the `._.` dictionary.

Here is a simple example before we dive in greater details.

.. code:: python

    import spacy
    nlp = spacy.load('en_coref_md')
    doc = nlp(u'My sister has a dog. She loves him.')

    doc._.has_coref
    doc._.coref_clusters

You can also ``import`` NeuralCoref model directly and call its ``load()`` method:

.. code:: python

    import en_coref_md

    nlp = en_coref_md.load()
    doc = nlp(u'My sister has a dog. She loves him.')

    doc._.has_coref
    doc._.coref_clusters

Doc, Span and Token Extension Attributes
----------------------------------------------
============================= ====================== ====================================================
**Attribute**                 **Type**               **Description**
``doc._.has_coref``           boolean                Has any coreference has been resolved in the Doc
``doc._.coref_clusters``      list of ``Cluster``    All the clusters of corefering mentions in the doc
``doc._.coref_resolved``      unicode                Unicode representation of the doc where each corefering mention is replaced by the main mention in the associated cluster.
``span._.is_coref``           boolean                Whether the span has at least one corefering mention
``span._.coref_cluster``      ``Cluster``            Cluster of mentions that corefer with the span
``token._.in_coref``          boolean                Whether the token is inside at least one corefering mention
``token._.coref_clusters``    list of ``Cluster``    All the clusters of corefering mentions that contains the token
============================= ====================== ====================================================

The Cluster class
-----------------
The Cluster class is a small container for a cluster of mentions.

A ``Cluster`` contains 3 attributes:

==================== ======================== ====================================================
**Attribute**        **Type**                 **Description**
``cluster.i``        int                      Index of the cluster in the Doc
``cluster.main``     ``Span``                 Span of the most representative mention in the cluster
``cluster.mentions`` list of ``Span``         All the mentions in the cluster
==================== ======================== ====================================================

The ``Cluster`` class also implements a few Python class methods to simplify the navigation inside a cluster:

======================== ======================== ====================================================
**Method**               **Output**               **Description**
``Cluster.__getitem__``  return ``Span``          Access a mention in the cluster
``Cluster.__iter__``     yields ``Span``          Iterate over mentions in the cluster
``Cluster.__len__``      return int               Number of mentions in the cluster
======================== ======================== ====================================================

Examples
--------

Here are some example on how you can navigate the coreference cluster chains.

.. code:: python

    import spacy
    nlp = spacy.load('en_coref_sm')
    doc = nlp(u'My sister has a dog. She loves him')

    doc._.coref_clusters
    doc._.coref_clusters[1].mentions
    doc._.coref_clusters[1].mentions[-1]
    doc._.coref_clusters[1].mentions[-1]._.coref_cluster.main

    token = doc[-1]
    token._.in_coref
    token._.coref_clusters

    span = doc[-1:]
    span._.is_coref
    span._.coref_cluster.main
    span._.coref_cluster.main._.coref_cluster

Using NeuralCoref as a server
=============================

A simple example of server script for integrating NeuralCoref in a REST API is provided as an example in `examples/server.py <examples/server.py>`_.

There are many other ways you can manage and deploy NeuralCoref. Some examples can be found in `spaCy Universe <https://spacy.io/universe/>`_.

Re-train the model / Extend to another language
===============================================

If you want to retrain the model or train it on another language, see our detailed `training instructions <./neuralcoref/train/training.md>`_ as well as our `detailed blog post <https://medium.com/huggingface/how-to-train-a-neural-coreference-model-neuralcoref-2-7bb30c1abdfe>`_

The training code will soon benefit from the same Cython refactoring than the inference code.
