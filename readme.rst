✨NeuralCoref: Coreference Resolution in spaCy with Neural Networks.
*********************************************************************

NeuralCoref is a pipeline extension for spaCy 2.0 that annotates and resolves coreference clusters using a neural network. NeuralCoref is production-ready, integrated in spaCy's NLP pipeline and easily extensible to new training datasets.

For a brief introduction to coreference resolution and NeuralCoref, please refer to our `blog post <https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30>`_.
NeuralCoref is written in Python/Cython and comes with pre-trained statistical models for English. It can be trained in other languages. NeuralCoref is accompanied by a visualization client `NeuralCoref-Viz <https://github.com/huggingface/neuralcoref-viz>`_, a web interface  powered by a REST server that can be `tried online <https://huggingface.co/coref/>`_. NeuralCoref is released under the MIT license.


✨ Version 3.1 out now! 100x faster and tightly integrated in spaCy pipeline.

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

Install NeuralCoref as a self-contained spaCy model
---------------------------------------------------

This is the easiest way to install NeuralCoref if you don't need to train the model on a new language or dataset or don't want to setup spaCy yourself.

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

To install a model with the latest version of spaCy, copy the **MODEL_URL** of the model you are interested in from the above table and type:

.. code:: bash

    pip install MODEL_URL --upgrade --force-reinstall --no-cache-dir

When using pip it is generally recommended to install packages in a virtual
environment to avoid modifying system state:

.. code:: bash

    venv .env
    source .env/bin/activate
    pip install MODEL_URL

Install NeuralCoref as a spaCy pipeline component
-------------------------------------------------

This is the best way to add NeuralCoref as a `pipeline component <https://spacy.io/usage/processing-pipelines>`_ to a spaCy model you already have.

First install NeuralCoref with the latest version of spaCy:
.. code:: bash

    pip install neuralcoref --upgrade --force-reinstall --no-cache-dir

Then, download NeuralCoref's model weights `here <https://github.com/huggingface/neuralcoref-models/releases/download/bare_weights-3.0.0/neuralcoref.tar.gz>`_ and extract the weights somewhere on your disk:

..code:: bash

    tar -xvzf ./neuralcoref.tar.gz

Install NeuralCoref from source
-------------------------------
Clone the repo and install using pip.

.. code:: bash

	git clone https://github.com/huggingface/neuralcoref.git
	cd neuralcoref
	pip install -e .


Usage
===============================
Loading NeuralCoref as a spaCy model
------------------------------------
NeuralCoref can be used as a spaCy model if you downloaded the full model during the installation (followed the above section "Install NeuralCoref as a self-contained spaCy model").

To load NeuralCoref, simply load the model you dowloaded above using ``spacy.load()`` with the model's name (e.g. `en_coref_md`) and process your text `as usual with spaCy <https://spacy.io/usage>`_.

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

Loading NeuralCoref as a spaCy pipeline component
-------------------------------------------------
NeuralCoref can be added as a pipeline extension to a spaCy model if you installed NeuralCoref and dowloaded its weights during the installation (followed the above section "Install NeuralCoref as a spaCy pipeline component").

To load NeuralCoref, simply load the model you dowloaded above using ``spacy.load()`` with the model's name (e.g. `en_coref_md`) and process your text `as usual with spaCy <https://spacy.io/usage>`_.

.. code:: python

    import spacy
    from neuralcoref import NeuralCoref

    nlp = spacy.load('en')
    nc = NeuralCoref(nlp.vocab).from_disk('path/to/neuralcoref/weights')
    nlp.add_pipe(nc)

    doc = nlp(u'My sister has a dog. She loves him.')

    doc._.has_coref
    doc._.coref_clusters

When you load NeuralCoref as a spaCy component, you can change its parameters in ``NeuralCoref.cfg``.

Here is the full list of the parameters:

=================== =========================== ====================================================
**Parameter**       **Type**                    **Description**
``greedyness``      float                       A number between 0 and 1 determining how greedy the model is about making coreference decisions (more greedy means more coreference links). The default value is 0.5.
``max_dist``        int                         How many mentions back to look when considering possible antecedents of the current mention. Decreasing the value will cause the system to run faster but less accurately. The default value is 50.
``max_dist_match``  int                         The system will consider linking the current mention to a preceding one further than `max_dist` away if they share a noun or proper noun. In this case, it looks `max_dist_match` away instead. The default value is 500.
``blacklist``       boolean                     Should the system resolve coreferences for pronouns in the following list: `["i", "me", "my", "you", "your"]`. The default value is True (coreference resolved).
``store_scores``    boolean                     Should the system store the scores for the coreferences in annotations. The default value is True.
``conv_dict``       dict(str, list(str))        Whether the token is inside at least one corefering mention
``h1``              int                         Size of the first hidden layer of the neural net. You should only change this if you retrain the model. The default value is 1000.
``h2``              int                         Size of the second hidden layer of the neural net. You should only change this if you retrain the model. The default value is 500.
``h3``              int                         Size of the third hidden layer of the neural net. You should only change this if you retrain the model. The default value is 500.
=================== =========================== ====================================================

Here is an example on how to change a parameter.

.. code:: python

    import spacy
    from neuralcoref import NeuralCoref

    nlp = spacy.load('en')
    nc = NeuralCoref(nlp.vocab).from_disk('path/to/neuralcoref/weights')

    nc.cfg['greedyness'] = 0.75

    nlp.add_pipe(nc)

Saving a custom spaCy model with NeuralCoref as a reusable package
------------------------------------------------------------------

Let's say you have added NeuralCoref to a spaCy model and maybe changed some parameters like greedyness. You can now save the full spaCy model so you only have to load the model later to get back the full spaCy pipeline.

The process is similar to the one `described on spaCy website <https://spacy.io/usage/training#saving-loading>`_ and comprises two steps:

1. save the model to the disk,
2. build a package for the model that you can then load with spaCy.

Here is an example on how to save the model to the disk:

.. code:: python

    # Add NeuralCoref to a spaCy model pipeline and do some parameters tweaking
    import spacy
    from neuralcoref import NeuralCoref
    nlp = spacy.load('en')
    nc = NeuralCoref(nlp.vocab).from_disk('path/to/neuralcoref/weights')
    nc.cfg['greedyness'] = 0.75
    nlp.add_pipe(nc)

    # Now let's save our model with NeuralCoref in the pipeline.
    nlp.to_disk('/path/to/save/my/model')

To build a package for the model you need to use the `package` CLI interface of NeuralCoref.

Using NeuralCoref once loaded
-----------------------------
NeuralCoref will resolve the coreferences and annotate them as `extension attributes <https://spacy.io/usage/processing-pipelines#custom-components-extensions>`_ in the spaCy ``Doc``,  ``Span`` and ``Token`` objects under the `._.` dictionary.

Here are a few examples on how you can navigate the coreference cluster chains and display clusters and mentions before we list all the extensions added by NeuralCoref to a spaCy document.

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

**Important**: NeuralCoref mentions are spaCy `Span objects <https://spacy.io/api/span>`_ which means you can access all the usual `Span attributes <https://spacy.io/api/span#attributes>`_ like ``span.start`` (index of the first token of the span in the document), ``span.end`` (index of the first token after the span in the document), etc...

Ex: ``doc._.coref_clusters[1].mentions[-1].start`` will give you the index of the first token of the last mention of the second coreference cluster in the document.

Here is the full list of the Doc, Span and Token Extension Attributes added by NeuralCoref:

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

Using NeuralCoref as a server
=============================

A simple example of server script for integrating NeuralCoref in a REST API is provided as an example in `examples/server.py <examples/server.py>`_.

There are many other ways you can manage and deploy NeuralCoref. Some examples can be found in `spaCy Universe <https://spacy.io/universe/>`_.

Re-train the model / Extend to another language
===============================================

If you want to retrain the model or train it on another language, see our detailed `training instructions <./neuralcoref/train/training.md>`_ as well as our `detailed blog post <https://medium.com/huggingface/how-to-train-a-neural-coreference-model-neuralcoref-2-7bb30c1abdfe>`_

The training code will soon benefit from the same Cython refactoring than the inference code.
