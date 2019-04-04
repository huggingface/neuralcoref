# coding: utf8
from __future__ import unicode_literals, absolute_import

import os
import shutil
import tarfile
import tempfile
import logging

from .neuralcoref import NeuralCoref
from .file_utils import NEURALCOREF_MODEL_URL, NEURALCOREF_MODEL_PATH, NEURALCOREF_CACHE, cached_path

__all__ = ['NeuralCoref', 'add_to_pipe']
__version__ = "3.9.0"

logger = logging.getLogger(__name__)

if os.path.exists(NEURALCOREF_MODEL_PATH) and os.path.exists(os.path.join(NEURALCOREF_MODEL_PATH, "cfg")):
    logger.info("Loading model from {}".format(NEURALCOREF_MODEL_PATH))
    local_model = cached_path(NEURALCOREF_MODEL_PATH)
else:
    if not os.path.exists(NEURALCOREF_MODEL_PATH):
        os.makedirs(NEURALCOREF_MODEL_PATH)
    logger.info("Getting model from {} or cache".format(NEURALCOREF_MODEL_URL))
    downloaded_model = cached_path(NEURALCOREF_MODEL_URL)

    logger.info("extracting archive file {} to dir {}".format(downloaded_model, NEURALCOREF_MODEL_PATH))
    with tarfile.open(downloaded_model, 'r:gz') as archive:
        archive.extractall(NEURALCOREF_CACHE)

def add_to_pipe(nlp):
    coref = NeuralCoref(nlp.vocab)
    nlp.add_pipe(coref, name='neuralcoref')
    return nlp
