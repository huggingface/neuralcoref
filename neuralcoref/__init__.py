# coding: utf8
from __future__ import unicode_literals

import shutil
import tarfile
import tempfile
import logging

from .neuralcoref import NeuralCoref
from .file_utils import NEURALCOREF_CACHE, cached_path

__all__ = ['NeuralCoref']

logger = logging.getLogger(__name__)

MODEL_URL = "https://s3.amazonaws.com/models.huggingface.co/neuralcoref/neuralcoref_model.tar.gz"
LOCAL_PATH = os.path.join(str(NEURALCOREF_CACHE), "/neuralcoref/")

try:
    local_model = cached_path(LOCAL_PATH)
except:
    os.makedirs(LOCAL_PATH)
    downloaded_model = cached_path(MODEL_URL)

    logger.info("extracting archive file {} to dir {}".format(downloaded_model, LOCAL_PATH))
    with tarfile.open(downloaded_model, 'r:gz') as archive:
        archive.extractall(LOCAL_PATH)
