import os
import tarfile
import logging

# Filter Cython warnings that would force everybody to re-compile from source (like https://github.com/numpy/numpy/pull/432).
import warnings

warnings.filterwarnings("ignore", message="spacy.strings.StringStore size changed")

from neuralcoref.neuralcoref import NeuralCoref
from neuralcoref.file_utils import (
    NEURALCOREF_MODEL_URL,
    NEURALCOREF_MODEL_PATH,
    NEURALCOREF_CACHE,
    cached_path,
)

__all__ = ["NeuralCoref", "add_to_pipe"]
__version__ = "4.1.0"

logger = logging.getLogger(__name__)

if os.path.exists(NEURALCOREF_MODEL_PATH) and os.path.exists(
    os.path.join(NEURALCOREF_MODEL_PATH, "cfg")
):
    logger.info(f"Loading model from {NEURALCOREF_MODEL_PATH}")
    local_model = cached_path(NEURALCOREF_MODEL_PATH)
else:
    if not os.path.exists(NEURALCOREF_MODEL_PATH):
        os.makedirs(NEURALCOREF_MODEL_PATH, exist_ok=True)
    logger.info(f"Getting model from {NEURALCOREF_MODEL_URL} or cache")
    downloaded_model = cached_path(NEURALCOREF_MODEL_URL)

    logger.info(
        f"extracting archive file {downloaded_model} to dir {NEURALCOREF_MODEL_PATH}"
    )
    with tarfile.open(downloaded_model, "r:gz") as archive:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(archive, NEURALCOREF_CACHE)


def add_to_pipe(nlp, **kwargs):
    coref = NeuralCoref(nlp.vocab, **kwargs)
    nlp.add_pipe(coref, name="neuralcoref")
    return nlp
