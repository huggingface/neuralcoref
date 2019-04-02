# coding: utf8
"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

1. Change the version in __init__.py and setup.py.

2. Commit these changes with the message: "Release: VERSION"

3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level allennlp directory.
   (this will build a wheel for the python version you use to build it - make sure you use python 3.x).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions of allennlp.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi neuralcoref

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

"""
from __future__ import unicode_literals, absolute_import

import os
import shutil
import tarfile
import tempfile
import logging

from .neuralcoref import NeuralCoref
from .file_utils import NEURALCOREF_MODEL_URL, NEURALCOREF_MODEL_PATH, NEURALCOREF_CACHE, cached_path

__all__ = ['NeuralCoref', 'add_to_pipe']
__version__ = "4.0.0"

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
