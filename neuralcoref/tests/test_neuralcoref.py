import spacy
from ..__init__ import add_to_pipe


def test_add_pipe():
    nlp = spacy.lang.en.English()
    add_to_pipe(nlp)
    assert "neuralcoref" in nlp.pipe_names
