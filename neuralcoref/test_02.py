import spacy, os, numpy, io
from spacy import util
from spacy.vectors import Vectors
from spacy.strings import StringStore
from neuralcoref.neuralcoref import NeuralCoref
nlp = spacy.load('en_core_web_lg')
coref = NeuralCoref(nlp.vocab)
disk_model_path = os.getcwd() + '/en_coref_lg/neuralcoref'
coref.from_disk(disk_model_path)
text = u"I know that Barbara and Sandy are here. I see Barbara watching TV. I hear Sandy breathing."
print("=!=" + text)
doc3 = nlp(text)
coref(doc3)
