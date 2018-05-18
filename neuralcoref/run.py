import spacy
from neuralcoref.pipeline_extension import CorefComponent
nlp = spacy.load('en_core_web_sm')
coref = CorefComponent(nlp)
nlp.add_pipe(coref) # add it to the pipeline
doc = nlp(u"Some text about Colombia and the Czech Republic")
