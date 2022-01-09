import spacy
nlp = spacy.load('en')

# doc = nlp(u'She likes her classes')

# for token in doc:
#     print(token.text, token.pos_ == 'DET') 

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
doc = nlp(u'Joe is really cool. He received his initial military training during the French and Indian War.')

doc._.has_coref
doc._.coref_clusters
print(doc._.coref_resolved)