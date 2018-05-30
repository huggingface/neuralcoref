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
doc = nlp(text)
coref(doc)

(single_model, pairs_model), cfg = coref.Model()
sl0 = single_model._layers[0]


model_path = os.getcwd() + '/neuralcoref/weights/'
s_weights, s_biases, p_weights, p_biases = [], [], [], []
for file in sorted(os.listdir(model_path)):
    if not file.endswith('.npy'):
        continue
    w = numpy.load(os.path.join(model_path, file)).astype(dtype='float32')
    if file.startswith("single_mention_weights"):
        s_weights.append(w)
    if file.startswith("single_mention_bias"):
        s_biases.append(w)
    if file.startswith("pair_mentions_weights"):
        p_weights.append(w)
    if file.startswith("pair_mentions_bias"):
        p_biases.append(w)

for i in range(len(s_weights)):
    w = s_weights[i]
    b = s_biases[i][:, 0]
    print("layer", i, w.shape, b.shape)
    print(single_model._layers[i].W.shape, single_model._layers[i].b.shape)
    single_model._layers[i].W = w
    single_model._layers[i].b = b
    print(numpy.max(single_model._layers[i].W - w))
    print(numpy.max(single_model._layers[i].b - b))

for i in range(len(p_weights)):
    w = p_weights[i]
    b = p_biases[i][:, 0]
    print("layer", i, w.shape, b.shape)
    print(pairs_model._layers[i].W.shape, pairs_model._layers[i].b.shape)
    pairs_model._layers[i].W = w
    pairs_model._layers[i].b = b
    print(numpy.max(pairs_model._layers[i].W - w))
    print(numpy.max(pairs_model._layers[i].b - b))

serializers = {
    'single_model': lambda p: p.open('wb').write(
        single_model.to_bytes()),
    'pairs_model': lambda p: p.open('wb').write(
        pairs_model.to_bytes()),
}
exclude = []
util.to_disk(model_path, serializers, exclude)

def load_embeddings_from_file(name, store):
    keys = []
    mat = numpy.load(name+"_embeddings.npy").astype(dtype='float32')
    with io.open(name+"_vocabulary.txt", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            keys.append(store.add(line.strip()))
    return keys, mat

keys, mat = load_embeddings_from_file(model_path + 'static_word', nlp.vocab.strings)
old_static = Vectors(shape=mat.shape, data=mat, keys=keys, name='coref_static')
keys, mat = load_embeddings_from_file(model_path + 'tuned_word', nlp.vocab.strings)
old_tuned = Vectors(shape=mat.shape, data=mat, keys=keys, name='coref_tuned')
