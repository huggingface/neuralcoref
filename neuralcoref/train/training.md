# How to train and modify the neural coreference model

Please check our [detailed blog post](https://medium.com/huggingface/how-to-train-a-neural-coreference-model-neuralcoref-2-7bb30c1abdfe) together with these short notes.

## Install

As always, we recommend creating a clean environment (conda or virtual env) to install and train the model.

You will need to install [pyTorch](http://pytorch.org/), the neuralcoref package with the additional training requirements and download a language model for spacy.
Currently this can be done (assuming an English language model) with

```bash
conda install pytorch -c pytorch
pip install -r ./train/training_requirements.txt -e .
python -m spacy download en
```

## Get the data

The following assumes you want to train on English, Arabic or Chinese.
If you want to train on another language, see the section [train on a new language](#train-on-a-new-language) below.

First, download the [OntoNotes 5.0 dataset](https://catalog.ldc.upenn.edu/LDC2013T19) from LDC.

Then, download the [CoNLL-2012 skeleton files](http://conll.cemantix.org/2012/data.html) from the CoNLL 2012 shared task site,
and combine these skeleton files with the OntoNotes files to get the `*._conll` text files which can be used as inputs for the training.

This can be done by executing the script [compile_coref_data.sh](/neuralcoref/train/conll_processing_script/compile_coref_data.sh)
or by following these steps:

- From the [CoNLL 2012 download site](http://conll.cemantix.org/2012/download/), download and extract:
  - http://conll.cemantix.org/2012/download/conll-2012-train.v4.tar.gz
  - http://conll.cemantix.org/2012/download/conll-2012-development.v4.tar.gz
  - http://conll.cemantix.org/2012/download/test/conll-2012-test-key.tar.gz
  - http://conll.cemantix.org/2012/download/test/conll-2012-test-official.v9.tar.gz
  - http://conll.cemantix.org/2012/download/conll-2012-scripts.v3.tar.gz
  - http://conll.cemantix.org/download/reference-coreference-scorers.v8.01.tar.gz
    - Move `reference-coreference-scorers` into the folder `conll-2012/` and rename to `scorer`
- If you are using Python 3.X, you have to edit the `conll-2012/v3/scripts/skeleton2conll.py` file
  - Change `except InvalidSexprException, e:` to `except InvalidSexprException as e`
  - Change all `print` statements to `print()`
- Create the `*._conll` text files by executing
  - `conll-2012/v3/scripts/skeleton2conll.sh -D path_to_ontonotes_folder/data/ conll-2012` (may take a little while)
  - This will create `*.v4_gold_conll` files in each subdirectory of the `conll-2012` `data` folder.
- Assemble the appropriate files into one large file each for training, development and testing
  - `my_lang` can be `english`, `arabic` or `chinese`
  - `cat conll-2012/v4/data/train/data/my_lang/annotations/*/*/*/*.v4_gold_conll >> train.my_lang.v4_gold_conll`
  - `cat conll-2012/v4/data/development/data/my_lang/annotations/*/*/*/*.v4_gold_conll >> dev.my_lang.v4_gold_conll`
  - `cat conll-2012/v4/data/test/data/my_lang/annotations/*/*/*/*.v4_gold_conll >> test.my_lang.v4_gold_conll`

## Prepare the data

Once you have the set of `*.v4_gold_conll` files, move these files into separate (`train`, `test`, `dev`) subdirectories inside a new directory. You can use the already present `data` directory or create another directory anywhere you want. Now, you can prepare the training data by running
[conllparser.py](/neuralcoref/train/conllparser.py) on each split of the data set (`train`, `test`, `dev`) as

```bash
python -m neuralcoref.train.conllparser --path ./$path_to_data_directory/train/
python -m neuralcoref.train.conllparser --path ./$path_to_data_directory/test/
python -m neuralcoref.train.conllparser --path ./$path_to_data_directory/dev/
```

Conllparser will:

- parse the `*._conll` files using spaCy,
- identify predicted mentions,
- compute the mentions features (see our blog post), and
- gather the mention features in a set of numpy arrays to be used as input for the neural net model.

## Train the model

Once the files have been pre-processed
(you should have a set of `*.npy` files in a sub-directory `/numpy` in each of your (`train`|`test`|`dev`) data folder),
you can start the training process using [learn.py](/neuralcoref/train/learn.py), for example as

```bash
python -m neuralcoref.train.learn --train ./data/train/ --eval ./data/dev/
```

There are many parameters and options for the training. You can list them with the usual

```bash
python -m neuralcoref.train.learn --help
```

You can follow the training by running [Tensorboard for pyTorch](https://github.com/lanpa/tensorboard-pytorch)
(it requires a version of Tensorflow, any version will be fine). Run it with `tensorboard --logdir runs`.

## Some details on the training

The model and the training as thoroughfully described in our
[very detailed blog post](https://medium.com/huggingface/how-to-train-a-neural-coreference-model-neuralcoref-2-7bb30c1abdfe).
The training process is similar to the mention-ranking training described in
[Clark and Manning (2016)](http://cs.stanford.edu/people/kevclark/resources/clark-manning-emnlp2016-deep.pdf), namely:

- A first step of training uses a standard cross entropy loss on the mention pair labels,
- A second step of training uses a cross entropy loss on the top pairs only, and
- A third step of training using a slack-rescaled ranking loss.

With the default option, the training will switch from one step to the other as soon as the evaluation stop increasing.

Traing the model with the default hyper-parameters reaches a test loss of about 61.2 which is lower than the mention ranking test loss of 64.7 reported in [Clark and Manning (2016)](http://cs.stanford.edu/people/kevclark/resources/clark-manning-emnlp2016-deep.pdf).

Some possible explanations:

- Our mention extraction function is a simple rule-based function (in [document.py](/document.py)) that was not extensively tuned on the CoNLL dataset and as a result only identifies about 90% of the gold mentions in the CoNLL-2012 dataset (see the evaluation at the start of the training) thereby reducing the maximum possible score. Manually tuning a mention identification module can be a lengthy process that basically involves designing a lot of heuristics to prune spurious mentions while keeping a high recall (see for example the [rule-based mention extraction used in CoreNLP](http://www.aclweb.org/anthology/D10-1048)). An alternative is to train an end-to-end identification module as used in the AllenAI coreference module but this is a lot more complex (you have to learn a pruning function) and the focus of the neuralcoref project is to have a coreference module with a good trade-off between accuracy and simplicity/speed.
- The hyper-parameters and the optimization procedure has not been fully tuned and it is likely possible to find better hyper-parameters and smarter ways to optimize. One possibility is to adjust the balance between the gradients backpropagated in the single-mention and the mentions-pair feedforward networks (see our [blog post](https://medium.com/huggingface/how-to-train-a-neural-coreference-model-neuralcoref-2-7bb30c1abdfe) for more details on the model architecture). Here again, we aimed for a balance between the accuracy and the training speed. As a result, the model trains in about 18h versus about a week for the original model of [Clark and Manning (2016)](http://cs.stanford.edu/people/kevclark/resources/clark-manning-emnlp2016-deep.pdf) and 2 days for the current state-of-the-art model of AllenAI.
- Again for the sake of high throughput, the parse tree output by the [standard English model](https://spacy.io/models/en#en_core_web_sm) of spaCy 2 (that we used for these tests) are slightly less accurate than the carefully tuned CoreNLP parse trees (but they are way faster to compute!) and will lead to a slightly higher percentage of wrong parsing annotations.
- Eventually, it may also be interesting to use newer word-vectors like [ELMo](https://arxiv.org/abs/1802.05365) as they were shown to be able to increase the state-or-the-art coreference model F1 test measure by more than 3 percents.

## Train on a new language

Training on a new language is now possible. However, do not expect it to be a plug-in operation as it involves finding a good annotated dataset and adapting the file-loading and mention-extraction functions to your file format and your language syntax (parse tree).

To boot-strap your work, I detail here the general step you should follow:

- Find a corpus with coreference annotations (as always, the bigger, the better).
- Check that spaCy [supports your language](https://spacy.io/models/) (i.e. is able to parse it). If not, you will have to find another parser that is able to parse your language and integrate it with the project (might involve quite large modifications to neuralcoref depending on the parser).
- Find a set of pre-trained word vectors in your language (gloVe or others).
- If your dataset does not follow the tabular `*_conll` file format (see [details on the CoNLL file format](http://conll.cemantix.org/2012/data.html) on the CoNLL website), you will have to tweak the `load_file` function in [conllparser.py](/conllparser.py) to adapt it to your file format.
- Adapt the mention extraction function to your language parse trees (`extract_mentions_spans` in [document.py](/document.py)) to reach an acceptable identification of mentions (the function should output the list of all possible mention in a document: pronouns, nouns, noun phrases and all the nested possible combinations).
- Re-train the model and tune the hyper-parameters.
