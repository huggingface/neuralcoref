# Personally Identifiable Information Processing

This is code for a multi-lingual Named Entity Recognition and PII processor.

Home for code for the PII Hackathon

- The code under the directory ontology will be used for Module 1.
- The code under the directory pii will be used for Module 2.
- The code under the directory masakhane-ner will be used for Module 3.

## Requirements and Installation

- pip install spacy==2.1.8
- git clone  https://github.com/ontocord/pii_processing
- cd pii_processing/
- python setup.py install
- python -m nltk.downloader punkt stopwords  wordnet
- python -m spacy download en_core_web_lg


## PII-Hackathon Reference Impelmentation

This repository includes code for the PII Hackathon for BigScience and AISC. It is based on original code by Ontocord, LLC (https://github.com/ontocord), Hugginface's Nueralcoref (https://github.com/huggingface/neuralcoref) and MasakhaNER https://github.com/masakhane-io/masakhane-ner which is in turn based on https://github.com/huggingface/transformers/.

All code is released under Apache 2.0, except Neuralcoref which is under the MIT License.


