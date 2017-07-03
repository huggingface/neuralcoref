# coding: utf8
"""Coref resolution"""

from __future__ import unicode_literals
from __future__ import print_function

from pprint import pprint

import os

import numpy as np

from neuralcoref.docs import Docs, FeaturesExtractor, MENTION_TYPE, NO_COREF_LIST

#######################
##### UTILITIES #######

MAX_FOLLOW_UP = 50

#######################
###### CLASSES ########

class Model:
    '''
    Coreference neural model
    '''
    def __init__(self, model_path, debug=False):
        self.antecedent_matrix = np.load(model_path + "antecedent_matrix.npy")
        self.anaphor_matrix = np.load(model_path + "anaphor_matrix.npy")
        self.pair_features_matrix = np.load(model_path + "pair_features_matrix.npy")
        self.pairwise_first_layer_bias = np.load(model_path + "pairwise_first_layer_bias.npy")
        self.anaphoricity_model = []
        self.debug = debug
        weights = []
        biases = []
        for file in sorted(os.listdir(model_path)):
            if file.startswith("anaphoricity_model_weights"):
                weights.append(np.load(os.path.join(model_path, file)))
            if file.startswith("anaphoricity_model_bias"):
                biases.append(np.load(os.path.join(model_path, file)))
        self.anaphoricity_model = list(zip(weights, biases))
        weights = []
        biases = []
        for file in sorted(os.listdir(model_path)):
            if file.startswith("pairwise_model_weights"):
                weights.append(np.load(os.path.join(model_path, file)))
            if file.startswith("pairwise_model_bias"):
                biases.append(np.load(os.path.join(model_path, file)))
        self.pairwise_model = list(zip(weights, biases))

    def _score(self, features, layers):
        if self.debug: print("🚗 input.shape: ", features.shape, "\n",
                             np.transpose(features[0:8]) if features.size > 8 else np.transpose(features))
        for weights, bias in layers:
            if self.debug: print("shapes", features.shape, weights.shape, bias.shape, "\n🏎 weights:",
                                 weights[0, 0:8] if weights.shape[1] > 8 else weights[0, :], "bias:",
                                 np.transpose(bias[0:8]) if bias.size > 8 else np.transpose(bias))
            features = np.matmul(weights, features) + bias
            if self.debug: print(" features: ", features[0:8, 0] if features.shape[0] > 8 else features[:, 0])
            if weights.shape[0] > 1:
                features = np.maximum(features, 0) # ReLU
                if self.debug: print(" features relu: ", features[0:8, 0] if features.shape[0] > 8 else features[:, 0])
        return np.sum(features)

    def get_anaphoricity_score(self, mention_embedding, anaphoricity_features):
        ''' Anaphoricity score for an anaphor '''
        if self.debug: print("anaphoricity_features", anaphoricity_features)
        first_layer_output = np.concatenate([mention_embedding, anaphoricity_features], axis=0)[:, np.newaxis]
        return self._score(first_layer_output, self.anaphoricity_model)

    def get_pairwise_score(self, antecedent, mention, pair_features):
        antecedent_embedding = np.matmul(self.antecedent_matrix, antecedent.embedding)
        anaphor_embedding = np.matmul(self.anaphor_matrix, mention.embedding)
        first_layer_output = antecedent_embedding + anaphor_embedding \
                             + np.matmul(self.pair_features_matrix, pair_features) + self.pairwise_first_layer_bias
        first_layer_output = np.maximum(first_layer_output, 0)[:, np.newaxis] # ReLU
        return self._score(first_layer_output, self.pairwise_model)


class Algorithm:
    ''' Main coreference resolution algorithm '''
    def __init__(self, nlp=None, greedyness=0.5, max_dist=50, max_dist_match=500, conll=False, use_no_coref_list=True, debug=False):
        self.greedyness = greedyness
        self.max_dist = max_dist
        self.max_dist_match = max_dist_match
        self.debug = debug
        
        if nlp is None:
            print("Loading spacy model")
            try:
                spacy.info('en_core_web_sm')
                model = 'en_core_web_sm'
            except IOError:
                print("No spacy 2 model detected, using spacy1 'en' model")
                model = 'en'
            nlp = spacy.load(model)

        model_path = "./spacykit/coreference/weights/conll/" if conll else "./spacykit/coreference/weights/"
        embed_model_path = "./spacykit/coreference/weights/"
        print("loading model from", model_path)
        self.docs = Docs(embed_model_path, conll=conll, nlp=nlp, use_no_coref_list=use_no_coref_list)
        self.feat_extractor = FeaturesExtractor(docs=self.docs, consider_speakers=conll)
        self.coref_model = Model(model_path)

        self.clusters = {}
        self.mention_to_cluster = []
        self.mentions_single_scores = {}
        self.mentions_pairs_scores = {}

    def _prepare_clusters(self):
        # One cluster for each mention intially
        self.mention_to_cluster = list(range(len(self.docs.mentions)))
        self.clusters = dict((i, [i]) for i in self.mention_to_cluster)
        self.mentions_single_scores = {}
        self.mentions_pairs_scores = {}
        for mention in self.mention_to_cluster:
            self.mentions_single_scores[mention] = None
            self.mentions_pairs_scores[mention] = {}


    def _merge_coreference_clusters(self, ant_idx, mention_idx):
        if self.mention_to_cluster[ant_idx] == self.mention_to_cluster[mention_idx]:
            return

        remove_id = self.mention_to_cluster[ant_idx]
        keep_id = self.mention_to_cluster[mention_idx]
        for idx in self.clusters[remove_id]:
            self.mention_to_cluster[idx] = keep_id
            self.clusters[keep_id].append(idx)

        del self.clusters[remove_id]
#        print("merging: ", self.clusters)

    def display_clusters(self):
        print(self.clusters)
        for key, mentions in self.clusters.items():
            print("cluster", key, "(", ", ".join(str(self.docs[m]) for m in mentions), ")")

    def _run_coref_on_mentions(self, mentions):
        if self.debug: print("☘️  run_coref_on_mentions", mentions)
        best_ant = {}
        n_ant = 0
        for mention_idx, ant_list in self.docs.get_candidate_pairs(mentions, self.max_dist, self.max_dist_match):
            mention = self.docs[mention_idx]
            if self.debug:
                print("\n🚝 ", mention, "\n features_")
                pprint(mention.features_)
                print("spans_embedding_")
                pprint(mention.spans_embedding_)
                print(mention.spans_embedding.shape)
                print("words_embedding_")
                pprint(mention.words_embedding_)
                for r in range(0, mention.embedding.shape[0], 50):
                    print(mention.embedding[r:r+8])

            ana_feats = self.feat_extractor.get_anaphoricity_features(mention)
            anaphoricity_score = self.coref_model.get_anaphoricity_score(mention.embedding, ana_feats)
            self.mentions_single_scores[mention_idx] = anaphoricity_score

            if self.debug: print("anaphoricity_score:", anaphoricity_score)

            best_score = anaphoricity_score - 50 * (self.greedyness - 0.5)
            for ant_idx in ant_list:
                antecedent = self.docs[ant_idx]
                if self.debug: print("🌺", mention, "-", antecedent)
                feats_, pwf = self.feat_extractor.get_pair_features(antecedent, mention)
                if self.debug:
                    print("pair features")
                    pprint(feats_)
                score = self.coref_model.get_pairwise_score(antecedent, mention, pwf)
                self.mentions_pairs_scores[mention_idx][ant_idx] = score

                if self.debug: print("PairwiseScore:", score)

                if score > best_score:
                    best_score = score
                    best_ant[mention_idx] = ant_idx
                    if self.debug: print("  🌈 new best score !")
            if mention_idx in best_ant:
                n_ant += 1
                self._merge_coreference_clusters(best_ant[mention_idx], mention_idx)
        return (n_ant, best_ant)

    def run_coref_on_utterances(self, last_utterances_added=False, follow_chains=True):
        ''' Compute coreference

        Arg:
            last_utterances_added: resolve coreference over the last utterances added to the docs
            follow_chains: follow coreference chains over the previous utterances in the docs
        '''

        self._prepare_clusters()

        mentions = list(self.docs.get_candidate_mentions(last_utterances_added=last_utterances_added))
        n_ant, antecedents = self._run_coref_on_mentions(mentions)
        mentions = antecedents.values()
        if follow_chains and n_ant > 0:
            i = 0
            while i < MAX_FOLLOW_UP:
                i += 1
                n_ant, antecedents = self._run_coref_on_mentions(mentions)
                mentions = antecedents.values()
                if n_ant == 0:
                    break

    def one_shot_coref(self, utterances, utterances_speakers_id=None, context=None,
                       context_speakers_id=None, speakers_names=None):
        self.docs.set_utterances(context, context_speakers_id, speakers_names)
        self.continuous_coref(utterances, utterances_speakers_id, speakers_names)
        return self.get_clusters()

    def continuous_coref(self, utterances, utterances_speakers_id=None, speakers_names=None):
        self.docs.add_utterances(utterances, utterances_speakers_id, speakers_names)
        self.run_coref_on_utterances(last_utterances_added=True, follow_chains=True)
        return self.get_clusters()

    def get_scores(self):
        return {"single_scores": self.mentions_single_scores,
                "pair_scores": self.mentions_pairs_scores}

    def get_clusters(self, remove_singletons=True, use_no_coref_list=True):
        clusters = self.clusters
        remove_id = []
        if use_no_coref_list:
            for key, mentions in clusters.items():
                cleaned_list = []
                for mention_idx in mentions:
                    mention = self.docs.mentions[mention_idx]
                    if mention.lower_ not in NO_COREF_LIST:
                        cleaned_list.append(mention_idx)
                        self.mention_to_cluster[mention_idx] = None
                clusters[key] = cleaned_list
            # Also clean up keys so we can build coref chains in self.get_most_representative
            added = {}
            for key, mentions in clusters.items():
                if self.docs.mentions[key].lower_ in NO_COREF_LIST:
                    remove_id.append(key)
                    self.mention_to_cluster[key] = None
                    if mentions:
                        added[mentions[0]] = mentions
            clusters.update(added)
        if remove_singletons:
            for key, mentions in clusters.items():
                if len(mentions) == 1:
                    remove_id.append(key)
                    self.mention_to_cluster[key] = None
        for rem in remove_id:
            del clusters[rem]

        return clusters

    def get_most_representative(self, last_utterances_added=True, use_no_coref_list=True):
        clusters = self.get_clusters(remove_singletons=True, use_no_coref_list=use_no_coref_list)
        coreferences = {}
        print("clusters", clusters.items())
        for key in self.docs.get_candidate_mentions(last_utterances_added=last_utterances_added):
            if self.mention_to_cluster[key] is None:
                continue
            mentions = clusters.get(self.mention_to_cluster[key], None)
            if mentions is None:
                continue
            representative = self.docs.mentions[key]
            print("representative", representative)
            for mention_idx in mentions[1:]:
                mention = self.docs.mentions[mention_idx]
                print("Against mention", mention)
                if mention.mention_type is not representative.mention_type:
                    if mention.mention_type == MENTION_TYPE["PROPER"] \
                        or (mention.mention_type == MENTION_TYPE["NOMINAL"] and
                                representative.mention_type == MENTION_TYPE["PRONOMINAL"]):
                        print("better !")
                        coreferences[self.docs.mentions[key]] = mention
                        representative = mention

        return coreferences
