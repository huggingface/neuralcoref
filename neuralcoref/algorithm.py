# coding: utf8
"""Coref resolution"""

from __future__ import unicode_literals
from __future__ import print_function

from pprint import pprint

import os
import spacy
import numpy as np

from data import Data, MENTION_TYPE, NO_COREF_LIST

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

#######################
##### UTILITIES #######

MAX_FOLLOW_UP = 50

#######################
###### CLASSES ########

class Model:
    '''
    Coreference neural model
    '''
    def __init__(self, model_path):
        weights, biases = [], []
        for file in sorted(os.listdir(model_path)):
            if file.startswith("single_mention_weights"):
                weights.append(np.load(os.path.join(model_path, file)))
            if file.startswith("single_mention_bias"):
                biases.append(np.load(os.path.join(model_path, file)))
        self.single_mention_model = list(zip(weights, biases))
        weights, biases = [], []
        for file in sorted(os.listdir(model_path)):
            if file.startswith("pair_mentions_weights"):
                weights.append(np.load(os.path.join(model_path, file)))
            if file.startswith("pair_mentions_bias"):
                biases.append(np.load(os.path.join(model_path, file)))
        self.pair_mentions_model = list(zip(weights, biases))

    def _score(self, features, layers):
        for weights, bias in layers:
            features = np.matmul(weights, features) + bias
            if weights.shape[0] > 1:
                features = np.maximum(features, 0) # ReLU
        return np.sum(features)

    def get_single_mention_score(self, mention_embedding, anaphoricity_features):
        first_layer_input = np.concatenate([mention_embedding,
                                            anaphoricity_features], axis=0)[:, np.newaxis]
        return self._score(first_layer_input, self.single_mention_model)

    def get_pair_mentions_score(self, antecedent, mention, pair_features):
        first_layer_input = np.concatenate([antecedent.embedding,
                                            mention.embedding,
                                            pair_features], axis=0)[:, np.newaxis]
        return self._score(first_layer_input, self.pair_mentions_model)


class Coref:
    '''
    Main coreference resolution algorithm
    '''
    def __init__(self, nlp=None, greedyness=0.5, max_dist=50, max_dist_match=500, conll=None, use_no_coref_list=True, debug=False):
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
        
        model_path = os.path.join(PACKAGE_DIRECTORY, "weights/conll/" if conll is not None else "weights/")
        embed_model_path = os.path.join(PACKAGE_DIRECTORY, "weights/")
        print("loading model from", model_path)
        self.data = Data(nlp, model_path=embed_model_path, conll=conll, use_no_coref_list=use_no_coref_list, consider_speakers=conll)
        self.coref_model = Model(model_path)

        self.clusters = {}
        self.mention_to_cluster = []
        self.mentions_single_scores = {}
        self.mentions_single_features = {}
        self.mentions_pairs_scores = {}
        self.mentions_pairs_features = {}

    ###################################
    #### ENTITY CLUSTERS FUNCTIONS ####
    ###################################

    def _prepare_clusters(self):
        '''
        Clean up and prepare one cluster for each mention
        '''
        self.mention_to_cluster = list(range(len(self.data.mentions)))
        self.clusters = dict((i, [i]) for i in self.mention_to_cluster)
        self.mentions_single_scores = {}
        self.mentions_single_features = {}
        self.mentions_pairs_scores = {}
        self.mentions_pairs_features = {}
        for mention in self.mention_to_cluster:
            self.mentions_single_scores[mention] = None
            self.mentions_single_features[mention] = None
            self.mentions_pairs_scores[mention] = {}
            self.mentions_pairs_features[mention] = {}

    def _merge_coreference_clusters(self, ant_idx, mention_idx):
        '''
        Merge two clusters together
        '''
        if self.mention_to_cluster[ant_idx] == self.mention_to_cluster[mention_idx]:
            return

        remove_id = self.mention_to_cluster[ant_idx]
        keep_id = self.mention_to_cluster[mention_idx]
        for idx in self.clusters[remove_id]:
            self.mention_to_cluster[idx] = keep_id
            self.clusters[keep_id].append(idx)

        del self.clusters[remove_id]

    def display_clusters(self):
        '''
        Print clusters informations
        '''
        print(self.clusters)
        for key, mentions in self.clusters.items():
            print("cluster", key, "(", ", ".join(str(self.data[m]) for m in mentions), ")")

    ###################################
    ####### MAIN COREF FUNCTIONS ######
    ###################################

    def run_coref_on_mentions(self, mentions):
        '''
        Run the coreference model on a mentions iterator or list
        '''
        best_ant = {}
        n_ant = 0
        for mention_idx, ant_list in self.data.get_candidate_pairs(mentions, self.max_dist, self.max_dist_match):
            mention = self.data[mention_idx]
            feats_, ana_feats = self.data.get_anaphoricity_features(mention)
            anaphoricity_score = self.coref_model.get_single_mention_score(mention.embedding, ana_feats)
            self.mentions_single_scores[mention_idx] = anaphoricity_score
            self.mentions_single_features[mention_idx] = {"spansEmbeddings": mention.spans_embeddings_, "wordsEmbeddings": mention.words_embeddings_, "features": feats_}

            best_score = anaphoricity_score - 50 * (self.greedyness - 0.5)
            for ant_idx in ant_list:
                antecedent = self.data[ant_idx]
                feats_, pwf = self.data.get_pair_features(antecedent, mention)
                score = self.coref_model.get_pair_mentions_score(antecedent, mention, pwf)
                self.mentions_pairs_scores[mention_idx][ant_idx] = score
                self.mentions_pairs_features[mention_idx][ant_idx] = {"pairFeatures": feats_, "antecedentSpansEmbeddings": antecedent.spans_embeddings_,
                                                                      "antecedentWordsEmbeddings": antecedent.words_embeddings_,
                                                                      "mentionSpansEmbeddings": mention.spans_embeddings_,
                                                                      "mentionWordsEmbeddings": mention.words_embeddings_ }

                if score > best_score:
                    best_score = score
                    best_ant[mention_idx] = ant_idx
            if mention_idx in best_ant:
                n_ant += 1
                self._merge_coreference_clusters(best_ant[mention_idx], mention_idx)
        return (n_ant, best_ant)

    def run_coref_on_utterances(self, last_utterances_added=False, follow_chains=True):
        ''' Run the coreference model on some utterances

        Arg:
            last_utterances_added: run the coreference model over the last utterances added to the data
            follow_chains: follow coreference chains over previous utterances
        '''
        self._prepare_clusters()
        mentions = list(self.data.get_candidate_mentions(last_utterances_added=last_utterances_added))
        n_ant, antecedents = self.run_coref_on_mentions(mentions)
        mentions = antecedents.values()
        if follow_chains and n_ant > 0:
            i = 0
            while i < MAX_FOLLOW_UP:
                i += 1
                n_ant, antecedents = self.run_coref_on_mentions(mentions)
                mentions = antecedents.values()
                if n_ant == 0:
                    break

    def one_shot_coref(self, utterances, utterances_speakers_id=None, context=None,
                       context_speakers_id=None, speakers_names=None):
        ''' Clear history, load a list of utterances and run the coreference model on them

        Arg:
            utterances : iterator or list of string corresponding to successive utterances
            utterances_speaker : iterator or list of speaker id for each utterance.
                If not provided, assume two speakers speaking alternatively.
                if utterances and utterances_speaker are not of the same length padded with None
            context : same as utterances but coreferences are not computed for this,
                      only used as possible antecedent to utterances mentions
            context_speaker : same as utterances_speaker
            speakers_names : dictionnary of list of acceptable speaker names for each speaker id
        Return:
            clusters of entities with coreference resolved
        '''
        self.data.set_utterances(context, context_speakers_id, speakers_names)
        self.continuous_coref(utterances, utterances_speakers_id, speakers_names)
        return self.get_clusters()

    def continuous_coref(self, utterances, utterances_speakers_id=None, speakers_names=None):
        '''
        Same as one-shot coref but don't clear the history.
        Only resolve coreferences for the mentions in the utterances
        (but use the mentions in previously loaded utterances as possible antecedents)
        '''
        self.data.add_utterances(utterances, utterances_speakers_id, speakers_names)
        self.run_coref_on_utterances(last_utterances_added=True, follow_chains=True)
        return self.get_clusters()

    ###################################
    ###### INFORMATION RETRIEVAL ######
    ###################################

    def get_scores(self):
        ''' Retrieve single and pair scores'''
        return {"single_scores": self.mentions_single_scores,
                "pair_scores": self.mentions_pairs_scores}

    def get_clusters(self, remove_singletons=True, use_no_coref_list=True):
        ''' Retrieve cleaned clusters'''
        clusters = self.clusters
        remove_id = []
        if use_no_coref_list:
            for key, mentions in clusters.items():
                cleaned_list = []
                for mention_idx in mentions:
                    mention = self.data.mentions[mention_idx]
                    if mention.lower_ not in NO_COREF_LIST:
                        cleaned_list.append(mention_idx)
                        self.mention_to_cluster[mention_idx] = None
                clusters[key] = cleaned_list
            # Also clean up keys so we can build coref chains in self.get_most_representative
            added = {}
            for key, mentions in clusters.items():
                if self.data.mentions[key].lower_ in NO_COREF_LIST:
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
        '''
        Find a most representative mention for each cluster

        Return:
            Dictionnary of {original_mention: most_representative_resolved_mention, ...}
        '''
        clusters = self.get_clusters(remove_singletons=True, use_no_coref_list=use_no_coref_list)
        coreferences = {}
        for key in self.data.get_candidate_mentions(last_utterances_added=last_utterances_added):
            if self.mention_to_cluster[key] is None:
                continue
            mentions = clusters.get(self.mention_to_cluster[key], None)
            if mentions is None:
                continue
            representative = self.data.mentions[key]
            for mention_idx in mentions[1:]:
                mention = self.data.mentions[mention_idx]
                if mention.mention_type is not representative.mention_type:
                    if mention.mention_type == MENTION_TYPE["PROPER"] \
                        or (mention.mention_type == MENTION_TYPE["NOMINAL"] and
                                representative.mention_type == MENTION_TYPE["PRONOMINAL"]):
                        coreferences[self.data.mentions[key]] = mention
                        representative = mention

        return coreferences
