# coding: utf8
"""Coref resolution"""

from __future__ import unicode_literals
from __future__ import print_function

import sys
import os
import spacy
import numpy as np

from compat import unicode_
from document import Document, MENTION_TYPE, NO_COREF_LIST

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

#######################
##### UTILITIES #######

MAX_FOLLOW_UP = 50

#######################
###### CLASSES ########

class Model(object):
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

    def _score(self, features, layers, debug=True):
        if debug:
            print("single_input.size", features.shape)
            print("single_input", unicode_(features[0:8]))
        for weights, bias in layers:
            features = np.matmul(weights, features) + bias
            if weights.shape[0] > 1:
                features = np.maximum(features, 0) # ReLU
            if debug:
                print("single_top_layer")
                print("single_top_layer.weight.size", weights.shape)
                print("self.single_top_layer.weight", weights[0:8, 0])
                print("single_top_layer.bias.size", bias.shape)
                print("self.single_top_layer.bias", bias[0:8, 0])
                print("features out shape", unicode_(features.shape))
                print("features out", unicode_(features[0:8]))
        score = np.sum(features)
        if debug: print("score", score)
        sys.exit()
        return score

    def get_single_mention_score(self, mention, single_features):
        print("📚 mention", mention)
        print("mention.spans_embeddings", mention.spans_embeddings[0:8])
        print("mention.words_embeddings.shape", mention.words_embeddings.shape)
        print("mention.words_embeddings", mention.words_embeddings[::50])
        print("mention.single_features", mention.features[0:8])
        first_layer_input = np.concatenate([mention.embedding,
                                            single_features], axis=0)[:, np.newaxis]
        return self._score(first_layer_input, self.single_mention_model)

    def get_pair_mentions_score(self, antecedent, mention, pair_features):
        print("📚 PAIR ", mention, antecedent)
        first_layer_input = np.concatenate([antecedent.embedding,
                                            mention.embedding,
                                            pair_features], axis=0)[:, np.newaxis]
        return self._score(first_layer_input, self.pair_mentions_model)


class Coref(object):
    '''
    Main coreference resolution algorithm
    '''
    def __init__(self, nlp=None, greedyness=0.5, max_dist=50, max_dist_match=500, conll=None,
                 use_no_coref_list=True, coref_model=None, data=None, debug=False):
        self.greedyness = greedyness
        self.max_dist = max_dist
        self.max_dist_match = max_dist_match
        self.debug = debug
        
        if coref_model is None:
            model_path = os.path.join(PACKAGE_DIRECTORY, "weights/conll/" if conll is not None else "weights/")
            print("loading model from", model_path)
            self.coref_model = Model(model_path)
        else:
            self.coref_model = coref_model

        if data is None:
            if nlp is None:
                print("Loading spacy model")
                try:
                    spacy.info('en_core_web_sm')
                    model = 'en_core_web_sm'
                except IOError:
                    print("No spacy 2 model detected, using spacy1 'en' model")
                    spacy.info('en')
                    model = 'en'
                nlp = spacy.load(model)

            #embed_model_path = os.path.join(PACKAGE_DIRECTORY, "weights/")
            self.data = Document(nlp, conll=conll, use_no_coref_list=use_no_coref_list)
        else:
            self.data = data

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

    def remove_singletons_clusters(self):
        remove_id = []
        for key, mentions in self.clusters.items():
            if len(mentions) == 1:
                remove_id.append(key)
                self.mention_to_cluster[key] = None
        for rem in remove_id:
            del self.clusters[rem]

    def display_clusters(self):
        '''
        Print clusters informations
        '''
        print(self.clusters)
        for key, mentions in self.clusters.items():
            print("cluster", key, "(", ", ".join(unicode_(self.data[m]) for m in mentions), ")")

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
            feats_, ana_feats = self.data.get_single_mention_features(mention)
            single_score = self.coref_model.get_single_mention_score(mention, ana_feats)
            self.mentions_single_scores[mention_idx] = single_score
            self.mentions_single_features[mention_idx] = {"spansEmbeddings": mention.spans_embeddings_, "wordsEmbeddings": mention.words_embeddings_, "features": feats_}

            best_score = single_score - 50 * (self.greedyness - 0.5)
            for ant_idx in ant_list:
                antecedent = self.data[ant_idx]
                feats_, pwf = self.data.get_pair_mentions_features(antecedent, mention)
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

    def run_coref_on_utterances(self, last_utterances_added=False, follow_chains=True, debug=False):
        ''' Run the coreference model on some utterances

        Arg:
            last_utterances_added: run the coreference model over the last utterances added to the data
            follow_chains: follow coreference chains over previous utterances
        '''
        if debug: print("== run_coref_on_utterances == start")
        self._prepare_clusters()
        if debug: self.display_clusters()
        mentions = list(self.data.get_candidate_mentions(last_utterances_added=last_utterances_added))
        n_ant, antecedents = self.run_coref_on_mentions(mentions)
        mentions = antecedents.values()
        if follow_chains and last_utterances_added and n_ant > 0:
            i = 0
            while i < MAX_FOLLOW_UP:
                i += 1
                n_ant, antecedents = self.run_coref_on_mentions(mentions)
                mentions = antecedents.values()
                if n_ant == 0:
                    break
        if debug: self.display_clusters()
        if debug: print("== run_coref_on_utterances == end")

    def one_shot_coref(self, utterances, utterances_speakers_id=None, context=None,
                       context_speakers_id=None, speakers_names=None):
        ''' Clear history, load a list of utterances and an optional context and run the coreference model on them

        Arg:
        - `utterances` : iterator or list of string corresponding to successive utterances (in a dialogue) or sentences.
            Can be a single string for non-dialogue text.
        - `utterances_speakers_id=None` : iterator or list of speaker id for each utterance (in the case of a dialogue).
            - if not provided, assume two speakers speaking alternatively.
            - if utterances and utterances_speaker are not of the same length padded with None
        - `context=None` : iterator or list of string corresponding to additionnal utterances/sentences sent prior to `utterances`. Coreferences are not computed for the mentions identified in `context`. The mentions in `context` are only used as possible antecedents to mentions in `uterrance`. Reduce the computations when we are only interested in resolving coreference in the last sentences/utterances.
        - `context_speakers_id=None` : same as `utterances_speakers_id` for `context`. 
        - `speakers_names=None` : dictionnary of list of acceptable speaker names (strings) for speaker_id in `utterances_speakers_id` and `context_speakers_id`
        Return:
            clusters of entities with coreference resolved
        '''
        self.data.set_utterances(context, context_speakers_id, speakers_names)
        self.continuous_coref(utterances, utterances_speakers_id, speakers_names)
        return self.get_clusters()

    def continuous_coref(self, utterances, utterances_speakers_id=None, speakers_names=None):
        '''
        Only resolve coreferences for the mentions in the utterances
        (but use the mentions in previously loaded utterances as possible antecedents)
        Arg:
            utterances : iterator or list of string corresponding to successive utterances
            utterances_speaker : iterator or list of speaker id for each utterance.
                If not provided, assume two speakers speaking alternatively.
                if utterances and utterances_speaker are not of the same length padded with None
            speakers_names : dictionnary of list of acceptable speaker names for each speaker id
        Return:
            clusters of entities with coreference resolved
        '''
        self.data.add_utterances(utterances, utterances_speakers_id, speakers_names)
        self.run_coref_on_utterances(last_utterances_added=True, follow_chains=True)
        return self.get_clusters()

    ###################################
    ###### INFORMATION RETRIEVAL ######
    ###################################

    def get_utterances(self, last_utterances_added=True):
        ''' Retrieve the list of parsed uterrances'''
        if last_utterances_added and len(self.data.last_utterances_loaded):
            return [self.data.utterances[idx] for idx in self.data.last_utterances_loaded]
        else:
            return self.data.utterances

    def get_resolved_utterances(self, last_utterances_added=True, use_no_coref_list=True):
        ''' Return a list of utterrances text where the '''
        coreferences = self.get_most_representative(last_utterances_added, use_no_coref_list)
        resolved_utterances = []
        for utt in self.get_utterances(last_utterances_added=last_utterances_added):
            resolved_utt = ""
            in_coref = None
            for token in utt:
                if in_coref is None:
                    for coref_original, coref_replace in coreferences.items():
                        if coref_original[0] == token:
                            in_coref = coref_original
                            resolved_utt += coref_replace.text.lower()
                            break
                    if in_coref is None:
                        resolved_utt += token.text_with_ws
                if in_coref is not None and token == in_coref[-1]:
                    resolved_utt += ' ' if token.whitespace_ and resolved_utt[-1] is not ' ' else ''
                    in_coref = None
            resolved_utterances.append(resolved_utt)
        return resolved_utterances

    def get_mentions(self):
        ''' Retrieve the list of mentions'''
        return self.data.mentions

    def get_scores(self):
        ''' Retrieve scores for single mentions and pair of mentions'''
        return {"single_scores": self.mentions_single_scores,
                "pair_scores": self.mentions_pairs_scores}

    def get_clusters(self, remove_singletons=False, use_no_coref_list=False):
        ''' Retrieve cleaned clusters'''
        clusters = self.clusters
        mention_to_cluster = self.mention_to_cluster
        remove_id = []
        if use_no_coref_list:
            for key, mentions in clusters.items():
                cleaned_list = []
                for mention_idx in mentions:
                    mention = self.data.mentions[mention_idx]
                    if mention.lower_ not in NO_COREF_LIST:
                        cleaned_list.append(mention_idx)
                clusters[key] = cleaned_list
            # Also clean up keys so we can build coref chains in self.get_most_representative
            added = {}
            for key, mentions in clusters.items():
                if self.data.mentions[key].lower_ in NO_COREF_LIST:
                    remove_id.append(key)
                    mention_to_cluster[key] = None
                    if mentions:
                        added[mentions[0]] = mentions
            for rem in remove_id:
                del clusters[rem]
            clusters.update(added)

        if remove_singletons:
            remove_id = []
            for key, mentions in clusters.items():
                if len(mentions) == 1:
                    remove_id.append(key)
                    mention_to_cluster[key] = None
            for rem in remove_id:
                del clusters[rem]

        return clusters, mention_to_cluster

    def get_most_representative(self, last_utterances_added=True, use_no_coref_list=True):
        '''
        Find a most representative mention for each cluster

        Return:
            Dictionnary of {original_mention: most_representative_resolved_mention, ...}
        '''
        clusters, _ = self.get_clusters(remove_singletons=True, use_no_coref_list=use_no_coref_list)
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
