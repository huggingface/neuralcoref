# cython: profile=True
# cython: infer_types=True
"""Coref resolution"""

import os
import spacy
import numpy as np

from neuralcoref.train.utils import PACKAGE_DIRECTORY, SIZE_SINGLE_IN
from neuralcoref.train.compat import unicode_
from neuralcoref.train.document import Document, MENTION_TYPE, NO_COREF_LIST

#######################
##### UTILITIES #######

MAX_FOLLOW_UP = 50

#######################
###### CLASSES ########


class Model(object):
    """
    Coreference neural model
    """

    def __init__(self, model_path):
        weights, biases = [], []
        for file in sorted(os.listdir(model_path)):
            if file.startswith("single_mention_weights"):
                w = np.load(os.path.join(model_path, file))
                weights.append(w)
            if file.startswith("single_mention_bias"):
                w = np.load(os.path.join(model_path, file))
                biases.append(w)
        self.single_mention_model = list(zip(weights, biases))
        weights, biases = [], []
        for file in sorted(os.listdir(model_path)):
            if file.startswith("pair_mentions_weights"):
                w = np.load(os.path.join(model_path, file))
                weights.append(w)
            if file.startswith("pair_mentions_bias"):
                w = np.load(os.path.join(model_path, file))
                biases.append(w)
        self.pair_mentions_model = list(zip(weights, biases))

    def _score(self, features, layers):
        for weights, bias in layers:
            # print("features", features.shape)
            features = np.matmul(weights, features) + bias
            if weights.shape[0] > 1:
                features = np.maximum(features, 0)  # ReLU
        return np.sum(features, axis=0)

    def get_multiple_single_score(self, first_layer_input):
        return self._score(first_layer_input, self.single_mention_model)

    def get_multiple_pair_score(self, first_layer_input):
        return self._score(first_layer_input, self.pair_mentions_model)


class Coref(object):
    """
    Main coreference resolution algorithm
    """

    def __init__(
        self,
        nlp=None,
        greedyness=0.5,
        max_dist=50,
        max_dist_match=500,
        conll=None,
        blacklist=True,
        debug=False,
    ):
        self.greedyness = greedyness
        self.max_dist = max_dist
        self.max_dist_match = max_dist_match
        self.debug = debug
        model_path = os.path.join(
            PACKAGE_DIRECTORY, "weights/conll/" if conll is not None else "weights/"
        )
        model_path = os.path.join(PACKAGE_DIRECTORY, "weights/")
        print("Loading neuralcoref model from", model_path)
        self.coref_model = Model(model_path)
        if nlp is None:
            print("Loading spacy model")
            try:
                spacy.info("en_core_web_sm")
                model = "en_core_web_sm"
            except IOError:
                print("No spacy 2 model detected, using spacy1 'en' model")
                spacy.info("en")
                model = "en"
            nlp = spacy.load(model)
        self.data = Document(
            nlp, conll=conll, blacklist=blacklist, model_path=model_path
        )
        self.clusters = {}
        self.mention_to_cluster = []
        self.mentions_single_scores = {}
        self.mentions_pairs_scores = {}

    ###################################
    #### ENTITY CLUSTERS FUNCTIONS ####
    ###################################

    def _prepare_clusters(self):
        """
        Clean up and prepare one cluster for each mention
        """
        self.mention_to_cluster = list(range(len(self.data.mentions)))
        self.clusters = dict((i, [i]) for i in self.mention_to_cluster)
        self.mentions_single_scores = {}
        self.mentions_pairs_scores = {}
        for mention in self.mention_to_cluster:
            self.mentions_single_scores[mention] = None
            self.mentions_pairs_scores[mention] = {}

    def _merge_coreference_clusters(self, ant_idx, mention_idx):
        """
        Merge two clusters together
        """
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
        """
        Print clusters informations
        """
        print(self.clusters)
        for key, mentions in self.clusters.items():
            print(
                "cluster",
                key,
                "(",
                ", ".join(unicode_(self.data[m]) for m in mentions),
                ")",
            )

    ###################################
    ####### MAIN COREF FUNCTIONS ######
    ###################################

    def run_coref_on_mentions(self, mentions):
        """
        Run the coreference model on a mentions list
        """
        best_ant = {}
        best_score = {}
        n_ant = 0
        inp = np.empty((SIZE_SINGLE_IN, len(mentions)))
        for i, mention_idx in enumerate(mentions):
            mention = self.data[mention_idx]
            print("mention.embedding", mention.embedding.shape)
            inp[: len(mention.embedding), i] = mention.embedding
            inp[: len(mention.embedding), i] = mention.features
            inp[: len(mention.embedding), i] = self.data.genre
        score = self.coref_model.get_multiple_single_score(inp.T)
        for mention_idx, s in zip(mentions, score):
            self.mentions_single_scores[mention_idx] = s
            best_score[mention_idx] = s - 50 * (self.greedyness - 0.5)

        for mention_idx, ant_list in self.data.get_candidate_pairs(
            mentions, self.max_dist, self.max_dist_match
        ):
            if len(ant_list) == 0:
                continue
            inp_l = []
            for ant_idx in ant_list:
                mention = self.data[mention_idx]
                antecedent = self.data[ant_idx]
                feats_, pwf = self.data.get_pair_mentions_features(antecedent, mention)
                inp_l.append(pwf)
            inp = np.stack(inp_l, axis=0)
            score = self.coref_model.get_multiple_pair_score(inp.T)
            for ant_idx, s in zip(ant_list, score):
                self.mentions_pairs_scores[mention_idx][ant_idx] = s
                if s > best_score[mention_idx]:
                    best_score[mention_idx] = s
                    best_ant[mention_idx] = ant_idx
            if mention_idx in best_ant:
                n_ant += 1
                self._merge_coreference_clusters(best_ant[mention_idx], mention_idx)
        return (n_ant, best_ant)

    def run_coref_on_utterances(
        self, last_utterances_added=False, follow_chains=True, debug=False
    ):
        """ Run the coreference model on some utterances

        Arg:
            last_utterances_added: run the coreference model over the last utterances added to the data
            follow_chains: follow coreference chains over previous utterances
        """
        if debug:
            print("== run_coref_on_utterances == start")
        self._prepare_clusters()
        if debug:
            self.display_clusters()
        mentions = list(
            self.data.get_candidate_mentions(
                last_utterances_added=last_utterances_added
            )
        )
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
        if debug:
            self.display_clusters()
        if debug:
            print("== run_coref_on_utterances == end")

    def one_shot_coref(
        self,
        utterances,
        utterances_speakers_id=None,
        context=None,
        context_speakers_id=None,
        speakers_names=None,
    ):
        """ Clear history, load a list of utterances and an optional context and run the coreference model on them

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
        """
        self.data.set_utterances(context, context_speakers_id, speakers_names)
        self.continuous_coref(utterances, utterances_speakers_id, speakers_names)
        return self.get_clusters()

    def continuous_coref(
        self, utterances, utterances_speakers_id=None, speakers_names=None
    ):
        """
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
        """
        self.data.add_utterances(utterances, utterances_speakers_id, speakers_names)
        self.run_coref_on_utterances(last_utterances_added=True, follow_chains=True)
        return self.get_clusters()

    ###################################
    ###### INFORMATION RETRIEVAL ######
    ###################################

    def get_utterances(self, last_utterances_added=True):
        """ Retrieve the list of parsed uterrances"""
        if last_utterances_added and len(self.data.last_utterances_loaded):
            return [
                self.data.utterances[idx] for idx in self.data.last_utterances_loaded
            ]
        else:
            return self.data.utterances

    def get_resolved_utterances(self, last_utterances_added=True, blacklist=True):
        """ Return a list of utterrances text where the """
        coreferences = self.get_most_representative(last_utterances_added, blacklist)
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
                    resolved_utt += (
                        " " if token.whitespace_ and resolved_utt[-1] is not " " else ""
                    )
                    in_coref = None
            resolved_utterances.append(resolved_utt)
        return resolved_utterances

    def get_mentions(self):
        """ Retrieve the list of mentions"""
        return self.data.mentions

    def get_scores(self):
        """ Retrieve scores for single mentions and pair of mentions"""
        return {
            "single_scores": self.mentions_single_scores,
            "pair_scores": self.mentions_pairs_scores,
        }

    def get_clusters(self, remove_singletons=False, blacklist=False):
        """ Retrieve cleaned clusters"""
        clusters = self.clusters
        mention_to_cluster = self.mention_to_cluster
        remove_id = []
        if blacklist:
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

    def get_most_representative(self, last_utterances_added=True, blacklist=True):
        """
        Find a most representative mention for each cluster

        Return:
            Dictionnary of {original_mention: most_representative_resolved_mention, ...}
        """
        clusters, _ = self.get_clusters(remove_singletons=True, blacklist=blacklist)
        coreferences = {}
        for key in self.data.get_candidate_mentions(
            last_utterances_added=last_utterances_added
        ):
            if self.mention_to_cluster[key] is None:
                continue
            mentions = clusters.get(self.mention_to_cluster[key], None)
            if mentions is None:
                continue
            representative = self.data.mentions[key]
            for mention_idx in mentions[1:]:
                mention = self.data.mentions[mention_idx]
                if mention.mention_type is not representative.mention_type:
                    if mention.mention_type == MENTION_TYPE["PROPER"] or (
                        mention.mention_type == MENTION_TYPE["NOMINAL"]
                        and representative.mention_type == MENTION_TYPE["PRONOMINAL"]
                    ):
                        coreferences[self.data.mentions[key]] = mention
                        representative = mention

        return coreferences
