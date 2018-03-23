#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Coreference resolution server example.
A simple server serving the coreference system.
"""
from __future__ import unicode_literals

import json
from wsgiref.simple_server import make_server
import falcon

from neuralcoref.algorithm import Coref
from neuralcoref.document import MENTION_LABEL
from neuralcoref.compat import unicode_

class CorefWrapper(Coref):
    def parse_and_get_mentions(self, utterances, utterances_speakers_id=None, context=None,
                               context_speakers_id=None, speakers_names=None):
        self.data.set_utterances(context, context_speakers_id, speakers_names)
        self.data.add_utterances(utterances, utterances_speakers_id, speakers_names)

    def run_coref(self):
        self.run_coref_on_utterances(last_utterances_added=True, follow_chains=True)
        coreferences = self.get_most_representative(use_no_coref_list=False)

        json_mentions = [{'index':          mention.index,
                          'start':          mention.start_char,
                          'end':            mention.end_char,
                          'utterance':      mention.utterance_index,
                          'type':           MENTION_LABEL[mention.mention_type],
                          'text':           mention.text} for mention in self.data.mentions]
        json_coreferences = [{'original': original.text,
                              'resolved': resolved.text} for original, resolved in coreferences.items()]
        scores = self.get_scores()
        return {"coreferences": json_coreferences,
                "mentions": json_mentions,
                "singleScores": scores["single_scores"],
                "pairScores": scores["pair_scores"]}


class AllResource(object):
    def __init__(self):
        self.coref = CorefWrapper()
        self.response = {}
        print("Server loaded")

    def on_get(self, req, resp):
        self.response = {}

        text_param = req.get_param("text")
        if text_param is not None:
            text = ",".join(text_param) if isinstance(text_param, list) else text_param
            text = unicode_(text)

            context = req.get_param_as_list("context")
            if context:
                context = [unicode_(utt) for utt in context]
            text_speaker = req.get_param("textspeaker")
            context_speakers = req.get_param_as_list("contextspeakers")
            speakers_names = req.get_param_as_dict("speakersnames")
            print("text", text)
            print("context", context)
            self.coref.parse_and_get_mentions(text, text_speaker,
                                              context, context_speakers,
                                              speakers_names)

            self.response.update(self.coref.run_coref())

        resp.body = json.dumps(self.response)
        resp.content_type = 'application/json'
        resp.append_header('Access-Control-Allow-Origin', "*")
        resp.status = falcon.HTTP_200

if __name__ == '__main__':
    RESSOURCE = AllResource()
    APP = falcon.API()
    APP.add_route('/', RESSOURCE)
    HTTPD = make_server('0.0.0.0', 8000, APP)
    HTTPD.serve_forever()
