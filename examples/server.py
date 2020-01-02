#!/usr/bin/env python
"""Coreference resolution server example.
A simple server serving the coreference system.
"""

import json
from wsgiref.simple_server import make_server
import falcon
import spacy
import neuralcoref

# Python 3
unicode_ = str


class AllResource(object):
    def __init__(self):
        self.nlp = spacy.load("en")
        neuralcoref.add_to_pipe(self.nlp)
        print("Server loaded")
        self.response = None

    def on_get(self, req, resp):
        self.response = {}

        text_param = req.get_param_as_list("text")
        print("text: ", text_param)
        if text_param is not None:
            text = ",".join(text_param) if isinstance(text_param, list) else text_param
            text = unicode_(text)
            doc = self.nlp(text)
            if doc._.has_coref:
                mentions = [
                    {
                        "start": mention.start_char,
                        "end": mention.end_char,
                        "text": mention.text,
                        "resolved": cluster.main.text,
                    }
                    for cluster in doc._.coref_clusters
                    for mention in cluster.mentions
                ]
                clusters = list(
                    list(span.text for span in cluster)
                    for cluster in doc._.coref_clusters
                )
                resolved = doc._.coref_resolved
                self.response["mentions"] = mentions
                self.response["clusters"] = clusters
                self.response["resolved"] = resolved

        resp.body = json.dumps(self.response)
        resp.content_type = "application/json"
        resp.append_header("Access-Control-Allow-Origin", "*")
        resp.status = falcon.HTTP_200


if __name__ == "__main__":
    RESSOURCE = AllResource()
    APP = falcon.API()
    APP.add_route("/", RESSOURCE)
    HTTPD = make_server("0.0.0.0", 8000, APP)
    HTTPD.serve_forever()
