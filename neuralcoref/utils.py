# coding: utf8
"""Utils"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import spacy
from tqdm import tqdm

def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=10):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

def mention_detection_debug(sentence):
    print(u"🌋 Loading spacy model")
    try:
        spacy.info('en_core_web_sm')
        model = 'en_core_web_sm'
    except IOError:
        print("No spacy 2 model detected, using spacy1 'en' model")
        spacy.info('en')
        model = 'en'
    nlp = spacy.load(model)
    doc = nlp(sentence.decode('utf-8'))
    mentions = extract_mentions_spans(doc, use_no_coref_list=False, debug=True)
    for mention in mentions:
        print(mention)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        sent = sys.argv[1]
        mention_detection_debug(sent)
