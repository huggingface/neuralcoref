# coding: utf8
"""Utils"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

DISTANCE_BINS = list(range(5)) + [5]*3 + [6]*8 + [7]*16 +[8]*32

def encode_distance(x):
    ''' Encode an integer or an array of integers as a (bined) one-hot numpy array '''
    def _encode_distance(d):
        ''' Encode an integer as a (bined) one-hot numpy array '''
        dist_vect = np.zeros((11,))
        if d < 64:
            dist_vect[DISTANCE_BINS[d]] = 1
        else:
            dist_vect[9] = 1
        dist_vect[10] = min(float(d), 64.0) / 64.0
        return dist_vect

    if isinstance(x, np.ndarray):
        arr_l = [_encode_distance(y)[np.newaxis, :] for y in x]
        out_arr = np.concatenate(arr_l)
    else:
        out_arr = _encode_distance(x)
    return out_arr

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
    else:
        front = []
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
        for _ in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for future in tqdm(futures):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out
