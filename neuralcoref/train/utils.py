"""Utils"""


from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np
from tqdm import tqdm

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE_PATH = os.path.join(
    PACKAGE_DIRECTORY, "test_batch_size.txt"
)  # fernandes.txt")#

SIZE_SPAN = 250  # size of the span vector (averaged word embeddings)
SIZE_WORD = 8  # number of words in a mention (tuned embeddings)
SIZE_EMBEDDING = 50  # size of the words embeddings
SIZE_FP = 70  # number of features for a pair of mention
SIZE_FP_COMPRESSED = (
    9
)  # size of the features for a pair of mentions as stored in numpy arrays
SIZE_FS = 24  # number of features of a single mention
SIZE_FS_COMPRESSED = 6  # size of the features for a mention as stored in numpy arrays
SIZE_GENRE = 7  # Size of the genre one-hot array
SIZE_MENTION_EMBEDDING = (
    SIZE_SPAN + SIZE_WORD * SIZE_EMBEDDING
)  # A mention embeddings (span + words vectors)
SIZE_SNGL_FEATS = SIZE_FS - SIZE_GENRE
SIZE_PAIR_FEATS = SIZE_FP - SIZE_GENRE
SIZE_SNGL_IN_NO_GENRE = SIZE_MENTION_EMBEDDING + SIZE_SNGL_FEATS
SIZE_PAIR_IN_NO_GENRE = 2 * SIZE_MENTION_EMBEDDING + SIZE_PAIR_FEATS

SIZE_PAIR_IN = (
    2 * SIZE_MENTION_EMBEDDING + SIZE_FP
)  # Input to the mentions pair neural network
SIZE_SINGLE_IN = (
    SIZE_MENTION_EMBEDDING + SIZE_FS
)  # Input to the single mention neural network

DISTANCE_BINS = list(range(5)) + [5] * 3 + [6] * 8 + [7] * 16 + [8] * 32
BINS_NUM = float(len(DISTANCE_BINS))
MAX_BINS = DISTANCE_BINS[-1] + 1


def encode_distance(x):
    """ Encode an integer or an array of integers as a (bined) one-hot numpy array """

    def _encode_distance(d):
        """ Encode an integer as a (bined) one-hot numpy array """
        dist_vect = np.zeros((11,), dtype="float32")
        if d < 64:
            dist_vect[DISTANCE_BINS[d]] = 1
        else:
            dist_vect[9] = 1
        dist_vect[10] = min(float(d), BINS_NUM) / BINS_NUM
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
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [
            function(**a) if use_kwargs else function(a) for a in array[:front_num]
        ]
    else:
        front = []
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [
            function(**a) if use_kwargs else function(a)
            for a in tqdm(array[front_num:])
        ]
    # Assemble the workers
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            "total": len(futures),
            "unit": "it",
            "unit_scale": True,
            "leave": True,
        }
        # #Print out the progress as tasks complete
        # for _ in tqdm(as_completed(futures), **kwargs):
        #     pass
    out = []
    # Get the results from the futures.
    for future in futures:  # tqdm(futures):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out
