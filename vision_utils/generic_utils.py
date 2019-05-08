import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_notebook

############################# Printing Utilities ##############################


def map_level_to_str(level):
    S = '\t' * level
    #S = '-'*level+'>'
    return S


def pprint_dict_keys(D, level=0):
    '''
    print python dictionary as a tree structure for exploration.
    '''
    if type(D) is dict:
        for k in D.keys():
            print(map_level_to_str(level) + k)
            pprint_dict_keys(D[k], level + 1)
    else:
        print(map_level_to_str(level) + '{},{}'.format(type(D), len(D)))

############################## Latex ##########################################


def to_latex(df, fn):
    '''
    write a dataframe as a latex table to a file. 
    '''
    with open(fn, 'w') as tf:
        tf.write('\\documentclass{article}\
				\\usepackage{booktabs}\
				\\begin{document}')
        tf.write(df.to_latex())
        tf.write('\end{document}')

############################## Command Line Processing ########################

def str2bool(v):  # https://stackoverflow.com/posts/43357954/revisions
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


############################## Misc ##########################################

def dict_to_param_str(d):
    '''
    Represents a dictionary as a concatenated list of key1_value1_key2_value2_...
    '''
    s = []
    for k, v in d.items():
        s.append(k)
        s.append(v)
    return '_'.join(map(str, s))


def find(a):
    return np.nonzero(a)[0]


# for progress bars.
import re
import psutil


def is_notebook():
    return any(
        re.search(r'\bjupyter-(lab|notebook)\b', x)
        for x in psutil.Process().parent().cmdline()
    )


def tqdm_fn(*args, **kwargs):
    if is_notebook():
        return tqdm_notebook(*args, **kwargs)
    return tqdm(*args, **kwargs)

# formatting serial output.

def fn_format(rootdir, i, prefix='', suffix='', n_pad=5):
    formatted_i = str(i).zfill(n_pad)
    return os.path.join(rootdir, '{}_{}.{}'.format(prefix, formatted_i, suffix))

# useful for caching stuff , then reloading instead of recalculating each time


def calc_or_load(fn, out_path, *args, **kwargs):
    if os.path.isfile(out_path):
        f = pickle.load(open(out_path, 'rb'))
    else:
        f = fn(*args, **kwargs)
        pickle.dump(f, open(out_path, 'wb'))
    return f
