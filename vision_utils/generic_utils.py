import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook


############################# Splitting into batches ##########################
# source: https://stackoverflow.com/a/312464/4395366
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

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


def df_to_latex(df, fn):
    '''
    write a dataframe as a latex table to a file. 
    '''
    with open(fn, 'w') as tf:
        tf.write('\\documentclass{article}\
				\\usepackage{booktabs}\
				\\begin{document}')
        tf.write(df.to_latex(index=False))
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

##################### Quickly draw text on image ################################
from PIL import ImageDraw, ImageFont
def render_image_w_text(img,cur_title,font_factor=50,fill=(255,0,0)):
    was_array=False
    if type(img) is not PIL.Image.Image: # assume an array
        was_array = True
        img = PIL.Image.fromarray(img)
    img = deepcopy(img)
    draw = ImageDraw.Draw(img)    
    y_gap = 15
    font_size = int(font_factor * img.width/600)
    font = ImageFont.truetype('/usr/share/fonts/truetype/FreeMonoBold.ttf',font_size)
    textsize=draw.textsize(cur_title,font=font)
    xy = (5,5)
    draw.rectangle([xy,(xy[0]+textsize[0],xy[1]+textsize[1])], fill=(255,0,0))
    draw.text((5,5), cur_title, fill, font=font)    
    return img