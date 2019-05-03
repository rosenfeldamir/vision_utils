import pandas as pd


############################# Printing Utilities ##############################

def map_level_to_str(level):
  S = '\t'*level
  #S = '-'*level+'>'
  return S
def pprint_dict_keys(D,level=0):
	'''
	print python dictionary as a tree structure for exploration.
	'''
	if type(D) is dict:
		for k in D.keys():
	  		print(map_level_to_str(level)+k)
	  		pprint_dict_keys(D[k],level+1)
	else:
		print(map_level_to_str(level)+'{},{}'.format(type(D),len(D)))

############################## Latex ##########################################

def to_latex(df,fn):	
	'''
	write a dataframe as a latex table to a file. 
	'''
	with open(fn,'w') as tf:
		tf.write('\\documentclass{article}\
				\\usepackage{booktabs}\
				\\begin{document}')
		tf.write(df.to_latex())
		tf.write('\end{document}')

############################## Misc ##########################################


def dict_to_param_str(d):
	'''
	Represents a dictionary as a concatenated list of key1_value1_key2_value2_...
	'''
	s = []
	for k,v in d.items():
		s.append(k)
		s.append(v)
	return '_'.join(map(str,s))


############################## Command Line Processing ########################

def str2bool(v):  # https://stackoverflow.com/posts/43357954/revisions
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')