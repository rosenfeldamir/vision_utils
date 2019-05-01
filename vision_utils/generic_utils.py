import pandas as pd
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