import re
from gensim.models.keyedvectors import KeyedVectors 

def load_word2vec_model(file_name):
	return KeyedVectors.load_word2vec_format(file_name, binary=True)