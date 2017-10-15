import numpy as np 
import gensim
import pandas as pd 
from pandas import DataFrame, Series
from utils import load_word2vec_model
from data_collection import get_data
from sklearn import svm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

class TextAnalyzer(object):

	def __init__(self, word2vec_model):
		self._model = word2vec_model

	def text_to_vectors(self, text):
		words = [w for w in text if w in self._model]
		if len(words) != 0:
			for w in words:
				yield self._model[w]

	def text_to_average_vector(self, text):
		vectors = self.text_to_vectors(text)
		vectors_sum = next(vectors, None)
		if vectors_sum is None:
			return None
		count = 1.
		for v in vectors:
			count += 1.
			vectors_sum = np.add(vectors_sum, v)
		return np.nan_to_num(vectors_sum / count)






def get_vector(analyzer, sentence):
	vector = analyzer.text_to_average_vector(sentence)
	return vector


def get_X_y(analyzer, data, sentences):
	X = list()
	y = list()
	for ix, sentence in enumerate(sentences):
		vector = get_vector(analyzer, sentence)
		if not vector is None:
			X.append(vector)
			y.append(data[ix][1])
	return (X, y)

if __name__=="__main__":

	# data[0] -> text
	# data[1] -> label (1 or 0)
	data, sentences = get_data("train.json")

	model = load_word2vec_model("GoogleNews-vectors-negative300.bin")
	ta = TextAnalyzer(model)

	X, y = get_X_y(ta, data, sentences)

	clf = svm.SVC()
	clf.fit(X, y)

	test_data, test_sentences = get_data("test.json")
	X, y = get_X_y(ta, test_data, test_sentences)
	
	correct = 0
	total = len(X)
	for ix, x in enumerate(X):
		pred = clf.predict(x)
		if pred == y[ix]:
			correct += 1


	print("%d correct out of %d total... %0.2f percent correct." % (correct, total, (correct / total) * 100))