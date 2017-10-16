import numpy as np 
import gensim
import pandas as pd 
from pandas import DataFrame, Series
from utils import load_word2vec_model
from data_collection import get_data
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support

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

	REPORT = "report.txt"

	# data[0] -> text
	# data[1] -> label (1 or 0)
	# data[2] -> id
	data, sentences = get_data("train.json")

	model = load_word2vec_model("GoogleNews-vectors-negative300.bin")
	ta = TextAnalyzer(model)

	X, y = get_X_y(ta, data, sentences)

	clf = svm.SVC()
	clf.fit(X, y)

	test_data, test_sentences = get_data("test.json")
	X, y = get_X_y(ta, test_data, test_sentences)


	test_ids = [item[2] for item in test_data]
	true_pos = false_pos = true_neg = false_neg = 0

	y_pred = list()
	for ix, x in enumerate(X):
		pred = clf.predict(x)
		y_pred.append(pred)
		if pred == 1 and y[ix] == 1:
			true_pos += 1
		elif pred == 1 and y[ix] != 1:
			false_pos += 1
		elif pred == 0 and y[ix] == 0:
			true_neg += 1
		else:
			false_neg += 1

	#precision, recall, fscore, _ = precision_recall_fscore_support(y, y_pred)

	#print("Precision = %.2f, Recall = %.2f, F-Score = %.2f" % (precision, recall, fscore))
	print("True Pos = %d, False Pos = %d, True Neg = %d, False Neg = %d" % (true_pos, false_pos, true_neg, false_neg))
	precision = true_pos / (true_pos + false_pos)
	recall = true_pos / (true_pos + false_neg)
	f1score = 2 * ((precision * recall) / (precision + recall))
	print("Precision = %.3f, Recall = %.3f, F1 Score = %.3f" % (precision, recall, f1score))

	with open(REPORT, "w") as f:
		f.write("True positives: %d\tFalse positives: %d\tTrue negatives: %d\tFalse negatives: %d\n" % (true_pos, false_pos, true_neg, false_neg))
		f.write("Precision: %f\tRecall: %f\tF1-score: %f\n" % (precision, recall, f1score))
		for ix, v in enumerate(test_ids):
			f.write("%d\t%s\n" % (v, (y_pred[ix] == 1 and "relevant" or "irrelevant")))