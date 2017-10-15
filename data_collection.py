from pandas import DataFrame
import json
from gensim.models import Word2Vec
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# TRAINING DATA "train.json"
# TESTING DATA  "test.json"

def _strip_string(s):
	s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
	s = re.sub(r"\'s", "", s)
	s = re.sub(r"\'ve", "", s)
	s = re.sub(r"n\'t", "", s)
	s = re.sub(r"\'re", "", s)
	s = re.sub(r"\'d", "", s)
	s = re.sub(r"\'ll", "", s)
	s = re.sub(r",", "", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", "", s)
	s = re.sub(r"\)", "", s)
	s = re.sub(r"\?", "", s)
	s = re.sub(r"\s{2,}", " ", s)
	return s

def get_data(value):
	func = hasattr(value, "read") and (lambda x: x) or (lambda x: open(x))
	data = list()
	with func(value) as f:
		for line in f:
			j = json.loads(line)
			label = j["label"] == "relevant" and 1 or 0
			data.append([str.lower(j["text"]), label])

#	data["binary_label"] = data.apply(lambda row: row["label"] == "relevant" and 1 or 0, axis=1)
	list_text = data[0]

	sentences = [[_strip_string(word).strip() for word in row[0].split()] for row in data]

	"""
	model = Word2Vec(sentences, size=100  , window=10, min_count=5, workers=4, hs=1, negative=0)
	sentences = [[i for i in s if i in model.wv.index2word] for s in sentences]

	vectorizer = TfidfVectorizer(min_df=0)
	X = vectorizer.fit_transform(" ".join(item) for item in sentences)
	scaler = StandardScaler()
	idf = scaler.fit_transform(vectorizer.idf_.reshape(-1, 1))
	tfidf = dict(zip(vectorizer.get_feature_names(), idf))
	for k,v in tfidf.items(): tfidf[k] = v[0]
	"""

	return (data, sentences)


def get_average_vectors(model, sentences, tfidf):
	avg_vectors = []
	for sentence in sentences:
		if len(sentence) > 0:
			avg_vectors.append(sum([model.wv[word] * tfidf[word] for word in sentence if word in model.wv and word in tfidf]) / len(sentence))
		else:
			avg_vectors.append([])
	return avg_vectors









