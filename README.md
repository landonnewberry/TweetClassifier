# TweetClassifier
Used Word2Vec and sklearn SVMs to classify tweets. In this example, all train/test tweets contain the keyword landslide. Based on labeled training set, the classifier is able to determine whether or not the tweet is referencing a natural disaster rather than quoting a song or a political landslide.

## Running the code:
Clone the repository and add "*GoogleNews-vectors-negative300.bin*" to the main directory with TextAnalyzer.py.

To begin: `source bin/activate`

The training data is stored in *train.json*, evaluation data is in *test.json*.

To run the classifier, `python TextAnalyzer.py`. It may take a while.

Report is stored in *report.txt*.
