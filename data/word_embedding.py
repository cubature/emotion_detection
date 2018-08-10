# -*- coding: UTF-8 -*-
import logging
import os
import _pickle as pkl
import numpy as np

from gensim.models import word2vec


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Sentences(object):
	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		for fname in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, fname)):
				sentence = line.split(';')[0]
				yield sentence.split()


sentences = Sentences('./corpus/processed')
model = word2vec.Word2Vec(sentences, min_count=1)

print(model['sadness'])

model.save('./models/word2vec_model')

paths = ['./corpus/processed/eval_processed.txt',
	'./corpus/processed/train_processed.txt',
	'./corpus/processed/test_processed.txt']

# emotion_embeddings = {}
# for emotion, index in [('anger', 0), ('fear', 1), ('joy', 2), ('sadness', 3)]:
# 	embedding = np.zeros(100, dtype=int)
# 	embedding[int(index * 100 / 4)] = 1 # default size from Word2Vec is 100, we have 4 emotions
# 	emotion_embeddings[emotion] = embedding
# emotion_embeddings = {
# 	'anger': [1, 0, 0, 0],
# 	'fear': [0, 1, 0, 0],
# 	'joy': [0, 0, 1, 0],
# 	'sadness': [0, 0, 0, 1]
# }
emotion_embeddings = {
	'anger': 0,
	'fear': 1,
	'joy': 2,
	'sadness': 3
}


for path in paths:
	with open(path, 'r', encoding='utf-8') as input_file:
		with open(path.replace("processed", "embedded").replace('txt', 'pkl'), 'wb') as output_file:
			sentences_embedded = []
			emotions_embedded = []
			for line in input_file.readlines():
				[sentence, emotion] = line.split(';')
				sentence_embedded = [model[word] for word in sentence.split()]
				emotion_embedded = emotion_embeddings[emotion.strip()]
				sentences_embedded.append(sentence_embedded)
				emotions_embedded.append(emotion_embedded)
			pkl.dump((sentences_embedded, emotions_embedded), output_file, 2)