# -*- coding: UTF-8 -*-
import logging
import os
import _pickle as pkl
import numpy as np
import pprint

from gensim.models import word2vec


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

emotion_set = set()

class Sentences(object):
	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		for fname in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, fname)):
				sentence, emotion = line.split(';')
				emotion_set.add(emotion.strip())
				yield sentence.split()


sentences = Sentences('./processed')
model = word2vec.Word2Vec(sentences, min_count=1)

print(model['sadness'])
print(emotion_set)

model.save('./embed_model/word2vec_model')

paths = ['./processed/train.txt',
	'./processed/eval.txt',
	'./processed/test.txt']

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

emotion_embeddings = {e: i for i, e in enumerate(emotion_set)}
with open('./embed_model/emotion_embeddings.pkl', 'wb') as output_emo_embed:
	pkl.dump(emotion_embeddings, output_emo_embed, 2)
	print(emotion_embeddings)


for path in paths:
	with open(path, 'r', encoding='utf-8') as input_file:
		with open(path.replace("processed", "embedded").replace('txt', 'pkl'), 'wb') as output_file:
			sentences_embedded = []
			emotions_embedded = []
			for line in input_file.readlines():
				[sentence, emotion] = line.split(';')
				if len(sentence) == 0:
					continue
				sentence_embedded = [model[word] for word in sentence.split()]
				emotion_embedded = emotion_embeddings[emotion.strip()]
				sentences_embedded.append(sentence_embedded)
				emotions_embedded.append(emotion_embedded)
			pkl.dump((sentences_embedded, emotions_embedded), output_file, 2)