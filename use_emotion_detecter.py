# -*- coding: UTF-8 -*- 
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from rnn_model import RNN_Model
import data_helper
import pprint

from gensim.models import word2vec

batch_size = 64

sentence = "i'm scared"

w2v_model = word2vec.Word2Vec.load("./data/models/word2vec_model_1")
max_len = 40

sentence_embedded = []
words = filter(lambda x: x in w2v_model.wv.vocab, sentence.split())
for word in words:
	sentence_embedded.append(w2v_model[word])
sentence_embedded = [sentence_embedded]
# sentence_embedded = [[w2v_model[word] for word in sentence.split()]]
emotion_useless = [4]
x1 = np.zeros([batch_size, max_len, len(sentence_embedded[0][0])])
y1 = np.zeros([batch_size])
mask = np.zeros([max_len, batch_size])

def padding_and_generate_mask(x, y, new_x, new_y, new_mask_x):

	for i, (x, y) in enumerate(zip(x, y)):
		# whether to remove sentences with length larger than max_len
		if len(x) <= max_len:
			new_x[i, 0:len(x)] = x
			new_mask_x[0:len(x), i] = 1
			new_y[i] = y
		else:
			new_x[i] = (x[0:max_len])
			new_mask_x[:, i] = 1
			new_y[i] = y
	new_set = (new_x, new_y, new_mask_x)
	del new_x, new_y
	return new_set

data = padding_and_generate_mask(sentence_embedded, emotion_useless, x1, y1, mask)

with tf.Session() as sess:
	# Restore variables from disk.
	saver = tf.train.import_meta_graph("./runs/checkpoints/model-2857.meta")
	saver.restore(sess, tf.train.latest_checkpoint("./runs/checkpoints/"))

	print("Model restored.")

	graph = sess.graph
	# # trainable variables in model
	# tvs = [v for v in tf.trainable_variables()]
	# for v in tvs:
	# 	print(v.name)
	# 	print(sess.run(v))

	# # tensors, operations in model
	# gv = [v for v in tf.global_variables()]
	# for v in gv:
	# 	print(v.name)

	# # tensors in model
	# n = [n for n in tf.get_default_graph().as_graph_def().node]
	# for t in n:
	# 	print(t.name)

	# # all operations in model
	# with open("./tensors_in_model", "w") as f:
		# # ops = [o for o in sess.graph.get_operations()] # 1
		# # ops = [o for o in tf.get_default_graph().get_operations()] # 2
		# ops = [o for o in graph.get_operations()]
		# for o in ops:
		# 	f.write(o.name + "\n")

	model_x = graph.get_tensor_by_name('model/Placeholder:0')
	model_y = graph.get_tensor_by_name('model/Placeholder_1:0')
	model_mask = graph.get_tensor_by_name('model/Placeholder_2:0')
	model_output = graph.get_tensor_by_name('model/accuracy/ArgMax:0')
	model_initial_state_fw = graph.get_tensor_by_name('model/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7:0')
	model_initial_state_bw = graph.get_tensor_by_name('model/MultiRNNCellZeroState_1/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7:0')

	# for step, (x, y, mask_x) in enumerate(data_helper.batch_iter(data, batch_size=batch_size)):

	# fetches = [model.correct_num, model.final_state]
	fetches = [model_output]
	feed_dict = {}
	feed_dict[model_x], feed_dict[model_y], feed_dict[model_mask] = data
	# for i, (c, h) in enumerate(model_initial_state_fw):
	# 	feed_dict[c] = state_fw[i].c
	# 	feed_dict[h] = state_fw[i].h
	# for i, (c, h) in enumerate(model_initial_state_bw):
	# 	feed_dict[c] = state_bw[i].c
	# 	feed_dict[h] = state_bw[i].h

	output = sess.run(fetches, feed_dict)
	print("The emotion of sentence \"" + sentence + "\" is:")
	print({
			0: 'anger',
			1: 'fear',
			2: 'joy',
			3: 'sadness'
		}[output[0][0]])

	# # Check the values of the variables
	# print("v1 : %s" % v1.eval())
	# print("v2 : %s" % v2.eval())