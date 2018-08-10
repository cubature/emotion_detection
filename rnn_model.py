# -*- coding: UTF-8 -*- 
import tensorflow as tf
import numpy as np
import pprint

class RNN_Model(object):

	def __init__(self, config, is_training=True):
		self.keep_prob = config.keep_prob
		# self.batch_size = tf.Variable(0, dtype=tf.int32, trainable=False)
		self.batch_size = config.batch_size

		num_step = config.num_step
		class_num = config.class_num # normally 4
		embed_dim = config.embed_dim
		# self._input_data = tf.placeholder(tf.float32, shape=[None, num_step]) # n_input = embed_dim ?
		self._input_data = tf.placeholder(tf.float32, shape=[None, num_step, embed_dim])
		# self._targets = tf.placeholder(tf.int64, shape=[None, class_num])
		self._targets = tf.placeholder(tf.int64, shape=[None])
		self.mask_x = tf.placeholder(tf.float32, [num_step, None])

		hidden_neural_size = config.hidden_neural_size
		vocabulary_size = config.vocabulary_size
		hidden_layer_num = config.hidden_layer_num
		# self.new_batch_size = tf.placeholder(tf.int32, shape=[], name="new_batch_size")
		# self._batch_size_update = tf.assign(self.batch_size, self.new_batch_size)

		# build LSTM network
		# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_neural_size, forget_bias=0.0, state_is_tuple=True)
		# if self.keep_prob < 1:
		# 	lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
		# 		lstm_cell, 
		# 		output_keep_prob=self.keep_prob
		# 	)

		# cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * hidden_layer_num, state_is_tuple=True)

		# # self._initial_state = cell.zero_state(self.batch_size.read_value(), dtype=tf.float32)
		# self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

		# BiLSTM
		lstm_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_neural_size, forget_bias=1.0, state_is_tuple=True)
		lstm_bw = tf.nn.rnn_cell.BasicLSTMCell(hidden_neural_size, forget_bias=1.0, state_is_tuple=True)
		if self.keep_prob < 1:
			lstm_fw = tf.nn.rnn_cell.DropoutWrapper(
				lstm_fw,
				output_keep_prob=self.keep_prob
			)
			lstm_bw = tf.nn.rnn_cell.DropoutWrapper(
				lstm_bw,
				output_keep_prob=self.keep_prob
			)
		cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_fw] * hidden_layer_num, state_is_tuple=True)
		cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_bw] * hidden_layer_num, state_is_tuple=True)
		self.initial_state_fw = cell_fw.zero_state(self.batch_size, dtype=tf.float32)
		self.initial_state_bw = cell_bw.zero_state(self.batch_size, dtype=tf.float32)

		# embedding layer
		# with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
		# 	embedding = tf.get_variable("embedding", [vocabulary_size, embed_dim], dtype=tf.float32)
		# 	inputs = tf.nn.embedding_lookup(embedding, self._input_data)
		inputs = self._input_data

		if self.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, self.keep_prob)

		# out_put = []
		# state = self._initial_state
		# out_put_fw = []
		# out_put_bw = []
		state_fw = self.initial_state_fw
		state_bw = self.initial_state_bw
		with tf.variable_scope("LSTM_layer"):
			# for time_step in range(num_step):
			# 	if time_step > 0:
			# 		tf.get_variable_scope().reuse_variables()
			# 	(cell_output, state) = cell(inputs[:, time_step, :], state)
			# 	out_put.append(cell_output)
			# self._final_state = state

			# with tf.variable_scope('fw'):
			# 	for time_step in range(num_step):
			# 		if time_step > 0:
			# 			tf.get_variable_scope().reuse_variables()
			# 		(cell_output_fw, state_fw) = cell_fw(inputs[:, time_step, :], state_fw)
			# 		out_put_fw.append(cell_output_fw)
			# 	self.final_state_fw = state_fw

			# with tf.variable_scope('bw'):
			# 	inputs = tf.reverse(inputs, [1])
			# 	for time_step in range(num_step):
			# 		if time_step > 0:
			# 			tf.get_variable_scope().reuse_variables()
			# 		(cell_output_bw, state_bw) = cell_bw(inputs[:, time_step, :], state_bw)
			# 		out_put_bw.append(cell_output_bw)
			# 	self.final_state_bw = state_bw

			# out_put_bw = tf.reverse(out_put_bw, [0])
			# out_put = tf.concat([out_put_fw, out_put_bw], 2)

			# for time_step in range(num_step):
			# 	if time_step > 0:
			# 		tf.get_variable_scope().reuse_variables()
			((out_put_fw, out_put_bw), (state_fw, state_bw)) = tf.nn.bidirectional_dynamic_rnn(
				cell_fw, 
				cell_bw, 
				inputs, 
				initial_state_fw=state_fw, 
				initial_state_bw=state_bw
			)
			out_put = tf.concat([out_put_fw, out_put_bw], 2)
			out_put = tf.transpose(out_put, [1, 0, 2])
			self.final_state_fw = state_fw
			self.final_state_bw = state_bw

		out_put = out_put * self.mask_x[:, :, None]

		with tf.name_scope("mean_pooling_layer"):
			out_put = tf.reduce_sum(out_put, 0) / (tf.reduce_sum(self.mask_x, 0)[:, None])

		with tf.name_scope("softmax_layer_and_output"):
			# softmax_w = tf.get_variable("softmax_w", [hidden_neural_size, class_num], dtype=tf.float32) # weight
			softmax_w = tf.get_variable("softmax_w", [hidden_neural_size * 2, class_num], dtype=tf.float32)
			softmax_b = tf.get_variable("softmax_b", [class_num], dtype=tf.float32) # bias
			self.logits = tf.matmul(out_put, softmax_w) + softmax_b

		with tf.name_scope("loss"):
			# self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits + 1e-10, labels=self._targets) # ont-hot represents class
			self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits + 1e-10, labels=self._targets) # int represents class
			self._cost = tf.reduce_mean(self.loss)

		with tf.name_scope("accuracy"):
			self.prediction = tf.argmax(self.logits, 1)
			# correct_prediction = tf.equal(self.prediction, tf.argmax(self._targets, 1))
			correct_prediction = tf.equal(self.prediction, self._targets)
			self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

		# add summary
		loss_summary = tf.summary.scalar("loss", self._cost)
		accuracy_summary = tf.summary.scalar("accuracy_summary", self.accuracy)

		if not is_training:
			return

		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self._lr = tf.Variable(0.0, trainable=False)

		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm)

		# keep track of gradient values and sparsity (optional)
		grad_summaries = []
		for g, v in zip(grads, tvars):
			if g is not None:
				grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
				sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
				grad_summaries.append(grad_hist_summary)
				grad_summaries.append(sparsity_summary)
		self.grad_summaries_merged = tf.summary.merge(grad_summaries)

		self.summary = tf.summary.merge([loss_summary, accuracy_summary, self.grad_summaries_merged])


		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		# optimizer.apply_gradients(zip(grad, tvars))
		self._train_op = optimizer.apply_gradients(zip(grads, tvars))

		self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self.new_lr)

	def assign_new_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self.new_lr: lr_value})

	def assign_new_batch_size(self, session, batch_size_value):
		session.run(self._batch_size_update, feed_dict={self.new_batch_size: batch_size_value})

	def assign_lr(self, session, lr_value):
		# 使用 session 来调用 lr_update 操作
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	@property
	def input_data(self):
		return self._input_data

	@property
	def targets(self):
		return self._targets

	# @property
	# def initial_state(self):
	# 	return self._initial_state

	@property
	def cost(self):
		return self._cost

	# @property
	# def final_state(self):
	# 	return self._final_state

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op