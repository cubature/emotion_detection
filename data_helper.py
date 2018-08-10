# -*- coding: UTF-8 -*- 
import _pickle as pkl
import numpy as np
import logging
import pprint

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

paths = [
	'./data/corpus/embedded/train_embedded.pkl',
	'./data/corpus/embedded/eval_embedded.pkl',
	'./data/corpus/embedded/test_embedded.pkl'
	]


def load_data(max_len, sort_by_len=True):
	train_set_x, train_set_y = pkl.load(open(paths[0], 'rb'))
	valid_set_x, valid_set_y = pkl.load(open(paths[1], 'rb'))
	test_set_x, test_set_y = pkl.load(open(paths[2], 'rb'))


	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))


	if sort_by_len:
		sorted_index = len_argsort(test_set_x)
		test_set_x = [test_set_x[i] for i in sorted_index]
		test_set_y = [test_set_y[i] for i in sorted_index]

		sorted_index = len_argsort(valid_set_x)
		valid_set_x = [valid_set_x[i] for i in sorted_index]
		valid_set_y = [valid_set_y[i] for i in sorted_index]

		sorted_index = len_argsort(train_set_x)
		train_set_x = [train_set_x[i] for i in sorted_index]
		train_set_y = [train_set_y[i] for i in sorted_index]

	train_set=(train_set_x, train_set_y)
	valid_set=(valid_set_x, valid_set_y)
	test_set=(test_set_x, test_set_y)

	new_train_set_x=np.zeros([len(train_set[0]), max_len, len(train_set[0][0][0])])
	# new_train_set_y=np.zeros([len(train_set[0]), len(train_set[1][0])])
	new_train_set_y=np.zeros([len(train_set[0])])

	new_valid_set_x=np.zeros([len(valid_set[0]), max_len, len(valid_set[0][0][0])])
	# new_valid_set_y=np.zeros([len(valid_set[0]), len(valid_set[1][0])])
	new_valid_set_y=np.zeros([len(valid_set[0])])

	new_test_set_x=np.zeros([len(test_set[0]), max_len, len(test_set[0][0][0])])
	# new_test_set_y=np.zeros([len(test_set[0]), len(test_set[1][0])])
	new_test_set_y=np.zeros([len(test_set[0])])

	mask_train_x=np.zeros([max_len, len(train_set[0])])
	mask_test_x=np.zeros([max_len, len(test_set[0])])
	mask_valid_x=np.zeros([max_len, len(valid_set[0])])


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


	train_set=padding_and_generate_mask(train_set[0], train_set[1], new_train_set_x, new_train_set_y, mask_train_x)
	test_set=padding_and_generate_mask(test_set[0], test_set[1], new_test_set_x, new_test_set_y, mask_test_x)
	valid_set=padding_and_generate_mask(valid_set[0], valid_set[1], new_valid_set_x, new_valid_set_y, mask_valid_x)

	return train_set, valid_set, test_set


def batch_iter(data, batch_size):
	x, y, mask_x = data
	x = np.array(x)
	y = np.array(y)
	data_size = len(x)
	num_batches_per_epoch = int((data_size - 1) / batch_size)
	for batch_index in range(num_batches_per_epoch):
		start_index = batch_index * batch_size
		end_index = min((batch_index + 1) * batch_size, data_size)
		return_x = x[start_index:end_index]
		return_y = y[start_index:end_index]
		return_mask_x = mask_x[:, start_index:end_index]
		yield (return_x, return_y, return_mask_x)


if __name__ == '__main__':
	a, b, c = load_data(40)

	x, y, z = a

	print(x[1]) # 3613, 40, 100
	print(y[1]) # 3613
	print(z.transpose()[1]) # 40, 3613
	print(len(y)) # 3613