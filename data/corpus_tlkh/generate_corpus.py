# -*- coding: UTF-8 -*-
import re
import os

path_corpus_original = './text_emotion_original.txt'

paths = ['./processed/test.txt',
	'./processed/eval.txt',
	'./processed/train.txt']


def emotion_converg(emotion):
	if emotion == "happiness" or emotion == "love" or emotion == "surprise" or emotion == "fun" or emotion == "enthusiasm":
		return "happy"
	if emotion == "empty" or emotion == "relief" or emotion == "neutral":
		return "neutral"
	if emotion == "worry" or emotion == "sadness":
		return "sad"
	if emotion == "hate" or emotion == "boredom":
		return "hate"
	if emotion == "anger":
		return "anger"


with open(path_corpus_original, 'r', encoding='utf-8') as corpus_original:
	count = 0
	path_tmp = './processed/temp.txt'
	with open(path_tmp, 'w', encoding='utf-8') as corpus_tmp:
		for line in corpus_original.readlines():
			count += 1
			# first line are the names of the columns, useless
			if count == 1:
				print(line)
				continue
			# print(line.split(',"'))
			sentence_emotion = line.split(',"')
			# delete the '\n' in the sentence
			sentence = re.sub(r'\\n', ' ', sentence_emotion[3].lower())
			# delete '@<name>', '&amp;'..., éèµ..., "' "or" '", --* not in f**k or sh*t-->cant support *
			sentence = re.sub(r'@\w+\b|&\w+;|\n|[^a-zA-Z1-9\'* ]|(?<!\w)\'|\'(?!\w)|(?<!\w)\*|\*(?!\w)', '', sentence)
			# delete the multiple continuous spaces
			sentence = re.sub(r' {2,}', ' ', sentence).strip()
			emotion = sentence_emotion[1]
			emotion = re.sub(r'\"', '', emotion)
			emotion = emotion_converg(emotion)
			corpus_tmp.write(sentence + ';' + emotion + '\n')
		count -= 1
		print(count)
	with open(path_tmp, 'r', encoding='utf-8') as corpus_tmp:
		end_of_train = int(count * 0.7)
		end_of_eval = int(count * 0.85)
		print(str(end_of_train) + ', ' + str(end_of_eval))
		for path in paths:
			with open(path, 'w', encoding='utf-8') as output_file:
				while True:
					count -= 1
					line = corpus_tmp.readline()
					output_file.write(line)
					if count <= 0 or (count - 1) == end_of_eval or (count - 1) == end_of_train:
						break
	os.remove(path_tmp)