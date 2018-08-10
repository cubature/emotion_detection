# -*- coding: UTF-8 -*-
import re

paths = ['./corpus/original/eval_original.txt',
	'./corpus/original/train_original.txt',
	'./corpus/original/test_original.txt']

for path in paths:
	with open(path, 'r', encoding='utf-8') as input_file:
		with open(path.replace("original", "processed"), 'w', encoding='utf-8') as output_file:
			for line in input_file.readlines():
				sentence_emotion = line.split('\t', 2)
				# delete the '\n' in the sentence
				sentence = re.sub(r'\\n', ' ', sentence_emotion[1].lower())
				# delete '@<name>', '&amp;'..., éèµ..., "' "or" '", --* not in f**k or sh*t-->cant support *
				sentence = re.sub(r'@\w+\b|&\w+;|\n|[^a-zA-Z1-9\'* ]|(?<!\w)\'|\'(?!\w)|(?<!\w)\*|\*(?!\w)', '', sentence)
				# delete the multiple continuous spaces
				sentence = re.sub(r' {2,}', ' ', sentence).strip()
				emotion = sentence_emotion[2].split('\t')[0]
				output_file.write(sentence + ';' + emotion + '\n')