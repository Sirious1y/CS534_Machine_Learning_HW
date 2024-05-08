from collections import Counter
import numpy as np


def read_file(filename):
	emails = list()
	label = list()
	with open(filename, 'r') as f:
		for line in f:
			row = line.strip().split()
			label.append(int(row[0]))
			emails.append(row[1:])
	return emails, label


def build_vocab(train, test, minn):
	# build vocabulary
	word_count = Counter()
	vocabulary = {}
	for email in train:
		# words = email.split()
		unique_words = set(email)
		for word in unique_words:
			word_count[word] += 1

	idx = 0
	for word, count in word_count.items():
		if count >= minn:
			vocabulary[word] = idx
			idx += 1

	# transform the training set
	new_train = np.zeros((len(train), len(vocabulary)))
	for i, email in enumerate(train):
		# words = set(email.split())
		for word in email:
			if word in vocabulary:
				new_train[i, vocabulary[word]] = 1

	# transform the test set
	new_test = np.zeros((len(test), len(vocabulary)))
	for i, email in enumerate(test):
		# words = set(email.split())
		for word in email:
			if word in vocabulary:
				new_test[i, vocabulary[word]] = 1

	return new_train, new_test, vocabulary


class Perceptron():

	def __init__(self, epoch):
		# initialization
		self.epoch = epoch
		self.w = None
		return

	def get_weight(self):
		return self.w

	def sample_update(self, x, y):
		mistake = 0
		new_w = self.w
		if np.dot(self.w, x) >= 0:
			if y != 1:
				new_w = self.w + y * x
				mistake = 1
		else:
			if y != -1:
				new_w = self.w + y * x
				mistake = 1
		return new_w, mistake

	def train(self, trainx, trainy):
		n, d = trainx.shape
		self.w = np.zeros(d)
		stats = {}
		for i in range(self.epoch):
			total_mistakes = 0
			for (x, y) in zip(trainx, trainy):
				new_w, mistake = self.sample_update(x, y)
				self.w = new_w
				total_mistakes += mistake
			stats[i+1] = total_mistakes
			if total_mistakes == 0:
				break
		return stats

	def predict(self, newx):
		pred_y = []
		for i in range(newx.shape[0]):
			if np.dot(self.w, newx[i]) >= 0:
				pred_y.append(1)
			else:
				pred_y.append(-1)
		return np.array(pred_y)



class AvgPerceptron(Perceptron):

	def get_weight(self):
		return super().get_weight()

	def train(self, trainx, trainy):
		n, d = trainx.shape
		self.w = np.zeros(d)
		stats = {}
		weights = []
		for i in range(self.epoch):
			total_mistakes = 0
			for (x, y) in zip(trainx, trainy):
				new_w, mistake = self.sample_update(x, y)
				self.w = new_w
				total_mistakes += mistake
			stats[i + 1] = total_mistakes
			weights.append(self.w)
			if total_mistakes == 0:
				break
		weights = np.array(weights)
		self.w = np.mean(weights, axis=0)

		return stats
	
	def predict(self, newx):
		return super().predict(newx)



