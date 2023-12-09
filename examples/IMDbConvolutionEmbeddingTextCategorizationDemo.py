import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from PyCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D, MultiClassTsetlinMachine

maxlen = 500

epochs = 100

batches = 100

hypervector_size = 1024
bits = 5

clauses = 10000
T = 8000
s = 10.0

NUM_WORDS=10000
INDEX_FROM=2

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, maxlen=maxlen, index_from=INDEX_FROM)

train_x, train_y = train
test_x, test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}

# Read from file instead, otherwise the same

print("Retrieving embeddings...")

encoding = {}
f = open("/data/near-lossless-binarization/binary_vectors_1024.vec", "r")
line = f.readline()
line = f.readline().strip()
while line:
	entries = line.split(" ")
	if entries[0] in word_to_id:
		values = np.unpackbits(np.fromstring(" ".join(entries[1:]), dtype=np.int64, sep=' ').view(np.uint8))
		encoding[word_to_id[entries[0]]] = np.unpackbits(np.fromstring(" ".join(entries[1:]), dtype=np.int64, sep=' ').view(np.uint8)).nonzero()
	line = f.readline().strip()
f.close()
	
print("Producing bit representation...")

print(train_y.shape[0])
X_train = np.empty((train_y.shape[0], maxlen*hypervector_size), dtype=np.uint32)
for e in range(train_y.shape[0]):
	position = 0
	for word_id in train_x[e]:
		if word_id in encoding:
			for bit_index in encoding[word_id]:
				X_train[e, position*hypervector_size + bit_index] = 1
			position += 1

X_train = X_train.reshape((train_y.shape[0], maxlen, hypervector_size))

Y_train = train_y.astype(np.uint32)

print(test_y.shape[0])
X_test = np.empty((test_y.shape[0], maxlen*hypervector_size), dtype=np.uint32)
for e in range(test_y.shape[0]):
	position = 0
	for word_id in test_x[e]:
		if word_id in encoding:
			for bit_index in encoding[word_id]:
				X_test[e, position*hypervector_size + bit_index] = 1
			position += 1

X_test = X_test.reshape((test_y.shape[0], maxlen, hypervector_size))

Y_test = test_y.astype(np.uint32)

batch_size_train = train_y.shape[0] // batches
batch_size_test = test_y.shape[0] // batches

tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (1, 1))
for i in range(epochs):
    start_training = time()
    for batch in range(batches):
    	print("Training batch", batch+1)
    	tm.fit(X_train[batch*batch_size_train:(batch+1)*batch_size_train,:], Y_train[batch*batch_size_train:(batch+1)*batch_size_train], epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    y_predict_test = np.zeros(0, dtype=np.uint32)
    for batch in range(batches):
    	print("Testing batch", batch+1)
    	y_predict_test = np.concatenate((y_predict_test, tm.predict(X_test[batch*batch_size_test:(batch+1)*batch_size_test,:])))
    stop_testing = time()

    result_test = 100*(y_predict_test == Y_test).mean()

    y_predict_train = np.zeros(0, dtype=np.uint32)
    for batch in range(batches):
    	y_predict_train = np.concatenate((y_predict_train, tm.predict(X_train[batch*batch_size_train:(batch+1)*batch_size_train,:])))

    result_train = 100*(tm.predict(X_train) == Y_train).mean()

    print("#%d Accuracy Test: %.2f%% Accuracy Train: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
