import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD as legacy_SGD
import random
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ["?", "!"]
data_file = open("intents.json").read()
intents = json.loads(data_file)


# Load intents from intents.json
with open("intents.json") as file:
    data = json.load(file)

# Preprocess the data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize each word in the pattern
        w = word_tokenize(pattern)
        words.extend(w)
        # Add the intent to the classes list
        classes.append(intent["tag"])

# Perform stemming and remove duplicates from the words list
words = list(set(words))

# Save words and classes to files
with open("words.pkl", "wb") as f:
    pickle.dump(words, f)

with open("classes.pkl", "wb") as f:
    pickle.dump(classes, f)

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent["tag"]))

        # add to our classes list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))

# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)

# Find the maximum length of the bags
max_len = 0

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in an attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in the current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Update the maximum length
    max_len = max(len(bag), max_len)

    # output is a '0' for each tag and '1' for the current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Pad the training data to have the same length
training = [
    (bag + [0] * (max_len - len(bag)), output_row) for bag, output_row in training
]

# shuffle our features and turn into np.array
random.shuffle(training)

# Separate the features (X) and labels (Y)
X = np.array([bag for bag, _ in training])
Y = np.array([output_row for _, output_row in training])

print("Training data created")
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to the number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(max_len,), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation="softmax"))

sgd = legacy_SGD(lr=0.01, momentum=0.9, nesterov=True)

# Compile model
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# fitting and saving the model
hist = model.fit(X, Y, epochs=200, batch_size=5, verbose=1)
model.save("bot_model.h5", hist)
