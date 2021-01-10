################################
# Alex Yang (ay2344)
# Artificial Intelligence
# Programming 3
################################

################################
# Part 4 - Tuning the Classifier
#
# Default Results (dev.txt) (k=1, c=1):
#   Precision:0.9508196721311475
#   Recall:0.8656716417910447
#   F-Score:0.9062499999999999
#   Accuracy:0.9784560143626571
#
# Improved Results (dev.txt) (k=2, c=0.5):
#   Precision:0.9516129032258065
#   Recall:0.8805970149253731
#   F-Score:0.9147286821705426
#   Accuracy:0.9802513464991023
#
# Improved Results (test.txt) (k=2, c=0.5):
#   Precision:0.9649122807017544
#   Recall:0.873015873015873
#   F-Score:0.9166666666666667
#   Accuracy:0.9820466786355476
################################


import sys
import string
import math


def extract_words(text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)

    return text.split()


class NbClassifier(object):

    def __init__(self, training_filename, stopword_file=None):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}
        self.exclude = []

        if stopword_file:
            stopwords = open(stopword_file, 'r')
            self.exclude = stopwords.read().split('\n')

        self.collect_attribute_types(training_filename, k=2)
        self.train(training_filename, c=0.5)

    def collect_attribute_types(self, training_filename, k=1):
        word_dict = {}  # key: word, value: frequency

        input_file = open(training_filename, 'r')
        lines = input_file.readlines()
        input_file.close()

        for line in lines:
            line = line.split('\t')
            words = extract_words(line[1])

            for word in words:
                if word in self.exclude:
                    continue

                if word in word_dict.keys():
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1

        self.attribute_types = set([word for word in word_dict.keys() if word_dict[word] >= k])

    def train(self, training_filename, c=1):
        input_file = open(training_filename, 'r')
        messages = input_file.readlines()

        word_labels = {}  # key: (word, label), value: count
        spam_count = 0
        ham_count = 0
        spam_words = 0
        ham_words = 0

        for message in messages:
            message = message.split('\t')
            label = message[0]
            words = extract_words(message[1])
            if label == 'spam':
                spam_count += 1
                spam_words += len(words)
            else:
                ham_count += 1
                ham_words += len(words)
            for word in words:
                if (word, label) in word_labels.keys():
                    word_labels[(word, label)] += 1
                else:
                    word_labels[(word, label)] = 1

        self.label_prior['spam'] = spam_count / len(messages)
        self.label_prior['ham'] = ham_count / len(messages)

        for word in self.attribute_types:
            if (word, 'spam') in word_labels.keys():
                self.word_given_label[(word, 'spam')] = \
                    (word_labels[(word, 'spam')] + c) / (spam_words + c * len(self.attribute_types))
            else:
                self.word_given_label[(word, 'spam')] = c / (spam_words + c * len(self.attribute_types))

            if (word, 'ham') in word_labels.keys():
                self.word_given_label[(word, 'ham')] = \
                    (word_labels[(word, 'ham')] + c) / (ham_words + c * len(self.attribute_types))
            else:
                self.word_given_label[(word, 'ham')] = c / (ham_words + c * len(self.attribute_types))

    def predict(self, text):
        words = extract_words(text)

        prob_ham = math.log(self.label_prior['ham'])
        prob_spam = math.log(self.label_prior['spam'])
        for word in words:
            if word in self.attribute_types:
                prob_ham += math.log(self.word_given_label[(word, 'ham')])
                prob_spam += math.log(self.word_given_label[(word, 'spam')])

        return {'ham': prob_ham,
                'spam': prob_spam}

    def evaluate(self, test_filename):
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        test_set_file = open(test_filename, 'r')
        lines = test_set_file.readlines()

        for line in lines:
            line = line.split('\t')
            label = line[0]

            probs = self.predict(line[1])
            prediction = max(probs, key=probs.get)

            if prediction == 'spam':
                if label == 'spam':
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if label == 'ham':
                    true_negatives += 1
                else:
                    false_negatives += 1

        total = true_positives + true_negatives + false_positives + false_negatives

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        fscore = (2 * precision * recall) / (precision + recall)
        accuracy = (true_negatives + true_positives) / total

        return precision, recall, fscore, accuracy


def print_result(result):
    print("Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(*result))


if __name__ == "__main__":

    classifier = NbClassifier(sys.argv[1])
    result = classifier.evaluate(sys.argv[2])
    print_result(result)
