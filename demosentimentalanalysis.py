import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
import csv
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import collections
from sklearn.svm import LinearSVC, SVC
import random

def create_word_features(words):
    my_dict = dict([(word, True) for word in words])
    return my_dict
print('---------------------------------------------------------------------------')
print('WELCOME TO  SENTIMENTAL ANALYSIS OF ONLINE IMDB MOVIE REVIEWS ')
print('---------------------------------------------------------------------------')
print('                               DATASET                               ')

neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append((create_word_features(words), "negative"))

#print(pos_reviews[0])
print('length of negative reviews')
print(len(neg_reviews))
pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append((create_word_features(words), "positive"))

#print(pos_reviews[0])
print('length of positive reviews')
print(len(pos_reviews))

train_set = neg_reviews[:750] + pos_reviews[:750]
test_set =  neg_reviews[750:] + pos_reviews[750:]
print('length of train set')
print(len(train_set))
print('length of test set')
print( len(test_set))


print('                                                             ')
classifierName = 'Naive Bayes'
classifiernb= NaiveBayesClassifier.train(train_set)
print('                                                             ')
print('---------------------------------------------------------------------------')
print('SAMPLE FROM IMBD REVIEW OF BIGIL')
review = '''
After ten years of epic waiting, we just got a chance to clarify that The Russo Brothers were an awful choice for Marvel Cinametic Universe. Infinity War, including over 30 main characters, starts with a quick plot without losing time to introduce our heroes. The movie goes on with funny bits popping out of nowhere during non-stop action sequences and sends us back home with a terrible ending. Suprisingly, it is not only bad because of lack of drama, but also the glorious fight between Avengers and Thanos brings out nothing original. The movie may serve as a great entertainment for early teens and comic-book fans, but surely not beyond that'''

review1='''Such an epic movie!!I went for 6.am show I just mesmerized everything was good and entertaining starting from the title block and end credits the film didn't lost its energy and screen play....This film deals much about women and their problems it said clearly by atlee with thalapathu's screen presence...overall it's going to be a blockbuster..wait and watch avoid negativity and negative comments
'''
words = word_tokenize(review1)
print(words)
words = create_word_features(words)
print(words)
print('---------------------------------------------------------------------------')
print('                               PREDICTION                               ')
print('NAIVE BAYES')
print (classifiernb.classify(words))
