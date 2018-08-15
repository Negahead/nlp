"""
The bag-of-words model is a way of representing text data when modeling text with machine learning algorithm.
The idea is to analysis and classify different 'bags of words'. And by matching the different categories,
we identify which 'bag' a certain block of text comes from.

machine learning algorithm cannot work with raw text directly; the text must be converted into numbers,
specifically, vectors of numbers.

In language processing, the vector x are derived from textual data, in order to reflect various linguistic
properties of the text.this is called feature extraction or feature encoding.A popular and simple method
of feature extraction with text data is called the bag-of-words model of text.

A bag-of-words model, is a way of extracting features from text for use in modeling.It is a representation
of text that describes the occurrence of words within a document.It is called a 'bag' of words, because
any information about the order or structure of words in the document is discarded.The model is only
concerned with whether known words occur in the document, not where in the document.

    It was the best of times,
    it was the worst of times,
    it was the age of wisdom,
    it was the age of foolishness
For this example, let's treat each line as a separate 'document' and the 4 lines as out entire corpus of documents.

The unique words here are:
    it was the best of times worst age wisdom foolishness
    1  1   1   1    1  1     1     1   1      1

That is a vocabulary of 10 words from a corpus containing 24 words.

The objective is to turn each document of free text into a vector that we can use as input or output
for a machine learning model, Because we know the vocabulary has 10 words, we can use a fixed-length
document representation of 10, with one position in the vector to score each word.

The simplest scoring method is to mark the presence of words as a boolean value, 0 for absent, 1 for
present:

'it was the worst of times' = [1,1,1,0,1,1,1,0,0,0]

But, for a large corpus, that the length of the vector might be thousands or millions of position. So A more
sophisticated approach is to create a vocabulary of grouped words, This both changes scope of the vocabulary
and allows the bag-of-words to capture a little bit more meaning of the document. In this approach, each
word or token is called a 'gram'. Creating a vocabulary of two-word pairs is in turn, called a bigram model,
Again only the bigrams that appear in the corpus are modeled, not all possible bigrams.

for example. the bigrams in the first line of text in the previous section, 'it was the best of times' are
as follows:
'it was',
'was the',
'the best',
'best of',
'of times'.

Some words tend to appear more often that other,Words such as 'is', 'the', 'a', are very common words in the
english language, If we take consider their raw frequency, we might not be able to effectively differentiate
between different classes of documents.

One approach is to rescale the frequency of words by how often they appear in all document, so that the scores
for frequent words like 'the' that are also frequent across all documents are penalized. This approach to
scoring is called Term Frequency - Inverse Document Frequency
    Term Frequency, is a scoring of the frequency of the word in the current document.
    Inverse Document Frequency, is a scoring of how rare the word is across the documents.

One common way to determine the value of the ter frequency is to basically just take the raw frequency of
term divided by the maximum frequency of any term in the document.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


def transform_words_into_feature_vectors():
    count = CountVectorizer()
    docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'
    ])
    bag = count.fit_transform(docs)
    print(count.vocabulary_)
    # this is the raw term frequencies tf(t, d), also called the 1-gram or unigram model.
    # each item or token in the vocabulary represents a single word.
    # n-grams of size 3 and 4 yield good performances in anti-spam filtering of email messages
    # 1-gram: "the", "sun", "is", "shining"
    # 2-grams : "the sun", "sun is", "is shining"
    print(bag.toarray())
    # the equations for the inverse document frequency implemented in scikit-learn is computed as
    # follows: idf(td,) = log[ (1+Nd) / (1 + df(d, t)) ]
    # similarly, the tf-idf computed in scikit-learn deviates slightly from the default equation we defined
    # earlier : tf-idf(t,d) = tf(t,d) * (idf(t,d) + 1)
    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    np.set_printoptions(precision=2)
    print(tfidf.fit_transform(bag).toarray())


if __name__ == '__main__':
    transform_words_into_feature_vectors()
