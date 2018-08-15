"""
A word embedding is a learning representation for text where words that have the same meaning have a similar
representation.

Word embeddings are the texts converted into numbers.

Word embedding are in fact a class of techniques where individual words are represented  real-valued
vectors in a predefined vector space. Each word is mapped to one vector and the vector values  are
learned in a way that resembles a neural network, and hence the technique if often lumped into the
field of deep learning.

Key to the approach is the idea of using a dense distribute representation for each word. Each word is
represented by a real-valued vector, often tens or hundreds of dimensions. Each word is associated with
a point in a vector space. The number of features is much smaller than the size of the vocabulary.

The distributed representation is learning based on the usage of words. This allows words that are
used om similar ways to result in having similar representation.

Word2Vec

    Word2Vec is a statistical method for efficiently learning a standalone word embedding from a text corpus

Two different learning models were introduced that can be used as part of the word2vec approach to learn
the word of embedding:
    continuous Bag-of-Words, or CBOW model.
    continuous Skip-Gram Model.

The CBOW model learns the embedding by predicting the current word based on its context.
The continuous skip-gram model learns by predicting the surrounding words given a current word.

Both models are focused on learning about words given their local usage context, where context is defined
by a window of neighboring words.

GloVe
    An extension to the word2Vec method for efficiently learning word vectors.


The idea behind all of the word embedding is to capture with them as much of the
semantical/morphological/context/hierarchical information as possible.
"""

"""
The skip-gram word2vec model
    the skip-gram model is trained to predict the surrounding words given the current word.
    consider the following example sentence:
        I love green eggs and ham
    Assuming a window size of three, this sentence can be broken down into the following sets of 
    (context, word) pairs:
        ([I, green], love)
        ([love, eggs], green)
        ([green, and], eggs)
        ...
    we can convert the preceding dataset to one of (input, output) pairs. That is , given an input
    word, we expect the skip-gram model to predict the output word:
    (love, I), (love, green), (green, love), (green, eggs), (eggs, green), (eggs, and), ...
    we can also generate additional negative samples:
    (love, sam), (green, thing)
    
    finally, we generate positive and negative examples for our classifier:
    ((love, I), 1), ((love, green), 1), ..., ((love, sam), 0), ((green, thing), 1) 
"""

from keras.preprocessing.text import *


def one_hot_encode():
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    corpus = {
        'Text of first document.',
        'Text of the second document made longer.',
        'Number three.',
        'This is number four.'
    }
    X = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(X.toarray())
    a = vectorizer.transform(['A new document']).toarray()
    print(a)

def skip_gram_word_2_vec():
    from keras.layers import Merge
    from keras.layers.core import Dense, Reshape
    from keras.layers.embeddings import Embedding
    from keras.models import Sequential
    from keras.preprocessing.sequence import skipgrams

    vocab_size = 500
    embed_size = 300

    word_model = Sequential()
    word_model.add(Embedding(vocab_size, embed_size,
                             embeddings_initializer='glorot_uniform',
                             input_length=1))
    word_model.add(Reshape((embed_size, )))

    context_model = Sequential()
    context_model.add(Embedding(vocab_size, embed_size,
                             embeddings_initializer='glorot_uniform',
                             input_length=1))
    context_model.add(Reshape((embed_size,)))

    model = Sequential()
    model.add(Merge([word_model, context_model], mode='dot'))
    model.add(Dense(1, init='glorot_uniform', activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    text = 'I love green eggs and ham'
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])

    word2id = tokenizer.word_index
    print(word2id)  # {'i': 1, 'love': 2, 'green': 3, 'eggs': 4, 'and': 5, 'ham': 6}
    id2word = {v: k for k, v in word2id.items()}
    wids = [word2id[w] for w in text_to_word_sequence(text)]
    print(wids)
    pairs, labels = skipgrams(wids, len(word2id))
    print(pairs, labels)


if __name__ == '__main__':
    skip_gram_word_2_vec()