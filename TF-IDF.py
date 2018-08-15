"""
term-frequency - inverse document frequency, also called TF-IDF, is a well know method to evaluate how
important is a word in a document.tf-idf is a very interesting way to convert the textual representation
of information into a vector-space model. VSM is an algebraic model representing textual information
as a vector. the components of this vector could representing the importance of a term(tf-idf) or even
the absence of presence of it in a document

Modeling the document into a vector space:

    Train Document Set:
        d1: The sky is blue
        d2: The sun is bright
    Test document set:
        d3: The sun in the sky is bright.
        d4: We can see the shining sun, the bright sun.

So the vocabulary of the words of the train document sets:
'blue', 'sun', 'bright', 'sky'

The term-frequency is nothing more than a measure of how many times the terms present in our vocabulary
are present in the documents d3 or d4, for example, the term-frequency of 'sun' in d4 is 2.

so the vector of d3 is [0, 1, 1, 1], the vector of d4 is [0, 2, 1, 0]

What tf-idf do is to scale down the frequent terms while scaling up the rare terms; a term that occurs
10 times more than another isn't 10 times more important than it.

Normalization
    the unit vector of [0, 2, 1, 0] is [0,2,1,0]/sqrt(2**2+1**2) = [0.0, 0.89442719, 0.4472136, 0.0]

Inverse document frequency weight
    Document space can be defined then as D={d1, d2, ... , dn}, in our case, D(train)={d1, d2},
    D(test)={d3, d4}, the cardinality of D(train) is |D(train)| = 2, |{d: t belongs d}| is the number of
    documents where the term t appears

    the feature matrix D is :
        0, 1, 1, 1,
        0, 2, 1, 0
    |D| is 2

    idf is the defined:
        idf(t) = log[|D| / (1 + |{d: t belongs d}|)]
    tf is defined:
        tf(t,d), which is actually the term count of term t in the document d.
    the formula for tf-idf is then:
        tf-idf = tf(t, d) * idf(t)

    for the test document set:
        idf(t1) = log(2/1) = 0.69314718
        idf(t2) = log(2/3) = -0.40546511
        idf(t3) = log(2/3) = -0.40546511
        idf(t4) = log(2/2) = 0

    M(tf-idf) = M(train) * M(idf)=
    0 1 1 1     *   0.69314718 0          0           0
    0 2 1 0         0         -0.40546511 0           0
                    0         0           -0.40546511 0
                    0         0           0           0

    =
    0 -0.40546511 -0.40546511 0
    0 -0.81093022 -0.40546511 0
    
    normalized M(tf-idf) is
    0 -0.70710678 -0.70710677 0
    0 -0.89442719 -0.4472136  0
"""

from sklearn.feature_extraction.text import CountVectorizer


def vector_text():
    traing_set = ('The sky is blue', 'The sun is bright')
    test_set = ('The sun in the sky is bright',
                'We can see the shining sun, the bright sun')
    vectorizer = CountVectorizer()
    print(vectorizer)
    vectorizer.fit_transform(traing_set)
    print(vectorizer.vocabulary)

if __name__ == '__main__':
    vector_text()