"""

The idea behind RNN is the make use of the sequential information. In a traditional neural network we assume
that all inputs are independent of each other. But for many tasks that is a very bad idea. If you want to
predict the next word in a sentence you better know which words came task for every element of a sequence.
Another way to think about RNN is that they have a memory which captures information about what has been
calculated so far.

If the sequence we care about is a sentence of 5 words, the network would be unrolled into a 5-layer
neural network, one layer for each word.

Unlike a traditional deep neural network, which uses different parameters at each layer, a RNN shares the
same parameters across all steps.This reflects the fact that we are performing the same task at each step.
just with different inputs, This greatly reduces the total number of parameters we need to learn.

The most commonly used type of RNNs are LSTMs, which are much better at capturing long-term dependencies
that vanilla RNNs are, LStMs are essentially the same thing as RNNs, they just have a different way of
computing the hidden state.

Given a sequence of words we want to predict the probability of each word given the previous words.
Language Models allow us to measure how likely a sentence is.
"""