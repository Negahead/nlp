"""
Assume you have a little robot that is trying to estimate the posterior  probability that you are happy or sad,
given that the robot was observed whether you are watching Game of Thrones(w), sleeping(s), crying(c) or
facebooking(f)

Let the unknown state be X=H if you are happy and X=S if you are sad
Let Y denote the observation, which can be w,s,c or f

we want to answer queries, such as:

    p(X=H|Y=f) ?
    p(X=S|Y=c) ?


For example :
    the distribution of X is 0.2 prob of sad and 0.8 prob of happy
    p(X=H) = 0.8    p(X=S) = 0.2

    p(w|S) = 0.1, p(s|S) = 0.3, p(c|S) = 0.5, p(f|S) = 0.1
    p(w|H) = 0.4, p(w|H) = 0.4, p(c|H) = 0.2, p(f|H) = 0.0

    So

    the probability of you are happy when you are watching Game of Thrones is
    p(X=H|Y=w) = [ p(Y=w|X=H) * p(X=H) ] / [p(Y=w|X=H)*p(X=H) + p(Y=W|X=S)*p(X=s)]
"""

from collections import deque

d={'一': 1,
 '二': 2,
 '三': 3,
 '四': 4,
 '五': 5,
 '六': 6,
 '七': 7,
 '八': 8,
 '九': 9,
 }

mul = {
'十': 10,
   '千':1000,
   '百': 100,
   '万': 10000,
'亿':100000000}

l = deque()

for char in '九千六百四十五':
    if char in d:
        l.append(d[char])
    if char in mul:
        if char == '万' and sum(l) < 10000:
            l = [sum(l)]
        if char == '万' and sum(l) > 10000:
            s = sum(l)
            l = [s - s % 10000, s % 10000]
        rightmost = l.pop()
        l.append(rightmost * mul[char])
        print(l)
        l = [sum(l)]

print(sum(l))
