from classifier import get_dataset, train_and_test_data
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB

data = get_dataset()
train_data, train_target, test_data, test_target = train_and_test_data(data)

nbc_mul = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=1.0)), ])
nbc_mul.fit(train_data, train_target)
predict = nbc_mul.predict(test_data)
count = 0
for left, right in zip(predict, test_target):
    if left == right:
        count += 1

print('Multinomial '+ str(count/len(test_target)))

nbc_ber = Pipeline([('vect', TfidfVectorizer()), ('clf', BernoulliNB(alpha=1.0)), ])
nbc_ber.fit(train_data, train_target)
predict = nbc_ber.predict(test_data)
count = 0
for left, right in zip(predict, test_target):
    if left == right:
        count += 1

print('BernoulliNB '+ str(count/len(test_target)))