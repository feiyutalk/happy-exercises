# 利用朴素贝叶斯完成康奈尔大学影评文本分类

# 前言

在以下的3篇博客中，分别对朴素贝叶斯的原理以及sklearn中的朴素贝叶斯库进行了介绍，在这里则是结合具体了项目，将前面学习的朴素贝叶斯基本原理应用到文本分类中。如果你对朴素贝叶斯不太了解，请先花点时间看下面3篇博客:

- [Naive Bayes Classifier](http://neuclil.com/2017/10/06/Naive-Bayes-Classifier/)
- [Naive Bayes Classifier II](http://neuclil.com/2017/10/07/Naive-Bayes-Classifier-II/)
- [Naive Bayes Classifier III](http://neuclil.com/2017/10/07/Naive-Bayes-Classifier-III/)

然后结合这篇小项目，让你能更加深刻的理解朴素贝叶斯算法。

该项目参考自 

作者：YoghurtIce
链接：http://www.jianshu.com/p/b8e0ae7cfa39
來源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

# 1. 数据来源

本文所用的数据是康奈尔大学网站的2M影评数据集，[CSDN下载地址](http://download.csdn.net/download/lsldd/9346233)。里面共计有1400条影评，700条好评，700条差评，作者已经为我们分好类别。

当然，这些数据已经下载好了，放在input目录下。

# 2. 代码说明

## 2.1 抽取文档特征

docclass.py文件主要用于从文本中抽取特征，这里的做法是先用正则表达式提取出单词，然后用stopwords集合进行过滤，去掉停止次，最后得到的特征实际上就是文档中的单词。

项目中的stopwords文件存放的就是停止词。

```python
def getwords(doc):
    splitter = re.compile('\\W*')
    words = [s.lower() for s in splitter.split(doc) if len(s) > 2 and len(s) < 20]
    # 过滤掉单词中包含数字的单词
    words = [word for word in words if word.isalpha()]
    with open(r'./stopwords.txt') as f:
        stopwords = f.read()
    stopwords = stopwords.split('\n')
    stopwords = set(stopwords)
    # 过滤掉一些经常出现的单词,例如 a,an,we,the
    words = [word for word in words if word not in stopwords]
    return set(words)
```

## 2.2 手写朴素贝叶斯分类器

```python
class classifier:
    def __init__(self, getfeatures):
        # Counts of feature/category combinations
        self.fc = {}
        # Counts of documents in each category
        self.cc = {}
        self.getfeatures = getfeatures
```

- fc 用于存放特征在不同分类下的数量， 如 {'python': {'bad': 0, 'good': 6}, 'money': {'bad': 5, 'good': 1}}
- cc 用于记录个分类被使用的次数，在后面算概率的时候需要用到
- getfeatures实际上就是docclass.py文件中的getwords函数

在该对象中加入几个函数，实现分类器的训练:

```python
    # 增加对特征/分类组合的计数值
    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    # 增加某一个分类的计数值:
    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    # 计算某一个特征在某一个分类中出现的次数
    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return self.fc[f][cat]
        else:
            return 0.0

    # 属于某一个分类的文档总数
    def catcount(self, cat):
        if cat in self.cc:
            return self.cc[cat]
        return 0

    # 所有的文档总数
    def totalcount(self):
        return sum(self.cc.values())

    # 所有文档的种类
    def categories(self):
        return self.cc.keys()
      
    # train函数接受一个文档和其所属分类（‘good’或者‘bad’），
    # 利用我们定义的getwords函数，对文档进行划分，划分成一个个独立的单词，
    # 然后调用incf函数，针对该分类为每个特征增加计数值，
    # 最后增加该分类的总计数值:
    def train(self, item, cat):
        features = self.getfeatures(item)
        # 针对该分类，为每个特征增加计数值
        for f in features:
            self.incf(f, cat)
        # 增加该分类的计数值
        self.incc(cat)
    
    # 上面的函数计算了每一个特征，在某一个分类中出现的次数，
    # 这个函数将其转换为概率，
    # 对于特定的单词，计算它在某个分类中所占的比例
    # 某个分类中出现该单词的文档数目 / 该分类的文档总数
    def fprob(self, f, cat):
        if self.catcount(cat) == 0:
            return 0
        # 特征在该分类中出现的次数 /
        # 该特征下文档的总数目
        return self.fcount(f, cat) / self.catcount(cat)
      
    # 在训练的样本中，由于单词‘money’只出现了一次，并且是一个赌博类的广告，
    # 因此被分类‘bad’类，那我们计算p('money' | 'good') = 0，
    # 这是非常危险和不公平的，由于我们训练样本的缺失，
    # 导致所有含有‘money’这个单词的文档都被判断为‘bad’类文档。
    # 显然这种结果是我们不愿意接受的，因此我们对概率进行一些加权，
    # 使一些即使在训练样本中没有出现的单词，在求条件概率的时候，不至于为0。具体做法如下：
    def weightedprob(self, f, cat, prf, weight=1, ap=0.5):
        # 使用fprob函数计算原始的条件概率
        basicprob = prf(f, cat)
        totals = sum([self.fcount(f, c) for c in self.categories()])
        bp = ((weight * ap) + (totals * basicprob)) / (weight + totals)
        return bp
```

当然了，分类器需要给它大量的数据进行训练，我们先通过一个简单的demo来看一下分类器是如何通过数据进行训练的:

```python
def sampletrain(cl):
     cl.train('nobody owns the water','good')
     cl.train('the quick rabbit jumps fences','good')
     cl.train('buy phamaceuticals now','bad')
     cl.train('make quick money at the online casino','bad')
     cl.train('the quick borwn fox jumps','good')
```

我们在这个分类器中实现了什么？实际上，在这个分类器中我们只是实现了对于单个特征（这里指的是单词），我们如何得到它的概率，即P(feature | category)。但是，对于现在的场景，我们接受到的是一个影评数据，也就是说，我们接受到的是一个document，现在我们想想如何计算P(Document | category)，我们可以通过朴素贝叶斯分类器来计算，朴素贝叶斯分类器它的前提是各个特征之间是相对独立的，尽管有时候并不如此，但是该分类器以其简单，有用，还是能够在文本分类中起到一定的效果。

假设我们已经注意到，有20%的‘bad’文档出现了‘python’单词- P('python'| 'bad') = 0.2，同时有80%的文档出现了单词‘casino’-P('casino'| 'bad')=0.8,那么当‘python’和‘casino’同时出现在一篇‘bad’文档的概率是P('casino' & 'python' | 'bad') = 0.8 * 0.2 = 0.16。
我们新建一个子类，继承自classifier,取名naivebayes,并添加一个docprob函数作者：

```python
class naivebayes(classifier):
    def __init__(self, getfeatures):
        classifier.__init__(self, getfeatures)

    # 这个函数用来计算P(document | category)的概率
    # 这里用到贝叶斯分类器的思想，即各个特征之间是相互独立的
    def docprob(self, item, cat):
        features = self.getfeatures(item)
        # Multiply the probabilities of all the features together
        p = 1
        for f in features:
            p *= self.weightedprob(f, cat, self.fprob)
        return p
```

**现在我们已经知道了如何计算P(Document|category)，但是我们需要知道的是，最终我们需要的结果是P(category|Document)，换而言之，对于一篇给定的文档，我们需要找出它属于各个分类的概率，我们感到欣慰的是，这就是贝叶斯需要解决的事情**
**在本例中：P(category|Document) = P(Document|category) \* P(category) / P(Document)P(Document|category) 已经被我们用 docprob 函数计算出来了，P(category)也很好理解和计算：代表我们随你选择一篇文档，它属于某个分类的概率。P(Document)对于所有的文档来说，都是一样的，我们直接选择忽略掉他**

```python
	def prob(self, item, cat):
        catprob = self.catcount(cat)/self.totalcount()
        docprob = self.docprob(item, cat)
        return docprob * catprob
```

然后，我们为分类器添加一个方法，接受到一个文档，就输出概率值最大的那个类别：

```python
    def classify(self, item):
        max = 0.0
        best = 'good'
        for cat in self.categories():
            probs = self.prob(item, cat)
            if probs > max:
                max = probs
                best = cat
        return best
```

当然，这时候我们就可以把我们的训练数据喂进来了，还记得上面提到的sampletrain函数吗，现在，我们改写这个函数，来读取我们的数据集:

```python
# 用于训练大量的数据
def sampletrain(cl, traindata, traintarget):
    for left, right in zip(traindata, traintarget):
        cl.train(left, right)


# 我们可以把需要训练的数据放在一个list里面或者迭代器里面，
# 其对应的分类也是如此
# 在函数中，我们使用traindata, traintarget分别替代我们的训练数据
# 和其对应的分类
def get_dataset():
    data = []
    for root, dirs, files in os.walk(r'./input/tokens/neg'):
        for file in files:
            realpath = os.path.join(root, file)
            with open(realpath, errors='ignore') as f:
                data.append((f.read(), 'bad'))
    for root, dirs, files in os.walk(r'./input/tokens/pos'):
        for file in files:
            realpath = os.path.join(root, file)
            with open(realpath, errors='ignore') as f:
                data.append((f.read(), 'good'))
    random.shuffle(data)
    return data


def train_and_test_data(data_):
    filesize = int(0.7 * len(data_))
    # 训练集和测试集的比例为7:3
    train_data_ = [each[0] for each in data_[:filesize]]
    train_target_ = [each[1] for each in data_[:filesize]]

    test_data_ = [each[0] for each in data_[filesize:]]
    test_target_ = [each[1] for each in data_[filesize:]]

    return train_data_, train_target_, test_data_, test_target_
```

写到这里，我们自己手写的贝叶斯分类器就算完成了，大家可以自己训练并预测以下，大概的预测准确率在68%左右，我们使用简单的、完全手写的贝叶斯分类器能够达到这样的效果，还是很不错的。

## 2.3 使用sklearn自带的贝叶斯分类器

如果对sklearn中贝叶斯分类器不熟悉的话，建议先去查看一下相关的资料，sklearn中封装了三种贝叶斯分类器，分别是多项式朴素贝叶斯分类器，伯努利朴素贝叶斯分类器，高斯朴素贝叶斯分类器，在这三种分类器中，前两种经常在文本分类中使用，我们通过如下的代码来训练模型，并预测结果

```python
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
```

得到的结果如下：

```python
Multinomial 0.7410926365795725
BernoulliNB 0.7814726840855107
```

可以看出，用sklearn自带的分类器效果会比我们自己写的贝叶斯分类器效果要好。