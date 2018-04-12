from docclass import getwords
import random
import os


# http://www.jianshu.com/p/b8e0ae7cfa39
class classifier:
    def __init__(self, getfeatures):
        # Counts of feature/category combinations
        # {'python': {'bad': 0, 'good': 6}, 'money': {'bad': 5, 'good': 1}}
        self.fc = {}
        # Counts of documents in each category
        # 变量cc是一个记录各分类被使用次数的词典。这一信息是我们稍后讨论的概率计算所需的。
        # {'good':100, 'bad':100}
        self.cc = {}
        # 该函数作用是从即将被归类的文档中提取出特征来
        self.getfeatures = getfeatures

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


# 朴素贝叶斯分类器
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

    # 上面只是获得了P(document | category)的概率
    # 但是我们需要的是 P(category | document)的概率
    # 我们借助贝叶斯公式 P(category | document) = P(document | category) * P(category) / P(document)
    def prob(self, item, cat):
        catprob = self.catcount(cat) / self.totalcount()
        docprob = self.docprob(item, cat)
        return docprob * catprob

    def classify(self, item):
        max = 0.0
        best = 'good'
        for cat in self.categories():
            probs = self.prob(item, cat)
            if probs > max:
                max = probs
                best = cat
        return best


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


# cl = classifier(getwords)
# cl.train('the quick brown fox jumps over the lazy dog', 'good')
# cl.train('make quick money in the online casino', 'bad')
# print(cl.fcount('quick', 'good'))
# print(cl.fcount('quick','bad'))
# sampletrain(cl)
# print(cl.fprob('quick','good'))
# cl = classifier(getwords)
# sampletrain(cl)
# print(cl.fprob('money','good'))
# print(cl.weightedprob('money','good', cl.fcount))

# cl = naivebayes(getwords)
# sampletrain(cl)
# print(cl.prob('quick rabbit', 'good'))
# print(cl.prob('quick rabbit', 'bad'))
# cl = naivebayes(getwords)
# sampletrain(cl)
# print(cl.classify('quick rabbit'))

if __name__ == '__main__':
    cl = naivebayes(getwords)
    data = get_dataset()
    train_data, train_target, test_data, test_target = train_and_test_data(data)
    sampletrain(cl, train_data, train_target)  # 对训练我们的分类器进行训练
    predict = []
    for each in test_data:
        predict.append(cl.classify(each))
    count = 0
    for left, right in zip(predict, test_target):
        if left == right:
            count += 1
    print(count / len(test_target))
