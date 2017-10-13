import re

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
