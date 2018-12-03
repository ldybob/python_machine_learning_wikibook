# -*- coding: utf-8 -*-

import codecs
import sys
from bs4 import BeautifulSoup
from konlpy.tag import Twitter
from gensim.models import word2vec


readFp = codecs.open("wiki.txt", "r", encoding="utf-8")
wakati_file = "wiki.wakati"
writeFp = open(wakati_file, "w", encoding="utf-8")

non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

twitter = Twitter()
i = 0
while True:
    try:
        line = readFp.readline()
        line = line.translate(non_bmp_map)
        if not line: 
            break
        if i % 20000 == 0:
            print("current - " + str(i))
        i += 1
        malist = twitter.pos(line, norm=True, stem=True)
        r = []
        for word in malist:
            if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                writeFp.write(word[0] + " ")                 
    except UnicodeEncodeError as e:
        if e.reason == 'surrogates not allowed':
            print(e)   
            
writeFp.close()

data = word2vec.Text8Corpus("wiki.wakati")
model = word2vec.Word2Vec(data, size=100)
model.save("wiki.model")
print("ok")