import string
import codecs

posSet = set()

posDict = {"Ag":0, "Bg":1, "Dg":2, "Mg":3, "Ng":4, "Qg":5, "Rg":6, "Tg":7, "Vg":8, "Yg":9, "a":10, "ad":11,
            "an":12, "b":13, "c":14, "d":15, "e":16, "email":17, "f":18, "h":19, "i":20, "j":21, "k":22, "l":23, 
            "m":24, "n":25, "nr":26, "nrf":27, "nrg":28, "ns":29, "nt":30, "nx":31, "nz":32, "o":33, "p":34, "q":35,
             "r":36, "s":37, "t":38, "tele":39, "u":40, "v":41, "vd":42, "vn":43, "w":44, "www":45, "x":46, "y":47, "z":48}

with codecs.open("data/cfh_all/train/text", "r", "gbk", "ignore") as f:
    for line in f:
        tokens = line.strip().split("\t")
        words = tokens[1].split(" ") + tokens[2].split(" ") 
        for word in words:
            word = word.strip()
            i = word.rfind(":")
            pos = word[i + 1:]
            posSet.add(pos)

l = 0
for pos in sorted(posSet):
    print "\"%s\":%d," % (pos, l),
    l += 1
