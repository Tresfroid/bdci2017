# -*- coding: utf-8 -*-  #
import  re
sen="剛買 四個 月 就 開始 有 問題 拖延 症 一直 拖著 , , "
# sen = re.sub("(,\s)+", ", ", sen)
# sen1 = re.sub("\s+", " ", sen)
# sent=sen1.strip(",")
# sent=sent.strip(" ")
# sent=sent.strip(",")
sent=re.sub("\s\,\s\,\s", " , " , sen)
# sent=sen.strip(",")
print  sent