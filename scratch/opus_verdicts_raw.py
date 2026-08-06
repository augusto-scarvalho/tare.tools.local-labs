import json
W={
# batch0
"t005":"2","t009":"2","t015":"1","t018":"2","t023":"2","t025":"2","t027":"2","t031":"1","t034":"2",
# batch1 (t030 missing)
"t006":"1","t012":"2","t016":"1","t019":"1","t024":"2","t026":"1","t032":"2","t035":"1",
# batch2
"t000":"2","t002":"1","t004":"2","t008":"2","t011":"1","t014":"1","t020":"1","t022":"1","t029":"1",
# batch3
"t001":"1","t003":"2","t007":"1","t010":"2","t013":"2","t017":"2","t021":"1","t028":"2","t033":"1",
}
allv={k:{"winner":v,"reason":"opus-4.8 high (blind, isolated)"} for k,v in W.items()}
json.dump(allv, open('runs/a2/gate3/opus_verdicts_pairwise.json','w'), indent=1)
print("verdicts:",len(allv),"(missing t030 -> 1 na expected)")
