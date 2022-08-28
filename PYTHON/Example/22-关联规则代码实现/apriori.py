

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))

def scanD(D,CK,minSupport):
    ssCnt = {}
    for tid in D:
        for can in CK:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(list(D)))
    retlist = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retlist.insert(0,key)
        supportData[key] = support
    return retlist,supportData
    
def aprioriGen(LK,k):
    retlist = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i+1,lenLK):
            L1 = list(LK[i])[:k-2]
            L2 = list(LK[j])[:k-2]
            if L1 == L2:
                retlist.append(LK[i]|LK[j])
    return retlist

def apriori(dataSet,minSupport=0.5):
    C1 = createC1(dataSet)
    L1,supportData = scanD(dataSet,C1,minSupport)
    L = [L1]
    k = 2    
    while(len(L[k-2]) > 0):
        CK = aprioriGen(L[k-2],k)
        LK,supk = scanD(dataSet,CK,minSupport)
        supportData.update(supk)
        L.append(LK)
        k += 1
    return L,supportData

def generateRules(L,supportData,minConf=0.6):
    rulelist = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item])for item in freqSet]
            rulessFromConseq(freqSet,H1,supportData,rulelist,minConf)
        
    
def rulessFromConseq(freqSet,H,supportData,rulelist,minConf=0.6):   
    m=len(H[0])
    while (len(freqSet) > m):
        H = calConf(freqSet,H,supportData,rulelist,minConf)
        if (len(H)>1):
            aprioriGen(H,m+1)
            m += 1
        else:
            break
    
def calConf(freqSet,H,supportData,rulelist,minConf=0.6):
    prunedh = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            rulelist.append((freqSet-conseq,conseq,conf))
            prunedh.append(conseq)
    return prunedh

if __name__ == '__main__':
    dataSet = loadDataSet()
    L,support = apriori(dataSet)
    i = 0 
    for freq in L:
        print ('项数',i+1,':',freq)
        i+=1
    rules = generateRules(L,support,minConf=0.5)
      
    
    
    
    
    
    