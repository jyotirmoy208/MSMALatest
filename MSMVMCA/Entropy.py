from scipy.stats import entropy
import numpy as np


def computeEntophy(testlabels, predictlabels):
    cluster1count = 0
    cluster0count = 0
    onecountFirst = 0
    secondcont = 0
    predlabeltscount=0
    for key in predictlabels:
        if key == 1:
            cluster1count = cluster1count + 1

            if predlabeltscount < 97:
                onecountFirst = onecountFirst + 1
        if key == 0:
            cluster0count = cluster0count + 1

            if predlabeltscount < 97:
                secondcont = secondcont + 1
        predlabeltscount=predlabeltscount+1
    problist1st = []
    problist1st.append(onecountFirst / cluster1count)
    problist1st.append((cluster1count - onecountFirst) / cluster1count)
    firstclusent = entropy(problist1st)
    problist2nd = []
    problist2nd.append(secondcont / cluster0count)
    problist2nd.append((cluster0count - secondcont) / cluster0count)
    secndclusent = entropy(problist2nd)
    totalentrophy=firstclusent*(cluster1count/len(predictlabels))+secndclusent*(cluster0count/len(predictlabels))
    return totalentrophy

def computeEntrophyForGlass(testlabels, predictlabels):
    cluster1count = 0
    cluster0count = 0
    cluster2count=0
    cluster3count=0
    cluster4Count=0
    cluster5count=0
    onecountFirst = 0
    secondcont = 0
    thirdcount=0
    fourthcount=0
    fiftcount=0
    sixthcount=0
    predlabeltscount = 0

    for key in predictlabels:
        if key == 1:
            cluster1count = cluster1count + 1

            if predlabeltscount <= 69 & predlabeltscount>=0 :
                onecountFirst = onecountFirst + 1
        if key == 2:
            cluster2count = cluster2count + 1

            if predlabeltscount <= 145 & predlabeltscount>=70:
                secondcont = secondcont + 1
        if key == 3:
            cluster3count = cluster3count + 1

            if predlabeltscount <= 162 & predlabeltscount>=146:
                thirdcount = thirdcount + 1
        if key == 4:
            cluster4Count = cluster4Count + 1

            if predlabeltscount <= 175 & predlabeltscount>=163:
                fourthcount = fourthcount + 1
        if key == 5:
            cluster5count = cluster5count + 1

            if predlabeltscount <= 184 & predlabeltscount>=176:
                fiftcount = fiftcount + 1
        if key == 0:
            cluster0count = cluster0count + 1

            if  predlabeltscount>=184:
                sixthcount = sixthcount + 1
        predlabeltscount = predlabeltscount+1
    problist1st = []
    problist1st.append(onecountFirst / cluster1count)
    problist1st.append((cluster1count - onecountFirst) / cluster1count)
    firstclusent = entropy(problist1st)
    problist2nd = []
    problist2nd.append(secondcont / cluster2count)
    problist2nd.append((cluster2count - secondcont) / cluster2count)
    secndclusent = entropy(problist2nd)

    problist3rd = []
    problist3rd.append(thirdcount / cluster3count)
    problist3rd.append((cluster3count - thirdcount) / cluster3count)
    thirdclusent = entropy(problist3rd)

    problist4th = []
    problist4th.append(fourthcount / cluster4Count)
    problist4th.append((cluster4Count - fourthcount) / cluster4Count)
    fourthclusent = entropy(problist4th)

    problist5th = []
    problist5th.append(fiftcount / cluster5count)
    problist5th.append((cluster5count - fiftcount) / cluster5count)
    fifthclusent = entropy(problist5th)

    problist6th = []
    problist6th.append(sixthcount / cluster0count)
    problist6th.append((cluster0count - sixthcount) / cluster0count)
    sixthclusent = entropy(problist6th)
    totalentrophy = firstclusent * (cluster1count / len(predictlabels)) + secndclusent * (
    cluster2count / len(predictlabels))+ thirdclusent * (
    cluster3count / len(predictlabels))+ fourthclusent * (
    cluster4Count / len(predictlabels))+ fifthclusent * (
    cluster5count / len(predictlabels))+ sixthclusent * (
    cluster0count / len(predictlabels))
    return totalentrophy
if __name__ == '__main__':
        test_labels = np.zeros(208, int)
        test_labels[0:97] = np.int(1)
        pred_labels = np.zeros(208, int)
        pred_labels[0:50] = np.int(1)
        totalentrophy=computeEntophy(test_labels,pred_labels)
        print(totalentrophy)

