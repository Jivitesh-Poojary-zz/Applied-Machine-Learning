#! /usr/bin/python

# Project partners: Atharva Tere (agtere@iu.edu) & Jivitesh Poojary (jpoojary@iu.edu)

# We had a discussion with Rin, Swami and Prashant before writing this code

'''
This is a template outlining the functions we are expecting for us to be able to
interface with an call your code. This is not all of the functions you need. You
will also need to make sure to bring your decision tree learner in somehow
either by copying your code into a learn_decision_tree function or by importing
your decision tree code in from the file your wrote for PA#1. You will also need
some functions to handle creating your data bags, computing errors on your tree,
and handling the reweighting of data points.

For building your bags numpy's random module will be helpful.
'''

# This is the only non-native library to python you need

import numpy as np;
import sys, os;


#
# This function reads the data from a csv file located at a speific location passed through command line arguments
#
def read_data(floc):
    # print("Location --> ",floc)
    import csv
    with open(floc, 'r') as f:
        reader = csv.reader(f)
        test_list = list(reader)
    if len(test_list) > 0:
        return test_list
    else:
        print("Read failed...")
        return None


# -------------------------------------------------------------------------------------------------------------------------------------
# This function is used to remove the first row/header of the csv file.
#
def remove_header(data):
    data.pop(0)
    if len(data) > 0:
        return data
    else:
        print("No data, only headers")
        import sys
        sys.exit("Exiting...")
        return None


# -------------------------------------------------------------------------------------------------------------------------------------
# Initializing a dictionary; key=distinct element in the class variable, value=count of each variable
#
def dist_count(rows):
    value_set = {}
    for row in rows:
        row_val = row[0]
        if row_val not in value_set: value_set[row_val] = 0
        value_set[row_val] += 1
    return value_set
# -------------------------------------------------------------------------------------------------------------------------------------
# Modifies the class lable [+1 remains +1, 0 is chaged to -1]
#
def classmod(data):
    data.pop(0)
    for i in range(len(data)):
        if int(data[i][0]) == 0:
            data[i][0] = -1
    return data
# -------------------------------------------------------------------------------------------------------------------------------------
# Changes all values from char type to int type
#
def char_to_int(data):
    icount = len(data) - 1
    for i in range(0, icount):
        jcount = len(data[i])
        for j in range(0, jcount):
            data[i][j] = int(data[i][j])
    if len(data) > 0:
        return data
    else:
        print("Something is wrong in char to int")
        import sys
        sys.exit("Exiting...")
        return None
# -------------------------------------------------------------------------------------------------------------------------------------
# Used to split data based on a column value
#
def split(r, c, v):
    p1 = []
    p2 = []
    for row in r:
        if int(row[c]) == int(v):
            p1.append(row)
        else:
            p2.append(row)
    return p1, p2
# -----------------------------------------------------------------------------------------------------------------------------
# Used to calculate frequencies of class labels
#
def kvp(data):
    value_set = {}
    for i in data:
        val = i[0]
        if val not in value_set: value_set[val] = 0
        value_set[val] += 1
    return value_set
# -----------------------------------------------------------------------------------------------------------------------------
# Weighted entropy calculation for Boosting
#
def encal(data):
    vs = kvp(data)
    from math import log
    ent = 0.0
    for i in vs.keys():
        n_e = 0
        n_s = len(data)
        # n_s=0
        # for k in data:
        #     n_s=n_s+k[126]
        for j in data:
            if j[0] == i:
                n_e = n_e + j[126]
        prob = 0.0
        prob = n_e / n_s
        if (prob <= 0):
            print(
                "################################## ALARM - Prob is -ve ######################################### Prob=",
                prob)
        if prob == 0.0:
            print(
                "################################## ALARM - Prob is 0 ######################################### Prob=",
                prob)
            ent = 999.99
        else:
            ent -= prob * (log(prob) / log(2))
    return ent


# -------------------------------------------------------------------------------------------------------------------------------------
# Entropy calculation for bagging
#
def encal_Bag(dataset):
    from math import log
    c_count = dist_count(dataset)  # Count of class variable
    c_total = len(dataset)  # Total observations in class variable
    entropy = 0.0
    for i in c_count.keys():  # For each distict value in class attribute
        prob = 0.0
        prob = c_count[i] / c_total  # Since P(E) = n(E)/n(S)
        entropy -= prob * (log(prob) / log(2))  # Since the python default is log to the base 2
    return (entropy)

# -----------------------------------------------------------------------------------------------------------------------------
# For boosting
class decision_node:
    def __init__(self, col_index=-1, val_node=1, ccount=None, t_b=None, f_b=None, leaf_node=0):
        self.col_index = col_index
        self.val_node = val_node
        self.ccount = ccount
        self.t_b = t_b
        self.f_b = f_b
        self.leaf_node = leaf_node

# -----------------------------------------------------------------------------------------------------------------------------
# For bagging
class node:
    def __init__(self, col_index=-1, val_node=None, ccount=999, t_b=None, f_b=None):
        self.col_index = col_index
        self.val_node = val_node
        self.ccount = ccount
        self.t_b = t_b
        self.f_b = f_b

# -----------------------------------------------------------------------------------------------------------------------------
# Decision tree learner for boosting
#
def build_for_boost(data, used=[], tn=0, fn=0):
    ent = encal(data)
    info = 0.0
    datasets = None
    col_value_package = None
    for i in range(1, 125):
        # print("Used", used)
        if i not in used:
            # print("Splitting on ",i)
            d1, d2 = split(data, i, 1)
            if len(d1) > 0 and len(d2) > 0:
                ent_d1 = encal(d1)
                ent_d2 = encal(d2)
                prob1 = len(d1) / len(data)
                prob2 = len(d2) / len(data)

                # Uncertainity after splitting
                u = prob1 * ent_d1 + prob2 * ent_d2

                # Information gain
                ig = ent - u

                if ig > info and len(d1) > 0 and len(d2) > 0:
                    info = ig
                    col_value_package = (i, 1)
                    datasets = d1, d2
            else:
                continue
    if info > 0:
        used.append(col_value_package[0])
        tn = tn + 1
        t = build_for_boost(datasets[0], used, tn, fn)
        fn = fn + 1
        f = build_for_boost(datasets[1], used, tn, fn)
        return decision_node(col_index=col_value_package[0], ccount=kvp(data), t_b=t, f_b=f, leaf_node=0)
    else:
        return decision_node(ccount=kvp(data), leaf_node=1)

# -----------------------------------------------------------------------------------------------------------------------------
# Decision tree learner for bagging
#
def build_for_bag(data):
    ent = encal_Bag(data)
    info = 0.0
    datasets = None
    col_value_package = None

    cols = len(data[0]) - 1

    for c in range(1, cols):

        global column_values
        column_values = {}

        for r in data:
            column_values[r[c]] = 1

        for val in column_values.keys():
            ds1, ds2 = split(data, c, val)
            prob1 = len(ds1) / len(data)
            prob2 = len(ds2) / len(data)
            info_temp_sample = prob1 * encal_Bag(ds1) + prob2 * encal_Bag(ds2)
            info_temp = ent - info_temp_sample

            if info_temp > info and len(ds1) > 0 and len(ds2) > 0:
                info = info_temp
                col_value_package = (c, val)
                # print('####', col_value_package)
                datasets = (ds1, ds2)
    # Check if the meax information gain is positive
    if info > 0:
        t = build_for_bag(datasets[0])
        f = build_for_bag(datasets[1])

        # print('%%%%%', t, f)
        return node(col_index=col_value_package[0], val_node=col_value_package[1], t_b=t, f_b=f)
    # If max information gain is not positive, you have reached a leaf node
    else:
        return node(ccount=data[0][0])

# -----------------------------------------------------------------------------------------------------------------------------
# Classify function for bagging
# Classifies give instance by using the decision tree
#
def classify_Bag(tuple, dt, d, cnt=0):
    if dt.ccount != 999:
        return dt.ccount
    else:
        v = tuple[dt.col_index]
        b = None
        if v == dt.val_node:
            if d < 0:
                return 1  # Classified as TRUE(1)
            else:
                b = dt.t_b
        else:
            if d < 0:
                return 0  # Classified as FALSE(0)
            else:
                b = dt.f_b
    cnt += 1
    d -= 1
    return classify_Bag(tuple, b, d, cnt)


# -------------------------------------------------------------------------------------------------------------------------------------
# Classifier wrapper for bagging
# This function is used to classify values in the test dataset by calling the classify function for each not row.
#
def classifier_Bag(TestFinal, dt, d, numbags):
    for i in TestFinal:
        classVar = []
        if d == 0:
            i.append(1)
        else:  # Bagging Logic -
            for j in range(0, numbags):
                classVar.append(str(classify_Bag(i, dt[j], d)))
                # print(i[0], '####' , classVar, '***', max(classVar))
            from statistics import mode
            #i.append(int(max(classVar)))
            i.append(int(mode(classVar)))
    CorrectClassify = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(TestFinal)):
        # print(TestFinal[i][0],'***',TestFinal[i][len(TestFinal[i])-1])
        if TestFinal[i][0] == TestFinal[i][len(TestFinal[i]) - 1]:
            CorrectClassify += 1
            if TestFinal[i][0] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if TestFinal[i][0] == 1:
                fn += 1
            else:
                fp += 1
    return CorrectClassify / len(TestFinal), tp, tn, fp, fn

# -----------------------------------------------------------------------------------------------------------------------------
# Classification function for boosting
# Uses the decision tree to classify given instance
#
def classify(tuple, dt, d, cnt=0):
    if dt.leaf_node == 1:
        if 1 in dt.ccount:
            return 1
        else:
            return -1
    else:
        v = tuple[dt.col_index]
        b = None
        if int(v) == 1:
            if d <= 0:
                if 1 in dt.ccount and -1 in dt.ccount:
                    if dt.ccount[1] > dt.ccount[-1]:
                        o = 1
                    else:
                        o = -1
                    return o
                else:
                    if 1 in dt.ccount:
                        return 1
                    else:
                        return -1
            else:
                b = dt.t_b
        else:
            if d <= 0:
                if 1 in dt.ccount and -1 in dt.ccount:
                    if dt.ccount[1] > dt.ccount[-1]:
                        o = 1
                    else:
                        o = -1
                    return o
                else:
                    if 1 in dt.ccount:
                        return 1
                    else:
                        return -1
            else:
                b = dt.f_b
    cnt += 1
    d -= 1
    return classify(tuple, b, d, cnt)

# -----------------------------------------------------------------------------------------------------------------------------
# Wrapper code for classification boosting
# Classifies test data set by calling the classify function iteratively
#
def classifier(data, dt, d, iter, count=126):
    for i in data:
        i.append(classify(i, dt, d))
    CorrectClassify = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(data)):
        if data[i][0] == data[i][count + iter]:
            CorrectClassify += 1
            if data[i][0] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if data[i][0] == 1:
                fn += 1
            else:
                fp += 1
    return CorrectClassify / len(data), tp, tn, fp, fn, data

# -----------------------------------------------------------------------------------------------------------------------------
# Function to update weights of instances needed for boosting
#
def weights_updater(data, alpha):
    import math
    n = 0
    for i in range(len(data)):
        if data[i][0] == data[i][127]:
            data[i][126] = data[i][126] * math.exp(-1 * alpha)
            n = n + 1
        else:
            data[i][126] = data[i][126] * math.exp(alpha)
    tw = 0.0
    for i in range(len(data)):
        tw = tw + data[i][126]
    for i in range(len(data)):
        data[i][126] = data[i][126] / tw
    return data, n

# -----------------------------------------------------------------------------------------------------------------------------
# Final predictions for boosting
#
def final_preds(testp2, alpha):
    for i in testp2:
        i[136] = alpha[0] * i[126] + alpha[1] * i[127] + alpha[2] * i[128] + alpha[3] * i[129] + alpha[4] * i[130] + \
                 alpha[5] * i[131] + alpha[6] * i[132] + alpha[7] * i[133] + alpha[8] * i[134] + alpha[9] * i[135]
        i[137] = alpha[0] * i[126] + alpha[1] * i[127] + alpha[2] * i[128] + alpha[3] * i[129] + alpha[4] * i[130]
        if i[136] > 0:
            i[136] = 1
        else:
            i[136] = -1
        if i[137] > 0:
            i[137] = 1
        else:
            i[137] = -1
    return testp2
# -----------------------------------------------------------------------------------------------------------------------------
# Boosting accuracy calculations
# Trees=5
#
def acc5(testp2):
    c = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in testp2:
        if i[0] == i[137]:
            c = c + 1
            if i[0] == 1:
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            if i[0] == 1:
                fn = fn + 1
            else:
                fp = fp + 1
    return c / len(testp2), tp, tn, fp, fn
# -----------------------------------------------------------------------------------------------------------------------------
# Boosting accuracy calculations
# Trees=10
#
def acc10(testp2):
    c = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in testp2:
        if i[0] == i[136]:
            c = c + 1
            if i[0] == 1:
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            if i[0] == 1:
                fn = fn + 1
            else:
                fp = fp + 1
    return c / len(testp2), tp, tn, fp, fn


'''
Function: load_and_split_data(datapath)
datapath: (String) the location of the UCI mushroom data set directory in memory

This function loads the data set. datapath points to a directory holding
agaricuslepiotatest1.csv and agaricuslepiotatrain1.csv. The data from each file
is loaded and returned. All attribute values are nomimal. 30% of the data points
are missing a value for attribute 11 and instead have a value of "?". For the
purpose of these models, all attributes and data points are retained. The "?"
value is treated as its own attribute value.

Two nested lists are returned. The first list represents the training set and
the second list represents the test set.
'''


def load_data(datapath, flag):
    train = read_data(datapath + 'agaricuslepiotatrain1.csv')
    test = read_data(datapath + 'agaricuslepiotatest1.csv')
    # Deleting column bruises-no from Train data
    trainp = []
    for i in train:
        temp = []
        for j in range(0, 127):
            if j != 21:
                temp.append(i[j])
        trainp.append(temp)
    testp = []
    # Deleting column bruises-no from Test data
    for i in test:
        temp = []
        for j in range(0, 127):
            if j != 21:
                temp.append(i[j])
        testp.append(temp)
    # Changing the class label index to 0
    for i in trainp:
        t = i[0]
        i[0] = i[20]
        i[20] = t
    # Changing the class label index to 0
    for i in testp:
        t = i[0]
        i[0] = i[20]
        i[20] = t
    # Changing the class label(+1 remains as +1, 0 becomes -1)
    if flag != "bag":
        testp2 = classmod(testp)
        trainp2 = classmod(trainp)
    # Changing all data types to integer
    if flag != "bag":
        testp2 = char_to_int(testp2)
        trainp2 = char_to_int(trainp2)
    else:
        testp.pop(0)
        trainp.pop(0)
        testp2 = char_to_int(testp)
        trainp2 = char_to_int(trainp)
    return trainp2, testp2


'''
Function: learn_bagged(tdepth, numbags, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numbags: (Integer)the number of bags to use to learn the trees
datapath: (String) the location in memory where the data set is stored

This function will manage coordinating the learning of the bagged ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''


def learn_bagged(tdepth, numbags, datapath):
    trainp2, testp2 = load_data(datapath, "bag")

    trainingData = {}

    # Storing decision trees in an array to be able to iterate on
    dt = []

    row_count = 0
    ranVal = []

    for row in trainp2:
        b = row
        trainingData.update({row_count: b})
        row_count += 1

    # Create ensembles and learn decision trees
    for i in range(0, numbags):
        ranVal = np.random.choice(range(1, row_count - 1), row_count - 1)
        bagList = []
        for j in range(0, row_count - 1):
            bagList.append(trainingData[ranVal[j]])

        dtCurrent = build_for_bag(bagList)  # Call build function to create the decision tree with EM_Training_Data_1 data
        dt.append(dtCurrent)

    # Call the classifier function to classify using ensembles
    a, tp, tn, fp, fn = classifier_Bag(testp2, dt, tdepth, numbags)

    print("Bagging completed with following results:")
    print("Depth=", tdepth, " | Trees = ", numbags, " | Accuracy =", a, " TP =", tp, " TN =", tn, " FP =", fp, " FN =",
          fn)


'''
Function: learn_boosted(tdepth, numtrees, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numtrees: (Integer) the number of boosted trees to learn
datapath: (String) the location in memory where the data set is stored

This function wil manage coordinating the learning of the boosted ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''


def learn_boosted(tdepth, numtrees, datapath):
    trainp2, testp2 = load_data(datapath, "boost")
    initial_weight = 0.0

    # Initial weight is set to 1/N
    initial_weight = 1 / len(trainp2)

    for i in trainp2:
        i.append(initial_weight)

    alpha = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    trees = [None, None, None, None, None, None, None, None, None, None]
    k=1
    for em in range(10):
        # print(len(trainp2[0]))
        # print("Learning tree")
        trees[em] = build_for_boost(trainp2)
        acc, tp, tn, fp, fn, trainp2 = classifier(trainp2, trees[em], tdepth, em + 1)
        error=0.0
        for it in range(len(trainp2)):
            if trainp2[it][0] != trainp2[it][126 + k]:
                error = error + trainp2[it][126]
        k = k + 1
        import math
        alpha[em] = 0.5 * math.log((1 - error) / error)
        trainp2, n = weights_updater(trainp2, alpha[em])
    for em in range(10):
        acc, tp, tn, fp, fn, testp2 = classifier(testp2, trees[em], tdepth, em + 1, 125)
    for i in testp2:
        i.append(0)
    for i in testp2:
        i.append(0)
    testp2 = final_preds(testp2, alpha)
    print("Boosting Completed with following results:")
    if numtrees == 5:
        a, tp, tn, fp, fn = acc5(testp2)
        print("Depth=", tdepth, " | Trees = 5 | Accuracy =", a, " TP =", tp, " TN =", tn, " FP =", fp, " FN =", fn)
    else:
        a, tp, tn, fp, fn = acc10(testp2)
        print("Depth=", tdepth, " | Trees = 10 | Accuracy =", a, " TP =", tp, " TN =", tn, " FP =", fp, " FN =", fn)


if __name__ == "__main__":
    # The arguments to your file will be of the following form:
    # <ensemble_type> <tree_depth> <num_bags/trees> <data_set_path>
    # Ex. bag 3 10 mushrooms
    # Ex. boost 1 10 mushrooms

    # Get the ensemble type
    entype = sys.argv[1];
    # Get the depth of the trees
    tdepth = int(sys.argv[2]);
    # Get the number of bags or trees
    nummodels = int(sys.argv[3]);
    # Get the location of the data set
    datapath = sys.argv[4];

    # Check which type of ensemble is to be learned
    if entype == "bag":
        # Learned the bagged decision tree ensemble
        learn_bagged(tdepth, nummodels, datapath);
    else:
        # Learned the boosted decision tree ensemble
        learn_boosted(tdepth, nummodels, datapath);

