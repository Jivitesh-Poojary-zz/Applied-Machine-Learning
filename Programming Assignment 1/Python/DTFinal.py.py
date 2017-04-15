# AML - Programming Assignment 1
#
# Team - Atharva Tere(agtere@iu.edu) and Jivitesh Poojary(jpoojary@iu.edu)
#
# Instructions to run the code:
#
# 1. Please make sure you load the absolute address of test and train data at following places:
#    Files are contained in the zip file attached
# 	A) Line 284: Absolute address of Monk-1-train dataset
# 	B) Line 291: Absolute address of Monk-2-train dataset
# 	C) Line 298: Absolute address of Monk-3-train dataset
# 	D) Line 298: Absolute address of own dataset - training set
# 	E) Line 323: Absolute address of test datasets - be carefule with this one, the training datasets are picked up by the for loop
# 	F) Line 340: Absolute address of own dataset - testing set
#
# 2. Once you click the run button after this, the code will generate the output and a graph will popup. This will be the accuracy curve. Once you close it, the error curve will pop up. You can check the console output for the accuracy, tp, tn, fp and fn at each depth(max 20).
#
# 3. For calculating depth-1 and depth-2 - I send these as arguments to the classify function. Classify fuction will the arguments and limit the number of nodes to be traversed in order to classify the test data at each depth.
# 	A] If the max depth set is n and the tree has depth<n, the complete tree will be considered for classification
#
# 4. Please reach out to me or Jivitesh on the email ids above for any clarification regarding the code
#
# This code was tested on a machine with quadcore 2.4GHz processor and 16GB DDR3 RAM


#-------------------------------------------------------------------------------------------------------------------------------------
# Citations:
#
# 1. Entropy and Information Gain calculation
#   a. Dr. Natarajan's slides
#   b. Weblink: http://www.math.unipd.it/~aiolli/corsi/0708/IR/Lez12.pdf
# 2. A few programming ideas were referenced from this website - http://kldavenport.com/pure-python-decision-trees/
# 3. Python module matplotlib (http://matplotlib.org/citing.html) has been used to create graphs
# 4. Math library was used to calculate logs (https://docs.python.org/2/library/math.html)
# 5. CSV library was used to read data from csv files (https://docs.python.org/2/library/csv.html)
#-------------------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------------------
# This function is used to read data from a csv file and store it as a list
# Input:
#   i. floc = absolute path of the csv file on your system
# Output:
#   i. Returns the list containing the data read from csv file
def read_data(floc):
    import csv
    with open(floc, 'r') as f:
        reader=csv.reader(f)
        test_list = list(reader)
    if len(test_list)>0:
        return test_list
    else:
        print("Read failed...")
        return None

#-------------------------------------------------------------------------------------------------------------------------------------
# This function is used to remove the first row/header of the csv file.
# Input:
#   i. data = list containing header files and data
# Output:
#   i. data = list containing data only
def remove_header(data):
    data.pop(0)
    if len(data)>0:
        return data
    else:
        print("No data, only headers")
        import sys
        sys.exit("Exiting...")
        return None

#-------------------------------------------------------------------------------------------------------------------------------------
# This function is used to ID column(last column in monks data) the first row/header of the csv file.
# Input:
#   i. data = list containing header files and data
# Output:
#   i. data = list containing data only
def remove_id(data):
    for i in range(0, len(data)):
        data[i].pop(len(data[i]) - 1)
    if len(data)>0:
        return data
    else:
        import sys
        print("Something is wrong removing IDs")
        sys.exit("Exiting...")
        return None

#-------------------------------------------------------------------------------------------------------------------------------------
# This function is used to convert the data from characters type to integer type
# Input:
#   i. data = list containing data as char type
# Output:
#   i. data = list containing data as int type
def char_to_int(data):
    icount = len(data) - 1
    for i in range(0, icount):
        jcount = len(data[i])
        for j in range(0, jcount):
            data[i][j] = int(data[i][j])
    if len(data)>0:
        return data
    else:
        print("Something is wrong in char to int")
        import sys
        sys.exit("Exiting...")
        return None

#-------------------------------------------------------------------------------------------------------------------------------------
# This function is used to split data tuples based on a value in a particular column
# Input:
#   i. r = list containing data tuples to be split
#   ii. c = column to be considered
#   iii. v = value of column to be split on
# Output
#   i. p1 = list containing all tuples here value of c'th column is greater than or equal to v
#   ii. p1 = list containing all tuples here value of c'th column is less than v
def split(r, c, v):
    p1 = []
    p2 = []
    for row in r:
        if int(row[c])>=int(v):
            p1.append(row)
        else:
            p2.append(row)
    return p1,p2
#-------------------------------------------------------------------------------------------------------------------------------------

def dist_count(rows):

    # Initializing a dictionary; key=distinct element in the class variable, value=count of each variable
    value_set = {}
    for row in rows:
        # First column is the class that we need distinct counts of
        row_val = row[0]

        # If its a new value, add a new key entry to dictionary
        if row_val not in value_set: value_set[row_val] = 0

        # Increase the count of respective key based on the value
        value_set[row_val] += 1
    return value_set

#-------------------------------------------------------------------------------------------------------------------------------------

def encal(dataset):
    from math import log
    c_count=dist_count(dataset) # Count of class variable
    c_total = len(dataset) # Total observations in class variable
    entropy=0.0
    for i in  c_count.keys(): # For each distict value in class attribute
        prob=0.0
        prob=c_count[i]/c_total # Since P(E) = n(E)/n(S)
        entropy-=prob*(log(prob)/log(2)) # Since the python default is log to the base 2
    return (entropy)

#-------------------------------------------------------------------------------------------------------------------------------------

class node:
    def __init__(self,col_index=-1,val_node=None,ccount=999,t_b=None,f_b=None):
        self.col_index=col_index
        self.val_node=val_node
        self.ccount=ccount
        self.t_b=t_b
        self.f_b=f_b

#-------------------------------------------------------------------------------------------------------------------------------------

# This function is used to build the decision tree
# Steps involved:
#     i) Read data from argument
#     ii) Calculate entropy of the parent data - this is the uncertainity at level j
#     iii) For each value v in every column
#             a. Split the data in two parts such as all the values less than or equal to v are in one dataset and others in other dataset
#             b. Calculate the entropy for each of the split data set
#             c. Multiply the entropy of each split dataset the probability of that dataset {p=n(split data)/n(parent data)}
#             d. Add both of the values => p(ds1)*entropy(ds1)+p(ds2)*entropy(ds2) - this is the uncertainity at level (j+1)
#             e. Subtract the entropy at level j+1 from entropy at level j => this loss in entropy, in turn loss in uncertainity is the information gain
#             f. Choose the best information gain
#     iv. For the column value pair that gives best information gain - split the data on this column-value pair and repear the whole process as long as information gain is 0
#             a. If the information gain is 0, it means you cannot split more and you have reached the leaf node - so return the decision / classification value
def build(data):

    # Entropy of parent data (Uncertainity on level j)
    ent=encal(data)

    # Initializations
    info=0.0
    datasets=None
    col_value_package=None

    cols=len(data[0])-1

    # Since class variable is the first column(index=0), that need not be checked
    for c in range (1,cols):

        # Dictionary initialization
        global column_values
        column_values={}

        for r in data:
            column_values[r[c]]=1

        # Information gain calculation for each value in each column
        for val in column_values.keys():

            # Splitting parent dataset
            ds1,ds2=split(data,c,val)

            # Probability of each dataset
            prob=0.0
            prob1=len(ds1)/len(data)
            prob2 = len(ds2) / len(data)

            # Entropy of split datasets (Uncertainity on level j+1)
            info_temp_sample=prob1*encal(ds1)+prob2*encal(ds2)

            # Decrease in entropy(uncertainity) = information gain
            info_temp=ent-info_temp_sample

            # Choosing the best(max in this case) information gain
            if info_temp>info and len(ds1)>0 and len(ds2)>0:
                info=info_temp
                col_value_package=(c,val)
                datasets=(ds1,ds2)

    # Check if the meax information gain is positive
    if info>0:

        # Recursive call to tree build function with child dataset
        t = build(datasets[0])
        f = build(datasets[1])

        # Return a decision node
        return node(col_index=col_value_package[0],val_node=col_value_package[1],t_b=t,f_b=f)
    # If max information gain is not positive, you have reached a leaf node
    else:
        # Return a leaf node
        # Since its a leaf node, class variable will have only one value
        # Hence, return the value of class variable of first tuple
        return node(ccount=data[0][0])

#-------------------------------------------------------------------------------------------------------------------------------------

# This function is written to classify a particular tuple.
# Inputs:
#     i. observation - the tuple to be classified
#     ii. dt - decision tree to be used
#     iii. d - mac depth allowed
# Output:
#     i. Classified value - 1 or 0
# Example call:
#     i. c=classify([1,1,1,1,1,1,1],dt1,9)
def classify(tuple,dt,d,cnt=0):
    if dt.ccount!=999:
        return dt.ccount
    else:
        v=tuple[dt.col_index]
        b=None
        if int(v)>=int(dt.val_node):
            if d<0:
                return 1 # Classified as TRUE(1)
            else:
                b = dt.t_b
        else:
            if d<0:
                return 0 # Classified as FALSE(0)
            else:
                b = dt.f_b
    cnt+=1
    d -= 1
    return classify(tuple,b,d,cnt)

# -------------------------------------------------------------------------------------------------------------------------------------

# This function is used to classify values in the test dataset by calling the classify function for each not row.
# Once each row is classified, following metrics are calculated:
#     i. Accuracy = Number of tuples correctly classified / Total number of tuples
#     ii. True Positives(tp) = Number of tuples which are actually true and are classified as true
#     ii. True Negatives(tn) = Number of tuples which are actually false and are classified as false
#     ii. False Positives(fp) = Number of tuples which are actually false and are classified as true
#     ii. False Negatives(fn) = Number of tuples which are actually true and are classified as false
# Input
#     i. floc = File location(absolute path, not logical) of test dataset
#     ii. dt = decision tree to be used
#     iii. d = max depth to be used
# Output
#     i. Accuracy
#     ii. True Positives(tp)
#     ii. True Negatives(tn)
#     ii. False Positives(fp)
#     ii. False Negatives(fn)
def classifier(floc,dt,d):
    TestSet=read_data(floc)
    headers=TestSet[0]
    TestPreProcess1=remove_header(TestSet)
    TestPreProcess2=remove_id(TestPreProcess1)
    TestFinal=char_to_int(TestPreProcess2)
    for i in TestFinal:
        if d==0:
            i.append(1)
        else:
            i.append(classify(i,dt,d))
    CorrectClassify=0
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range (0,len(TestFinal)):
        if TestFinal[i][0]==TestFinal[i][len(TestFinal[i])-1]:
            CorrectClassify+=1
            if TestFinal[i][0]==1:  tp+=1
            else:   tn+=1
        else:
            if TestFinal[i][0]==1:  fn+=1
            else:   fp+=1
    return CorrectClassify/len(TestFinal),tp,tn,fp,fn

# Reading the training data set for first deision tree - Monks1-train
DataSet = read_data('C:\\Users\\Atharva\\Desktop\\Monks Data\\train_m1.csv')
DataPreProcess1 = remove_header(DataSet) # Removing headers
DataPreProcess2 = remove_id(DataPreProcess1) # Removing the id column
DataFinal = char_to_int(DataPreProcess2) # Convert all values from type char to type int
dt1 = build(DataFinal) # Call build function to create the decision tree with Monks1-train data

# Reading the training data set for second deision tree - Monks2-train
DataSet = read_data('C:\\Users\\Atharva\\Desktop\\Monks Data\\train_m2.csv')
DataPreProcess1 = remove_header(DataSet) # Removing headers
DataPreProcess2 = remove_id(DataPreProcess1) # Removing the id column
DataFinal = char_to_int(DataPreProcess2)
dt2 = build(DataFinal) # Call build function to create the decision tree with Monks2-train data

# Reading the training data set for third deision tree - Monks3-train
DataSet = read_data('C:\\Users\\Atharva\\Desktop\\Monks Data\\train_m3.csv')
DataPreProcess1 = remove_header(DataSet) # Removing headers
DataPreProcess2 = remove_id(DataPreProcess1) # Removing the id column
DataFinal = char_to_int(DataPreProcess2) # Convert all values from type char to type int
dt3 = build(DataFinal) # Call build function to create the decision tree with Monks3-train data

# Reading the training data set for fourth deision tree - German - Loan defaults
DataSet = read_data('C:\\Users\\Atharva\\Google Drive\\AML\\German2_Train_flt_check.csv')
DataPreProcess1 = remove_header(DataSet) # Removing headers
# No column named "ID" needs to be removed
DataFinal = char_to_int(DataPreProcess1) # Convert all values from type char to type int
dt4 = build(DataFinal) # Call build function to create the decision tree with Monks1-train data

metrics1=[]
metrics2=[]
metrics3=[]
metrics4=[]

# Storing decision trees in an array to be able to iterate on
dt=(dt1,dt2,dt3)

# Calculating accuracy at different depths for Monks-1, Monks-2 and Monks-3
# and storing accuracy metrics in metrics1, metrics2 and metrics3 respectively
for i in range (0,20):
    for j in range(0,3):
        metrics_row=[]
        metrics_row.append(i)
        floc='C:\\Users\\Atharva\\Desktop\\Monks Data\\test_m' + str(j + 1) + '.csv'
        a,tp,tn,fp,fn=classifier(floc,dt[j],i)
        metrics_row.append(a*100)
        metrics_row.append(tp)
        metrics_row.append(tn)
        metrics_row.append(fp)
        metrics_row.append(fn)
        if j==0:
            metrics1.append(metrics_row)
        if j==1:
            metrics2.append(metrics_row)
        if j==2:
            metrics3.append(metrics_row)

# Using classifier function to to classify the data in own dataset and calculating the accuracy metrics
# same is stored in metrics 4
for i in range (0,20):
    metrics_row=[]
    metrics_row.append(i)
    floc = 'C:\\Users\\Atharva\\Google Drive\\AML\\German2_Test_flt_check.csv'
    a, tp, tn, fp, fn = classifier(floc, dt4, i)
    metrics_row.append(a * 100)
    metrics_row.append(tp)
    metrics_row.append(tn)
    metrics_row.append(fp)
    metrics_row.append(fn)
    metrics4.append(metrics_row)

#-------------------------------------------------------------------------------------------------------------------------------------

# Function to print accuracy metrics
def res_print(res):
    for i in res:
        print(" Depth: ",i[0])
        print("   Accuracy:",i[1])
        print("   True Positive:", i[2])
        print("   True Negative:", i[3])
        print("   False Positive:", i[4])
        print("   False negative:", i[5])

# -------------------------------------------------------------------------------------------------------------------------------------

# Printing final output

print("----------------------------------------- Monks1 -----------------------------------------")
res_print(metrics1)

print("----------------------------------------- Monks2 -----------------------------------------")
res_print(metrics2)

print("----------------------------------------- Monks3 -----------------------------------------")
res_print(metrics3)

print("----------------------------------------- Own data -----------------------------------------")
res_print(metrics4)

#-------------------------------------------------------------------------------------------------------------------------------------

# Graph plotting

# Extracting accuracy into another list for plotting
# Error% = 100 - accuracy%
acplot=[]
acplot_c=[]
for i in range(0,len(metrics1)):
    acplot.append((metrics1[i][1] + metrics2[i][1] + metrics3[i][1] + metrics4[i][1]) / 4)
    acplot_c.append(((100-metrics1[i][1]) + (100-metrics2[i][1]) + (100-metrics3[i][1]) + (100-metrics4[i][1])) / 4)

# Extracting accuracy into another list for plotting
# Error% = 100 - accuracy%
acplot1=[]
acplot1_c=[]
for i in range(0,len(metrics1)):
    acplot1.append(metrics1[i][1])
    acplot1_c.append(100-metrics1[i][1])

# Extracting accuracy into another list for plotting
# Error% = 100 - accuracy%
acplot2=[]
acplot2_c=[]
for i in range(0,len(metrics2)):
    acplot2.append(metrics2[i][1])
    acplot2_c.append(100-metrics2[i][1])

# Extracting accuracy into another list for plotting
# Error% = 100 - accuracy%
acplot3=[]
acplot3_c=[]
for i in range(0,len(metrics3)):
    acplot3.append(metrics3[i][1])
    acplot3_c.append(100-metrics3[i][1])

# Extracting accuracy into another list for plotting
# Error% = 100 - accuracy%
acplot4=[]
acplot4_c=[]
for i in range(0,len(metrics4)):
    acplot4.append(metrics4[i][1])
    acplot4_c.append(100-metrics4[i][1])

#-------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.title("Accuracy of Decision Tree across depths")
plt.xlabel("Depth of the decision tree")
plt.ylabel("Accuracy in %")
plt.axis([0,20,0,100])

plt.plot(acplot, label="Average")
plt.draw()

plt.plot(acplot1, label="Monks-1")
plt.draw()

plt.plot(acplot2, label="Monks-2")
plt.draw()

plt.plot(acplot3, label="Monks-3")
plt.draw()

plt.plot(acplot4, label="Own data")
plt.draw()

plt.legend(loc=4)
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt2

plt2.title("Error of Decision Tree across depths")
plt2.xlabel("Depth of the decision tree")
plt2.ylabel("Error in %")
plt2.axis([0,20,0,100])

plt2.plot(acplot_c, label="Average")
plt2.draw()

plt2.plot(acplot1_c, label="Monks-1")
plt2.draw()

plt2.plot(acplot2_c, label="Monks-2")
plt2.draw()

plt2.plot(acplot3_c, label="Monks-3")
plt2.draw()

plt2.plot(acplot4_c, label="Own data")
plt2.draw()

plt2.legend(loc=1)
plt2.show()
