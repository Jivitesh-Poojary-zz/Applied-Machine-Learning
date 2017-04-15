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