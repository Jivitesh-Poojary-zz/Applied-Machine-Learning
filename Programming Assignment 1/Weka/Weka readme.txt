Readme file for Weka

Team partners:
Atharva Tere (agtere@iu.edu) and jivitesh Poojary(jpoojary@iu.edu)

Steps followed:

Monks1, Monks2, Monks3:
1. Convert the class variable to nominal in excel by substituting 1=Yes and 0=No
2. Load the train set data in Weka (preprocess tab)
3. Load the test set data in Weka (classify tab)
4. Choose the appropriate class variable 
5. Choose the J48 algorithm with default values and go!

Own data set:
0. Use the RemovePercent filter in Weka to split the data set
1. Convert the class variable to nominal in excel by substituting 1=Yes and 0=No
2. Load the train set data in Weka (preprocess tab)
3. Load the test set data in Weka (classify tab)
4. Choose the appropriate class variable 
5. Choose the J48 algorithm with default values and go!

Note:
I have on purpose not used the NumericToNominal function in weka for following reasons:
1. It converts all the attribute from numeric to nominal instead of converting any one specified attribute
	a. Because of this, it reduces the accuracy of the decision tree since only a==b comaprisons can be made instead of a<=b or a>=b
	b. If any attribute in the test dataset has a value which is not present in training data set, it throws an error [again, because a<=b or a>=b type of comparisons are not allowed]
