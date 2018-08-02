from sklearn import svm
import random
'''
Used to read in X matrix (data set).
Data saved locally (contains data that may/may not be publically shared : from Amazon Turk)
'''
with open("./spambase/spambase.data", 'r') as fobj:
    X = [[float(num) for num in line.split(',')] for line in fobj]

random.shuffle(X)

'''
Used to read in Y vector (attribute names).
Attributes are the features that we are looking at from each line 
'''
y = []
for row in X:
    y.append(row[len(row)-1])


'''
We're going to stratify X in containing only word freq, char freq, capital freq
and compare the different accuracy rates and false positivies using these prediction
models.
'''
#Obtaining only word attributes
wordX = []
for row in X:
    wordX.append(row[:48])

#Training classifier on first 4000 samples (using Support Vector Model)
clfWord = svm.SVC()
clfWord.fit(wordX[0:4000], y[0:4000])
testWordX = clfWord.predict(wordX[4000:])

#Obtaining prediction accuracy and false positive rate (Does the predicted match the actual?)
countWord = 0
countWordFalse = 0
for i in range(len(testWordX)):
    if testWordX[i] == y[4000+i]:
        countWord = countWord + 1
    if testWordX[i] == 1 and y[4000+i] == 0:
        countWordFalse = countWordFalse + 1

print("Words:")
print("Percent spam = " + str(countWord/len(y[4000:])))
print("Percent false positive = " + str(countWordFalse/len(y[4000:])))
print()
#-------------------------------------------------------
#Trying out different combinations of features to see which yields best results (most accuracy, low false positive rate)
#Obtaining only char attributes for training 
charX = []
for row in X:
    charX.append(row[48:54])

#Training classifier on first 4000 samples
clfChar = svm.SVC()
clfChar.fit(charX[0:4000], y[0:4000])
testCharX = clfChar.predict(charX[4000:])

#Obtaining prediction accuracy and false positive rate
countChar = 0
countCharFalse = 0
for i in range(len(testCharX)):
    if testCharX[i] == y[4000+i]:
        countChar = countChar + 1
    if testCharX[i] == 1 and y[4000+i] == 0:
        countCharFalse = countCharFalse + 1
print("Chars:")
print("Percent spam = " + str(countChar/len(y[4000:])))
print("Percent false positive = " + str(countCharFalse/len(y[4000:])))
print()

#-----------------------------------------------------
#Obtaining only capitalization attributes
capX = []
for row in X:
    capX.append(row[54:])

#Training classifier on first 4000 samples
clfCap = svm.SVC()
clfCap.fit(capX[0:4000], y[0:4000])
testCapX = clfCap.predict(capX[4000:])

#Obtaining prediction accuracy and false positive rate
countCap = 0
countCapFalse = 0
for i in range(len(testCapX)):
    if testCapX[i] == y[4000+i]:
        countCap = countCap + 1
    if testCapX[i] == 1 and y[4000+i] == 0:
        countCapFalse = countCapFalse + 1
print("Capital:")
print("Percent spam = " + str(countCap/len(y[4000:])))
print("Percent false positive = " + str(countCapFalse/len(y[4000:])))
print()

#-----------------------------------------------------
#Obtaining word and char attributes
wordCharX = []
for row in X:
    wordCharX.append(row[:54])

#Training classifier on first 4000 samples
clfWordChar = svm.SVC()
clfWordChar.fit(wordCharX[0:4000], y[0:4000])
testWordCharX = clfWordChar.predict(wordCharX[4000:])

#Obtaining prediction accuracy and false positive rate
countWordChar = 0
countWordCharFalse = 0
for i in range(len(testWordCharX)):
    if testWordCharX[i] == y[4000+i]:
        countWordChar = countWordChar + 1
    if testWordCharX[i] == 1 and y[4000+i] == 0:
        countWordCharFalse = countWordCharFalse + 1
print("Words and characters:")
print("Percent spam = " + str(countWordChar/len(y[4000:])))
print("Percent false positive = " + str(countWordCharFalse/len(y[4000:])))
print()

#-----------------------------------------------------
#Obtaining word and cap attributes
wordCapX = []
for row in X:
    wordCapX.append(row[:48] + row[54:])

#Training classifier on first 4000 samples
clfWordCap = svm.SVC()
clfWordCap.fit(wordCapX[0:4000], y[0:4000])
testWordCapX = clfWordCap.predict(wordCapX[4000:])

#Obtaining prediction accuracy and false positive rate
countWordCap = 0
countWordCapFalse = 0
for i in range(len(testWordCapX)):
    if testWordCapX[i] == y[4000+i]:
        countWordCap = countWordCap + 1
    if testWordCapX[i] == 1 and y[4000+i] == 0:
        countWordCapFalse = countWordCapFalse + 1
print("Words and capitals:")
print("Percent spam = " + str(countWordCap/len(y[4000:])))
print("Percent false positive = " + str(countWordCapFalse/len(y[4000:])))
print()

#-----------------------------------------------------
#Obtaining char and cap attributes
charCapX = []
for row in X:
    charCapX.append(row[48:])

#Training classifier on first 4000 samples
clfCharCap = svm.SVC()
clfCharCap.fit(charCapX[0:4000], y[0:4000])
testCharCapX = clfCharCap.predict(charCapX[4000:])

#Obtaining prediction accuracy and false positive rate
countCharCap = 0
countCharCapFalse = 0
for i in range(len(testCharCapX)):
    if testCharCapX[i] == y[4000+i]:
        countCharCap = countCharCap + 1
    if testCharCapX[i] == 1 and y[4000+i] == 0:
        countCharCapFalse = countCharCapFalse + 1
print("Characters and capitals:")
print("Percent spam = " + str(countCharCap/len(y[4000:])))
print("Percent false positive = " + str(countCharCapFalse/len(y[4000:])))
print()

#-----------------------------------------------------
#All attributes combined 
#Training classifier on first 4000 samples
clf = svm.SVC()
clf.fit(X[0:4000], y[0:4000])
testX = clf.predict(X[4000:])

#Obtaining prediction accuracy and false positive rate
count = 0
countFalse = 0
for i in range(len(testX)):
    if testX[i] == y[4000+i]:
        count = count + 1
    if testX[i] == 1 and y[4000+i] == 0:
        countFalse = countFalse + 1
print("Total:")
print("Percent spam = " + str(count/len(y[4000:])))
print("Percent false positive = " + str(countFalse/len(y[4000:])))
print()
