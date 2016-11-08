import matplotlib.pyplot as plot
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression

data = np.genfromtxt("census_train.csv", dtype=None, usecols=(0), skip_header=1, delimiter=",")
plot.hist(data)
plot.title("census_train Age histogram")
plot.xlabel("Age")
plot.ylabel("Amount")
"""plot.show()"""

files = [("census_train.csv", "modified_census_train.csv"), ("census_test.csv", "modified_census_test.csv")]
for inputFileName, outputFileName in files:
    with open(inputFileName, "rb") as inputFile, open(outputFileName, "wb") as outputFile:
        reader = csv.reader(inputFile)
        next(reader, None)
        writer = csv.writer(outputFile)
        for row in reader:
            writer.writerow((row[0], row[1], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14]))

trainingDataset = np.genfromtxt("modified_census_train.csv", dtype=None, delimiter=",")
xTrain = trainingDataset[:, 1:]
yTrain = trainingDataset[:, 0]
model = LogisticRegression()
model = model.fit(xTrain, yTrain)

testDataset = np.genfromtxt("modified_census_test.csv", dtype=None, delimiter=",")
xTest = testDataset[:, 1:]
yTest = testDataset[:, 0]
score = model.score(xTest, yTest)
print "Model accuracy:"
print score