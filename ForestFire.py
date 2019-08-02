from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree

# Spark initialization
conf = SparkConf().setMaster("local").setAppName("ForestFire")
sc = SparkContext(conf = conf)

def splitToKeyValue(str):
	s = str.split(",")
	return  ((int(s[0]), int(s[1]), int(s[2]), int(s[3])), float(s[4]))

# LabeledPoint(label, features)
# features of a point is in the format of [pcp,temp,ws,rhum]
def generatePoint(lst):
	val = lst[1]
	return LabeledPoint(val[1], [val[0][0][0][0], val[0][0][0][1], val[0][0][1], val[0][1]])

# Load data from HDFS
burned = sc.textFile("hdfs:///user/maria_dev/forestfire/burned-area.txt")
pcp = sc.textFile("hdfs:///user/maria_dev/forestfire/precipitation.txt")
temp = sc.textFile("hdfs:///user/maria_dev/forestfire/temperature.txt")
ws = sc.textFile("hdfs:///user/maria_dev/forestfire/wind-speed.txt")
rhum = sc.textFile("hdfs:///user/maria_dev/forestfire/humidity.txt")

# Transform data into KeyValues, Key = (lat,lon,month,year)
iburned = burned.map(lambda x:splitToKeyValue(x))
ipcp = pcp.map(lambda x:splitToKeyValue(x))
itemp = temp.map(lambda x:splitToKeyValue(x))
iws = ws.map(lambda x:splitToKeyValue(x))
irhum = rhum.map(lambda x:splitToKeyValue(x))

# A very expensive pre-processing to combine all values based on their keys
cdata = ipcp.join(itemp).join(iws).join(irhum).join(iburned)
cdata.cache()

# Generate data points for mllib
pdata = cdata.map(lambda x:generatePoint(x))

(trainData, testData) = pdata.randomSplit([0.7, 0.3])

# Train the training dataset
model = DecisionTree.trainRegressor(trainData, categoricalFeaturesInfo={}, impurity='variance', maxDepth=5, maxBins=32)

# Make predictions
test = testData.map(lambda x: x.features)
pred = model.predict(test)

# Compute error
comb = testData.map(lambda x: x.label).zip(pred)
mse = comb.map(lambda x: (x[0] - x[1]) * (x[0] - x[1])).sum() / float(testData.count())

print 'Mean Squared Error = ' + str(mse)
print 'Regression tree model :'
print model.toDebugString()

