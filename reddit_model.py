from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
import cleantext
import os
from pyspark.sql.types import ArrayType, StringType, BooleanType, DoubleType, IntegerType
from pyspark.ml.feature import CountVectorizer
# Bunch of imports (may need more)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def clean_wrapper(body):
    parsed_text, unigrams, bigrams, trigrams = cleantext.sanitize(body)
    unigrams = unigrams.split(" ")
    bigrams = bigrams.split(" ")
    trigrams = trigrams.split(" ")
    ret = unigrams
    ret.extend(bigrams)
    ret.extend(trigrams)
    return ret

def checkState(state):
    states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
    return state in states

def positiveUDF(titties):
    if titties[1] > 0.2:
        return 1
    else:
        return 0

def negativeUDF(titties):
    if titties[1] > 0.25:
        return 1
    else:
        return 0

def main(context):
    """Main function takes a Spark SQL context."""
    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED
    
    # TASK 1
    # Load the data into PySpark.
    
    # For the comments:
    if not os.path.exists("./comments.parquet"):
        comments = context.read.json("comments-minimal.json.bz2")
        comments.write.parquet("comments.parquet")

    # For the submissions:
    if not os.path.exists("./submissions.parquet"):
        submissions = context.read.json("submissions.json.bz2")
        submissions.write.parquet("submissions.parquet")
    #submissions.printSchema()

    # For labelled data:
    if not os.path.exists("./labels.parquet"):
        labels = context.read.format('csv').options(header='true', inferSchema='true').load("labeled_data.csv")
        labels.write.parquet("labels.parquet")
    
    
    # TASK 2
    # Code for Task 2...
    # For task 2, we will join the labels and comments
    
    commentsParquet = context.read.parquet("comments.parquet")
    commentsParquet.createOrReplaceTempView("comments")

    labelsParquet = context.read.parquet("labels.parquet")
    labelsParquet.createOrReplaceTempView("labels")
    
    # Now, compute the join:
    if not os.path.exists("./joinedComments.parquet"):
        joinedComments = context.sql("SELECT labels.Input_id, labels.labeldem, labels.labelgop, labels.labeldjt, body FROM comments JOIN labels on id=Input_id")
        joinedComments.write.parquet("joinedComments.parquet")
    joinedComments = context.read.parquet("joinedComments.parquet")
    joinedComments.createOrReplaceTempView("joinedComments")
    #joinedComments.printSchema()
    
    # TASK 3
    # NOT NEEDED 

    # TASK 4
    # Register the user defined function
    context.registerFunction("sanitize", clean_wrapper, ArrayType(StringType()))

    # TASK 5
    if not os.path.exists("./santized.parquet"):
        sanitizedText = context.sql("SELECT Input_id, labeldem, labelgop, labeldjt, sanitize(body) as body FROM joinedComments")
        sanitizedText.write.parquet("sanitized.parquet")
    


    # TASK 6A
    sanitizedText = context.read.parquet("sanitized.parquet")
    sanitizedText.createOrReplaceTempView("sanitizedText")
    cv = CountVectorizer(inputCol="body", outputCol="features", minDF=10.0, binary=True)
    fitted = cv.fit(sanitizedText)
    vector = fitted.transform(sanitizedText)
    # TASK 6B
    vector.createOrReplaceTempView("vector")
    pos = context.sql("SELECT *, if(labeldjt=1, 1, 0) AS label FROM vector")
    neg = context.sql("SELECT *, if(labeldjt=-1, 1, 0) AS label FROM vector")


    # TASK 7
        # Initialize two logistic regression models.
    # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
    poslr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
    neglr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
    # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
    posEvaluator = BinaryClassificationEvaluator()
    negEvaluator = BinaryClassificationEvaluator()
    # There are a few parameters associated with logistic regression. We do not know what they are a priori.
    # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
    # We will assume the parameter is 1.0. Grid search takes forever.
    posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
    negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
    # We initialize a 5 fold cross-validation pipeline.
    posCrossval = CrossValidator(
        estimator=poslr,
        evaluator=posEvaluator,
        estimatorParamMaps=posParamGrid,
        numFolds=5)
    negCrossval = CrossValidator(
        estimator=neglr,
        evaluator=negEvaluator,
        estimatorParamMaps=negParamGrid,
        numFolds=5)
    # Although crossvalidation creates its own train/test sets for
    # tuning, we still need a labeled test set, because it is not
    # accessible from the crossvalidator (argh!)
    # Split the data 50/50
    posTrain, posTest = pos.randomSplit([0.5, 0.5])
    negTrain, negTest = neg.randomSplit([0.5, 0.5])
    # Train the models
    print("Training positive classifier...")
    posModel = posCrossval.fit(posTrain)
    print("Training negative classifier...")
    negModel = negCrossval.fit(negTrain)

    # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    posModel.save("project2/pos.model")
    negModel.save("project2/neg.model")
    


    # TASK 8 and TASK 9
    # Create the submissions and comments tables from the parquets:
    if not os.path.exists("sanitizedJoinedData.parquet"):
        submissions = context.read.parquet("submissions.parquet")
        submissions.createOrReplaceTempView("submissions")

        comments = context.read.parquet("comments.parquet")
        comments.createOrReplaceTempView("comments")
        comments = comments.sample(False, 0.2, None)
        joinedData = context.sql("SELECT comments.link_id AS id, comments.body, comments.created_utc, submissions.title, comments.author_flair_text, submissions.score AS submission_score, comments.score as comments_score FROM comments JOIN submissions ON REPLACE(comments.link_id, 't3_', '')=submissions.id AND comments.body NOT LIKE '%/s%' AND comments.body NOT LIKE '&gt%'")
        #joinedData.show(joinedData.count(), False)
        #print(str(joinedData.count()))

        # Repeating earlier tasks: Tasks 4 and 5
        joinedData.createOrReplaceTempView("joinedData")
        # Re-register temporary function since we are forced to:
        context.registerFunction("sanitize", clean_wrapper, ArrayType(StringType()))
        print("writing sanitized parquet now")
        sanitizedJoinedData = context.sql("SELECT id, created_utc, title, author_flair_text, submission_score, comments_score, sanitize(body) AS body FROM joinedData")
        sanitizedJoinedData.write.parquet("sanitizedJoinedData.parquet")
    
    
    sanitizedJoinedData = context.read.parquet("sanitizedJoinedData.parquet")
    sanitizedJoinedData = sanitizedJoinedData.sample(False, 0.2, None)
    cv = CountVectorizer(inputCol="body", outputCol="features", minDF=10.0, binary=True)
    newVector = fitted.transform(sanitizedJoinedData)

    seenPosModel = CrossValidatorModel.load("project2/pos.model")
    seenNegModel = CrossValidatorModel.load("project2/neg.model")

    posResult = seenPosModel.transform(newVector)
    posResult = posResult.selectExpr("id", "created_utc", "title", "author_flair_text", "submission_score", "comments_score", "body", "features", "probability as positive_probability")

    
    cumResult = seenNegModel.transform(posResult)
    cumResult = cumResult.selectExpr("id", "created_utc", "title", "author_flair_text", "submission_score", "comments_score", "body", "features", "positive_probability", "probability as negative_probability")


    cumResult.createOrReplaceTempView("cumResult")

    context.registerFunction("positiveFunc", positiveUDF, IntegerType())
    context.registerFunction("negativeFunc", negativeUDF, IntegerType())
    cumResult = context.sql("SELECT id, created_utc, title, author_flair_text, submission_score, comments_score, body, features, positiveFunc(positive_probability) AS positive_probability,negativeFunc(negative_probability) AS negative_probability FROM cumResult")
    cumResult.write.parquet("cumResult.parquet")

    
    # TASK 10

    cumResult = context.read.parquet("cumResult.parquet")
    cumResult.createOrReplaceTempView("cumResult")
    # Actual 10.2
    
    task10_6 = context.sql("SELECT DATE(FROM_UNIXTIME(created_utc)) AS date_created, SUM(positive_probability)/COUNT(positive_probability) AS pos, SUM(negative_probability)/COUNT(negative_probability) AS neg FROM cumResult GROUP BY date_created ORDER BY date_created")
    task10_6.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("task10_6.csv")
    
    # Top 10 posts:

    if not os.path.exists("./task10_top_pos.csv"):
        task10_top_pos = cumResult.groupBy('title')\
            .agg(
                 (F.sum('positive_probability') / F.count(F.lit(1))).alias('pct_pos'),
                 F.count(F.lit(1)).alias('count')
                 )\
                .orderBy(F.desc('pct_pos'), F.desc('count')).limit(10)\
                .select('title', 'pct_pos')
        task10_top_pos.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("task10_top_pos.csv")
    if not os.path.exists("./task10_top_neg.csv"):
        task10_top_neg = cumResult.groupBy('title')\
            .agg(
                 (F.sum('negative_probability') / F.count(F.lit(1))).alias('pct_neg'),
                 F.count(F.lit(1)).alias('count')
                 )\
                .orderBy(F.desc('pct_neg'), F.desc('count')).limit(10)\
                .select('title', 'pct_neg')
        task10_top_neg.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("task10_top_neg.csv")


    # 10.1
    # Get the number of records
    totalRows = cumResult.count()
    # Calculate percentages
    task10_1 = context.sql("SELECT SUM(positive_probability)/ {0} AS pos, SUM(negative_probability)/{1} AS neg FROM cumResult".format(totalRows, totalRows))

    # 10.2
    task10_2 = context.sql("SELECT DAYOFWEEK(FROM_UNIXTIME(created_utc)) AS date_created, SUM(positive_probability)/COUNT(positive_probability) AS pos, SUM(negative_probability)/COUNT(negative_probability) AS neg FROM cumResult GROUP BY date_created")
     
    # 10.3
    context.registerFunction("checkStateWrapper", checkState, BooleanType())
    task10_3 = context.sql("SELECT author_flair_text AS state, SUM(positive_probability)/COUNT(positive_probability) AS pos, SUM(negative_probability)/COUNT(negative_probability) AS neg FROM cumResult WHERE(checkStateWrapper(author_flair_text)) GROUP BY author_flair_text")

    # 10.4
    task10_4 = context.sql("SELECT comments_score, SUM(positive_probability)/COUNT(positive_probability) AS pos, SUM(negative_probability)/ COUNT(negative_probability) AS neg FROM cumResult GROUP BY comments_score")
    task10_5 = context.sql("SELECT submission_score, SUM(positive_probability)/COUNT(positive_probability) AS pos, SUM(negative_probability)/ COUNT(negative_probability) AS neg FROM cumResult GROUP BY submission_score")
#    cumResult.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("cumResults.csv")
    task10_1.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("task10_1.csv")
    task10_2.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("task10_2.csv")
    task10_3.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("task10_3.csv")
    task10_4.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("task10_4.csv")
    task10_5.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("task10_5.csv")
    
if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    comments = sqlContext.read.json("comments-minimal.json.bz2")
    submissions = sqlContext.read.json("submissions.json.bz2")
    main(sqlContext)
