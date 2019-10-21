import os
import sys
import shutil
from pyspark.sql import SQLContext
from pyspark.conf import SparkConf
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import *

try:
    shutil.rmtree("results")
except OSError as e:
    print ("%s does not exist - %s." % (e.filename, e.strerror))

def sortLettersUDF(letters):
    return ''.join(sorted(letters))

def getKeyLettersUDF(letters, positions):
    keyLetters = ''
    for p in positions:
        keyLetters += letters[p]
    return keyLetters

def myConcat(*cols):
    return F.concat(*[F.coalesce(c, F.lit("")) for c in cols])

def validWordUDF(answerLetters,colID):
    for c in colID:
      if c in answerLetters:
        answerLetters = answerLetters.replace(c,"", 1)
      else:
        return False
    return True

def updateLettersUDF(answerLetters,colID):
    for c in colID:
      if c in answerLetters:
        answerLetters = answerLetters.replace(c,"", 1)
      else:
        return "NOT_VALID"
    return answerLetters

#Setup spark context with crossJoin enabled, since I use crossJoin to get all possible matches
sc.stop()
conf = SparkConf().setAll([('spark.sql.crossJoin.enabled', 'true')])
sc = SparkContext(conf=conf).getOrCreate("MyApps")
sqlContext = SQLContext(sc)

validWord_udf = F.udf(validWordUDF, StringType())
updateLetters_udf = F.udf(updateLettersUDF, StringType())
sortLetters_udf = F.udf(sortLettersUDF, StringType())
sqlContext.udf.register("getKeyLetters_udf", getKeyLettersUDF)

#load input puzzle and create temp table
df = sqlContext.read.json("inputs/puzzleTest1.json")
df.withColumn("sortedLetters", sortLetters_udf(df["letters"])) \
  .registerTempTable("input")
  
#load dictionary as csv to reduce proccessing required to pivot json fields
dictSchema=[StructField('colID',StringType(),True),
       StructField('colValue',IntegerType(),True)]
finalStruct=StructType(fields=dictSchema)
df2 = sqlContext.read.csv(path='inputs/freq_dict.csv',header=True,schema=finalStruct,ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)

newDf = df2\
  .withColumn("sortedID",sortLetters_udf(df2["colID"]))\
  .withColumn("wordLen", F.length('colID'))
newDf.registerTempTable("dictionary")

#join on sorted characters to get unscrambled possibilities
sqlContext.sql("""select puzzle_id,
                         letters, 
                         keyPositions, 
                         colID, 
                         colValue, 
                         getKeyLetters_udf(colID,keyPositions) as keyLetters, 
                         answerLengths 
                  from input i inner join dictionary d on d.sortedID = i.sortedLetters
""").registerTempTable("unscrambled")
#aggregate to posibilities into list
unscrambled = sqlContext.sql("""select puzzle_id, 
                                       letters, 
                                       collect_list(keyLetters) as keyLetters, 
                                       answerLengths 
                                from unscrambled 
                                group by puzzle_id,letters,answerLengths
""")

#explode posibilities to get list of key letters from each possible combination of possible unscrambled words
letters_pivoted = unscrambled.groupby('puzzle_id','answerLengths').pivot('letters').agg(F.max("keyLetters"))

n=2
letters_exploded = letters_pivoted
for col in [c for c in letters_pivoted.columns if c not in {"puzzle_id","answerLengths"}]:
  letters_exploded = letters_exploded\
    .withColumn("col"+str(n), F.explode_outer(letters_exploded.columns[n]))
  n += 1

#for each final answer word, get all possible valid unscrambled words then remove those letters for next segment check. Repeat twice.
#validate/filter at each step
final_letters = letters_exploded\
  .withColumn("answerLetters",myConcat(*[c for c in letters_exploded.columns if c.startswith("col")]))\
  .select("puzzle_id","answerLetters","answerLengths")\
  .withColumn("answerLength1",F.col("answerLengths")[0])\
  .withColumn("answerLength2",F.col("answerLengths")[1])\
  .withColumn("answerLength3",F.col("answerLengths")[2])

words = final_letters.alias('a').join(newDf.alias('b'),F.col('b.wordLen') == F.col('a.answerLength1'))
words = words \
  .withColumn("valid",validWord_udf(words["answerLetters"],words["b.colID"]))\
  .withColumn("remainingLetters",updateLetters_udf(words["answerLetters"],words["b.colID"]))\
  .filter("valid=True")\
  .drop("sortedID","wordLen","valid")\
  .withColumnRenamed("colID", "colIDb")\
  .withColumnRenamed("colValue", "colValueb")

words = words.alias('a').join(newDf.alias('c'),F.col('c.wordLen') == F.col('a.answerLength2'))
words = words\
  .withColumn("valid",validWord_udf(words["remainingLetters"],words["c.colID"]))\
  .withColumn("remainingLetters2",updateLetters_udf(words["remainingLetters"],words["c.colID"]))\
  .filter("valid=True")\
  .drop("sortedID","wordLen","valid","remainingLetters")\
  .withColumnRenamed("colID", "colIDc")\
  .withColumnRenamed("colValue", "colValuec")

words = words.alias('a').join(newDf.alias('d'),F.col('d.wordLen') == F.col('a.answerLength3'))
final_words = words\
  .withColumn("valid",validWord_udf(words["remainingLetters2"],words["d.colID"]))\
  .filter("valid=True")\
  .withColumn("answerWords",myConcat(F.col("colIDb"),F.col("colIDc"),F.col("d.colID")))\
  .withColumn("valid2",validWord_udf(words["answerLetters"],F.col("answerWords")))\
  .filter("valid2=True")\
  .select([F.col('a.puzzle_id'),F.col('a.answerLetters'),F.col('answerWords'),F.col('colValueb'),F.col('colValuec'),F.col('d.colValue').alias("colValued")])

#sum word frequencies. Lowest number is most likely
final_words\
  .withColumn("freq_sum",F.when(F.col("colValueb") == 0, 99999).otherwise(F.col("colValueb"))+\
              F.when(F.col("colValuec") == 0, 99999).otherwise(F.col("colValuec"))+\
              F.when(F.col("colValued") == 0, 99999).otherwise(F.col("colValued")))\
  .drop("answerLetters","colValueb","colValuec","colValued","valid")\
  .orderBy('freq_sum', ascending=True)\
  .write.format('json').save('results')
