import com.johnsnowlabs.nlp.Finisher
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{MinHashLSH, _}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, _}
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.types._
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

object unsupervised_minhash {
  def main(args: Array[String]): Unit = {

    // Create the spark session first
    val ss = SparkSession.builder().master("local").appName("minhashApp").getOrCreate()
    import ss.implicits._
    // For implicit conversions like converting RDDs to DataFrames
    val currentDir = System.getProperty("user.dir") // get the current directory



    println("reading from input files...")
    println

    // Read the contents of the csv file in a dataframe
    val trainDF = ss.read.format("csv").option("header", "true").load("train.csv")
    trainDF.printSchema()
    val descriptionsDF = ss.read.format("csv").option("header", "true").load("product_descriptions.csv")
    descriptionsDF.printSchema()
    val attributesDF = ss.read.format("csv").option("header", "true").load("attributes.csv")
    attributesDF.printSchema()

    val newAttributesDF = attributesDF.filter(attributesDF("name") === "MFG Brand Name")
    val newNewAttributesDF = newAttributesDF.select("product_uid", "value")

    val consolidated = trainDF.join(descriptionsDF, "product_uid").join(newNewAttributesDF, "product_uid")
      /*select columns for df*/
      .select(trainDF("product_uid"), trainDF("product_title"), trainDF("search_term"),
      trainDF("relevance"), descriptionsDF("product_description"), newNewAttributesDF("value"))
      /*turn all columns in lower caps*/
      .withColumn("product_description", lower(col("product_description"))).withColumn("product_title", lower(col("product_title")))
      .withColumn("search_term", lower(col("search_term"))).withColumn("value", lower(col("value")))

    val consolidated_with_index = consolidated.withColumn("index", monotonically_increasing_id())

    /*clear unused dfs*/
    trainDF.unpersist()
    descriptionsDF.unpersist()
    attributesDF.unpersist()
    newAttributesDF.unpersist()
    newNewAttributesDF.unpersist()
    consolidated.unpersist()

    /*removeSpecials function*/
    def removeSpecials: String => String =
    /*replacing special characters*/
      _.replaceAll("""[\p{Punct}&&[^-]]""", "")
        //.replaceAll("\\$|\\?\\+\\-\\-\\[\\&\\<\\>\\)\\(\\_\\,\\;\\:\\!\\^\\~\\@\\#\\*\\.", " ")
        .replaceAll("&nbsp", " ")
        .replaceAll("nbsp", "")
        .replaceAll("&amp", "&")
        //.replaceAll("&#39;", "'")
        .replaceAll("/>/Agt/", "")
        .replaceAll("</a<gt/", "")
        .replaceAll("gt/>", "")
        .replaceAll("/>", "")
        .replaceAll("<br", "")
        //.replaceAll("[']","")
        //.replaceAll("[\"]", "")
        /*replacing similar metric words*/
        //.replaceAll("inches|inch", "in")
        .replaceAll("foot|feet|ft", "ft")
        .replaceAll("pounds|pound|lbs|lb", "lb")
        //.replaceAll("square|sq", "sq")
        //.replaceAll("cubic|cu", "cu")
        .replaceAll("gallons|gallon", "gal")
        .replaceAll("ounces|ounce", "oz")
        .replaceAll("centimeters|centimeter", "cm")
        .replaceAll("milimeters|milimeter", "mm")
        .replaceAll("degrees|degree|°", "deg")
        .replaceAll("volts", "volt")
        .replaceAll("wattage|watts", "watt")
        .replaceAll("ampere|amps|amperes", "amp")
        .replaceAll("qquart|quart", "qt")
        .replaceAll("gallons per minute|gallon per minute|gal per minute|gallons/min|gallon/min", "gal per min")
        .replaceAll("gallons per hour|gallon per hour|gal per hour|gallons/hour|gallon/hour", "gal per hr")
        .replaceAll("hrs|hrs.|hours|hour", "hr")
        .replaceAll("mins|minutes|minute", "min")

    /*calling removeSpecials*/
    val udf_removeSpecials = udf(removeSpecials)
    val consolidatedRemovedSpecials = consolidated_with_index.withColumn("product_description", udf_removeSpecials($"product_description")).withColumn("product_title", udf_removeSpecials($"product_title"))
      .withColumn("search_term", udf_removeSpecials($"search_term")).withColumn("value", udf_removeSpecials($"value"))
    //consolidated.unpersist()
    //consolidatedRemovedSpecials.show()

    val titleDF = consolidatedRemovedSpecials.select("index", "product_title", "relevance").withColumnRenamed("product_title", "original_text")
    val queryDF = consolidatedRemovedSpecials.select("index", "search_term", "relevance").withColumnRenamed("search_term", "original_text")

    /* tokenization */
    val tokenizer = new Tokenizer().setInputCol("original_text").setOutputCol("tokenized_text")
    val tokenizedTitle = tokenizer.transform(titleDF)
    val tokenizedQuery = tokenizer.transform(queryDF)

    /* stop word removal */
    val stop_words_remover = new StopWordsRemover()
      .setInputCol("tokenized_text")
      .setOutputCol("filtered_text")

    val joinSeq = udf { (words: Seq[String]) => words.mkString(" ") }

    val removedStopwordsTitle = stop_words_remover.transform(tokenizedTitle)
    val removedStopwordsQuery = stop_words_remover.transform(tokenizedQuery)

    val removedStopwordsTitleJoinedSeq = removedStopwordsTitle.withColumn("filtered_text", joinSeq($"filtered_text")).drop("tokenized_text")
    val removedStopwordsQueryJoinedSeq = removedStopwordsQuery.withColumn("filtered_text", joinSeq($"filtered_text")).drop("tokenized_text")

    /* Spark-NLP */

    val finisher = new Finisher().setInputCols("stem")

    val explainPipelineModel = PretrainedPipeline("explain_document_ml").model

    val pipeline = new Pipeline().
      setStages(Array(
        explainPipelineModel,
        finisher
      ))

    val dataTitle = removedStopwordsTitleJoinedSeq.withColumnRenamed("filtered_text", "text")
    val dataQuery = removedStopwordsQueryJoinedSeq.withColumnRenamed("filtered_text", "text")

    val modelTitle = pipeline.fit(dataTitle)
    val modelQuery = pipeline.fit(dataQuery)

    val annotations_title_df = modelTitle.transform(dataTitle)
    val annotations_query_df = modelQuery.transform(dataQuery)

    // Word count to vector for each wiki content
    val vocabSize = 1000000
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("finished_stem").setOutputCol("features").setVocabSize(vocabSize).setMinDF(5).fit(annotations_title_df)

    val isNoneZeroVector = udf({v: Vector => v.numNonzeros > 0}, DataTypes.BooleanType)

    cvModel.save("cvmodel")
    val loadedModel = CountVectorizerModel.load("cvmodel")

    val vectorizedTitleDF = loadedModel.transform(annotations_title_df).filter(isNoneZeroVector(col("features")))
    val vectorizedQueryDF = loadedModel.transform(annotations_query_df).filter(isNoneZeroVector(col("features")))

    /* convert relevance to binary */
    val binaryConverter = udf((relevance: Double) => {
      if (relevance >= 1.5)
        1
      else
        0
    })

    val vectorizedTitleDFJoinedSeq = vectorizedTitleDF.withColumn("finished_stem", joinSeq($"finished_stem")).drop("text")
      .withColumn("relevance", binaryConverter($"relevance"))
    val vectorizedQueryDFJoinedSeq = vectorizedQueryDF.withColumn("finished_stem", joinSeq($"finished_stem")).drop("text")
      .withColumn("relevance", binaryConverter($"relevance"))

    vectorizedTitleDFJoinedSeq.show()
    vectorizedQueryDFJoinedSeq.show()

    vectorizedTitleDFJoinedSeq.write.parquet("vectorizedTitleDFJoinedSeq.parquet")
    vectorizedQueryDFJoinedSeq.write.parquet("vectorizedQueryDFJoinedSeq.parquet")

    vectorizedTitleDFJoinedSeq.rdd.isEmpty()
    vectorizedQueryDFJoinedSeq.rdd.isEmpty()


    val newVectorizedTitleDFJoinedSeq = ss.read.parquet("vectorizedTitleDFJoinedSeq.parquet").orderBy(desc("index")).limit(500)
    val newVectorizedQueryDFJoinedSeq = ss.read.parquet("vectorizedQueryDFJoinedSeq.parquet").orderBy(desc("index")).limit(500)

    newVectorizedTitleDFJoinedSeq.show()
    newVectorizedQueryDFJoinedSeq.show()

    val mh = new MinHashLSH()
      .setNumHashTables(5)
      .setInputCol("features")
      .setOutputCol("hashes")

    val model = mh.fit(newVectorizedTitleDFJoinedSeq)

    // Feature Transformation
    val dataset1_LSH = model.transform(newVectorizedTitleDFJoinedSeq)
    val dataset2_LSH = model.transform(newVectorizedQueryDFJoinedSeq)

    // Compute the locality sensitive hashes for the input rows, then perform approximate
    // similarity join.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxSimilarityJoin(transformedA, transformedB, 0.6)`
    println("Approximately joining dfA and dfB on Jaccard distance smaller than 1:")
    val jaccardDF = model.approxSimilarityJoin(dataset1_LSH, dataset2_LSH, 1, "JaccardDistance")
      .select(col("datasetA.index").alias("indexA"),
        col("datasetB.index").alias("indexB"), col("datasetA.original_text").alias("textA"),
        col("datasetB.original_text").alias("textB"), col("datasetA.relevance").alias("relevance"),
        col("JaccardDistance"))

    jaccardDF.show(100)

    //jaccardDF.dropDuplicates("indexA", "indexB").show(200, false)

    //val newJaccardDF = jaccardDF.join(jaccardDF.groupBy("indexA", "indexB").count().where("count = 1").drop("count"), Seq("indexA", "indexB"), "left_anti").show()

    /* jaccard threshold */
    val jaccardPredictor = udf((jaccard: Double) => {
      if (jaccard > 0.95)
        0
      else
        1
    })

    val predictionsDF = jaccardDF.withColumn("JaccardDistance", jaccardPredictor($"JaccardDistance"))
      .withColumnRenamed("JaccardDistance", "prediction").filter(col("indexA") === col("indexB"))

    predictionsDF.show(500, false)

    val f1valueTP = udf((pred: Double, rel: Double) => {
      if( pred  == rel && pred == 1.0){
        1.0
      }
      else{
        0.0
      }
    })
    val f1valueTN = udf((pred: Double, rel: Double) => {
      if( pred  == rel && pred == 0.0){
        1.0
      }
      else{
        0.0
      }
    })
    val f1valueFP = udf((pred: Double, rel: Double) => {
      if( pred  != rel && pred == 1.0 && rel == 0.0){
        1.0
      }
      else{
        0.0
      }
    })
    val f1valueFN = udf((pred: Double, rel: Double) => {
      if( pred  != rel && pred == 0.0 && rel == 1.0){
        1.0
      }
      else{
        0.0
      }
    })
    println("before f1")
    val f1DF = predictionsDF
      .withColumn("TP", f1valueTP($"prediction", $"relevance"))
      .withColumn("TN", f1valueTN($"prediction", $"relevance"))
      .withColumn("FP", f1valueFP($"prediction", $"relevance"))
      .withColumn("FN", f1valueFN($"prediction", $"relevance"))
    f1DF.show(10, false)

    println("read sum")
    val sumTP = f1DF.agg(sum("TP")).first.get(0).toString.toDouble
    val sumTN = f1DF.agg(sum("TN")).first.get(0).toString.toDouble
    val sumFP = f1DF.agg(sum("FP")).first.get(0).toString.toDouble
    val sumFN = f1DF.agg(sum("FN")).first.get(0).toString.toDouble
    println(sumTP + ", " + sumTN + ", " + sumFP + ", " + sumFN)

    val accuracyF1 = ( sumTP + sumTN ) / ( sumTP + sumTN + sumFP + sumFN )
    println("accuracyF1: " + accuracyF1)
    val recallF1 = sumTP / (sumTP + sumFN)
    println("recallF1: " + recallF1)
    val precisionF1 = sumTP / (sumTP + sumFP)
    println("precisionF1: " + precisionF1)
    val f1score = 2 * (recallF1 * precisionF1) / (recallF1 + precisionF1)
    println("F1 score: " + f1score)

  }
}