import org.apache.spark
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
//import sqlContext.implicits._
import org.apache.spark.sql.types.{DoubleType, FloatType}
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.Finisher
import org.apache.spark.ml.Pipeline

object unsupervised_cosine {
  def main(args: Array[String]): Unit = {

    // Create the spark session first
    val ss = SparkSession.builder().master("local").appName("unsupervisedApp").getOrCreate()
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

    val consolidated_with_index = consolidated.withColumn("index",monotonically_increasing_id())

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
        .replaceAll("&nbsp", " ")
        .replaceAll("nbsp", "")
        .replaceAll("&amp", "&")
        .replaceAll("/>/Agt/", "")
        .replaceAll("</a<gt/", "")
        .replaceAll("gt/>", "")
        .replaceAll("/>", "")
        .replaceAll("<br", "")
        /*replacing similar metric words*/
        .replaceAll("foot|feet|ft", "ft")
        .replaceAll("pounds|pound|lbs|lb", "lb")
        .replaceAll("gallons|gallon", "gal")
        .replaceAll("ounces|ounce", "oz")
        .replaceAll("centimeters|centimeter", "cm")
        .replaceAll("milimeters|milimeter", "mm")
        .replaceAll("degrees|degree|Β°", "deg")
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

    /* word2vec */
    val word2vec = new Word2Vec()
      .setMinCount(0)
      .setInputCol("finished_stem")
      .setOutputCol("Word2VecFeatures")
      .setVectorSize(100)

    val word2vecmodel = word2vec.fit(annotations_title_df)

    word2vecmodel.save("word2vecmodel")
    val loadedModel = Word2VecModel.load("word2vecmodel")
    val word2vecTitleDF = loadedModel.transform(annotations_title_df)
    val word2vecQueryDF = loadedModel.transform(annotations_query_df)

    /* convert relevance to binary */
    val binaryConverter = udf((relevance: Double) => {
      if (relevance >= 2.0)
        1
      else
        0
    })

    val word2vecTitleDFJoinedSeq = word2vecTitleDF.withColumn("finished_stem", joinSeq($"finished_stem")).drop("text")
      .withColumn("relevance", binaryConverter($"relevance"))
    val word2vecQueryDFJoinedSeq = word2vecQueryDF.withColumn("finished_stem", joinSeq($"finished_stem")).drop("text")
      .withColumn("relevance", binaryConverter($"relevance"))

    /* Create super dataframe */
    val word2vecTitleDFJoinedSeqRenamed = word2vecTitleDFJoinedSeq.withColumnRenamed("original_text", "title_original_text")
      .withColumnRenamed("finished_stem", "title_finished_stem").withColumnRenamed("Word2VecFeatures", "title_Word2VecFeatures")
    val word2vecQueryDFJoinedSeqRenamed = word2vecQueryDFJoinedSeq.withColumnRenamed("original_text", "query_original_text")
      .withColumnRenamed("finished_stem", "query_finished_stem").withColumnRenamed("Word2VecFeatures", "query_Word2VecFeatures")

    word2vecTitleDFJoinedSeqRenamed.show()
    word2vecQueryDFJoinedSeqRenamed.show()

    val superdf = word2vecTitleDFJoinedSeqRenamed.join(word2vecQueryDFJoinedSeqRenamed, Seq("index", "relevance"))
    superdf.show()

    /* calculate cosine similarity */
    val cosineSimilarity = udf((vectorA: Vector, vectorB: Vector) => {
      var dotProduct = 0.0
      var normA = 0.0
      var normB = 0.0
      var index = vectorA.size - 1

      for (i <- 0 to index) {
        dotProduct += vectorA(i) * vectorB(i)
        normA += Math.pow(vectorA(i), 2)
        normB += Math.pow(vectorB(i), 2)
      }
      dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
    })

    /* Cosine similarity of search vs title */

    val cosine1 = superdf.withColumn("cosine_similarity", cosineSimilarity($"title_Word2VecFeatures", $"query_Word2VecFeatures"))
    val newCosine2 = cosine1.na.drop

    newCosine2.write.parquet("newCosine3.parquet")

    val newCosine1 = ss.read.parquet("newCosine3.parquet")
    println("read parquet")
    newCosine1.select("title_original_text", "query_original_text", "cosine_similarity", "relevance").orderBy(desc("cosine_similarity")).show(200, false)
    newCosine1.printSchema()
    println("after cosine")

    /* similarity threshold */
    val cosinePredictor = udf((similarity: Double) => {
      if (similarity >= 0.6)
        1
      else
        0
    })

    val predictionsDF = newCosine1.withColumn("cosine_similarity", cosinePredictor($"cosine_similarity"))
      .withColumnRenamed("cosine_similarity", "prediction")
      .select("title_original_text", "query_original_text", "prediction", "relevance")

    predictionsDF.show(200, false)

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

