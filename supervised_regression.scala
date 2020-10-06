import com.johnsnowlabs.nlp.Finisher
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object supervised_regression {
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

    consolidatedRemovedSpecials.printSchema()

    val titleDF = consolidatedRemovedSpecials.select("index", "product_title", "relevance").withColumnRenamed("product_title", "original_text")
    val queryDF = consolidatedRemovedSpecials.select("index", "search_term", "relevance").withColumnRenamed("search_term", "original_text")
    val descriptionDF = consolidatedRemovedSpecials.select("index", "product_description", "relevance").withColumnRenamed("product_description", "original_text")
    val brandDF = consolidatedRemovedSpecials.select("index", "value", "relevance").withColumnRenamed("value", "original_text")

    /* tokenization */
    val tokenizer = new Tokenizer().setInputCol("original_text").setOutputCol("tokenized_text")
    val tokenizedTitle = tokenizer.transform(titleDF)
    val tokenizedQuery = tokenizer.transform(queryDF)
    val tokenizedDesc = tokenizer.transform(descriptionDF)
    val tokenizedBrand = tokenizer.transform(brandDF)

    /* stop word removal */
    val stop_words_remover = new StopWordsRemover()
      .setInputCol("tokenized_text")
      .setOutputCol("filtered_text")

    val joinSeq = udf { (words: Seq[String]) => words.mkString(" ") }

    val removedStopwordsTitle = stop_words_remover.transform(tokenizedTitle)
    val removedStopwordsQuery = stop_words_remover.transform(tokenizedQuery)
    val removedStopwordsDesc = stop_words_remover.transform(tokenizedDesc)
    val removedStopwordsBrand = stop_words_remover.transform(tokenizedBrand)

    val removedStopwordsTitleJoinedSeq = removedStopwordsTitle.withColumn("filtered_text", joinSeq($"filtered_text")).drop("tokenized_text")
    val removedStopwordsQueryJoinedSeq = removedStopwordsQuery.withColumn("filtered_text", joinSeq($"filtered_text")).drop("tokenized_text")
    val removedStopwordsDescJoinedSeq = removedStopwordsDesc.withColumn("filtered_text", joinSeq($"filtered_text")).drop("tokenized_text")
    val removedStopwordsBrandJoinedSeq = removedStopwordsBrand.withColumn("filtered_text", joinSeq($"filtered_text")).drop("tokenized_text")

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
    val dataDesc = removedStopwordsDescJoinedSeq.withColumnRenamed("filtered_text", "text")
    val dataBrand = removedStopwordsBrandJoinedSeq.withColumnRenamed("filtered_text", "text")

    val modelTitle = pipeline.fit(dataTitle)
    val modelQuery = pipeline.fit(dataQuery)
    val modelDesc = pipeline.fit(dataDesc)
    val modelBrand = pipeline.fit(dataBrand)

    val annotations_title_df = modelTitle.transform(dataTitle)
    val annotations_query_df = modelQuery.transform(dataQuery)
    val annotations_desc_df = modelDesc.transform(dataDesc)
    val annotations_brand_df = modelBrand.transform(dataBrand)

    /* word2vec */
    val word2vec = new Word2Vec()
      .setMinCount(0)
      .setInputCol("finished_stem")
      .setOutputCol("Word2VecFeatures")
      .setVectorSize(100)

    /* tf-idf */
    val hashingTF = new HashingTF()
      .setInputCol("finished_stem").setOutputCol("tf-idf_rawFeatures").setNumFeatures(10000)

    val featurizedDataTitleDF = hashingTF.transform(annotations_title_df)
    val featurizedDataQueryDF = hashingTF.transform(annotations_query_df)
    val featurizedDataDescDF = hashingTF.transform(annotations_desc_df)
    val featurizedDataBrandDF = hashingTF.transform(annotations_brand_df)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("tf-idf_rawFeatures").setOutputCol("tf-idf_features")
    val idfMod = idf.fit(featurizedDataTitleDF)
    idfMod.save("tf-idfmodel")
    val idfModel = IDFModel.load("tf-idfmodel")

    val rescaledDataTitleDF = idfModel.transform(featurizedDataTitleDF)
    val rescaledDataQueryDF = idfModel.transform(featurizedDataQueryDF)
    val rescaledDataDescDF = idfModel.transform(featurizedDataDescDF)
    val rescaledDataBrandDF = idfModel.transform(featurizedDataBrandDF)
    rescaledDataTitleDF.show()
    rescaledDataQueryDF.show()
    rescaledDataDescDF.show()
    rescaledDataBrandDF.show()

    //val word2vecmodel = word2vec.fit(annotations_title_df)

    //word2vecmodel.save("word2vecmodel")
    val loadedModel = Word2VecModel.load("word2vecmodel")
    val word2vecTitleDF = loadedModel.transform(rescaledDataTitleDF)
    val word2vecQueryDF = loadedModel.transform(rescaledDataQueryDF)
    val word2vecDescDF = loadedModel.transform(rescaledDataDescDF)
    val word2vecBrandDF = loadedModel.transform(rescaledDataBrandDF)

    val word2vecTitleDFJoinedSeq = word2vecTitleDF.withColumn("finished_stem", joinSeq($"finished_stem")).drop("text")
    val word2vecQueryDFJoinedSeq = word2vecQueryDF.withColumn("finished_stem", joinSeq($"finished_stem")).drop("text")
    val word2vecDescDFJoinedSeq = word2vecDescDF.withColumn("finished_stem", joinSeq($"finished_stem")).drop("text")
    val word2vecBrandDFJoinedSeq = word2vecBrandDF.withColumn("finished_stem", joinSeq($"finished_stem")).drop("text")

    /* Create super dataframe */
    val word2vecTitleDFJoinedSeqRenamed = word2vecTitleDFJoinedSeq.withColumnRenamed("original_text", "title_original_text")
      .withColumnRenamed("finished_stem", "title_finished_stem").withColumnRenamed("Word2VecFeatures", "title_Word2VecFeatures")
      .withColumnRenamed("tf-idf_features", "title_tf-idf_features")
    val word2vecQueryDFJoinedSeqRenamed = word2vecQueryDFJoinedSeq.withColumnRenamed("original_text", "query_original_text")
      .withColumnRenamed("finished_stem", "query_finished_stem").withColumnRenamed("Word2VecFeatures", "query_Word2VecFeatures")
      .withColumnRenamed("tf-idf_features", "query_tf-idf_features")
    val word2vecDescDFJoinedSeqRenamed = word2vecDescDFJoinedSeq.withColumnRenamed("original_text", "desc_original_text")
      .withColumnRenamed("finished_stem", "desc_finished_stem").withColumnRenamed("Word2VecFeatures", "desc_Word2VecFeatures")
      .withColumnRenamed("tf-idf_features", "desc_tf-idf_features")
    val word2vecBrandDFJoinedSeqRenamed = word2vecBrandDFJoinedSeq.withColumnRenamed("original_text", "brand_original_text")
      .withColumnRenamed("finished_stem", "brand_finished_stem").withColumnRenamed("Word2VecFeatures", "brand_Word2VecFeatures")
      .withColumnRenamed("tf-idf_features", "brand_tf-idf_features")

    val superdf = word2vecTitleDFJoinedSeqRenamed.join(word2vecQueryDFJoinedSeqRenamed, "index")
      .join(word2vecDescDFJoinedSeqRenamed, "index").join(word2vecBrandDFJoinedSeqRenamed, "index")
      /*select columns for df*/
      .select(word2vecTitleDFJoinedSeqRenamed("index"), word2vecTitleDFJoinedSeqRenamed("title_finished_stem"),
      word2vecTitleDFJoinedSeqRenamed("title_Word2VecFeatures"), word2vecTitleDFJoinedSeqRenamed("title_tf-idf_features"),
      word2vecQueryDFJoinedSeqRenamed("query_finished_stem"), word2vecQueryDFJoinedSeqRenamed("query_Word2VecFeatures"),
      word2vecQueryDFJoinedSeqRenamed("query_tf-idf_features"), word2vecDescDFJoinedSeqRenamed("desc_finished_stem"),
      word2vecDescDFJoinedSeqRenamed("desc_Word2VecFeatures"), word2vecDescDFJoinedSeqRenamed("desc_tf-idf_features"),
      word2vecBrandDFJoinedSeqRenamed("brand_finished_stem"), word2vecBrandDFJoinedSeqRenamed("brand_Word2VecFeatures"),
      word2vecBrandDFJoinedSeqRenamed("brand_tf-idf_features"), word2vecBrandDFJoinedSeqRenamed("relevance"))

    superdf.show()

    superdf.write.parquet("final_superdf.parquet")

    val final_superdf = ss.read.parquet("final_superdf.parquet")
    final_superdf.printSchema()

    /* calculate euclidean distance */
    val euclideanDistance = udf((vectorA: Vector, vectorB: Vector) => {
      var sum = 0.0
      val index = vectorA.size - 1

      for (i <- 0 to index) {
        sum += Math.pow(vectorA(i) - vectorB(i), 2)
      }
      Math.sqrt(sum)
    })

    /* calculate cosine similarity */
    val cosineSimilarity = udf((vectorA: Vector, vectorB: Vector) => {
      var dotProduct = 0.0
      var normA = 0.0
      var normB = 0.0
      val index = vectorA.size - 1

      for (i <- 0 to index) {
        dotProduct += vectorA(i) * vectorB(i)
        normA += Math.pow(vectorA(i), 2)
        normB += Math.pow(vectorB(i), 2)
      }
      dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
    })

    /* Cosine similarity and euclidean distance of search vs title */
    val cosine1 = final_superdf.withColumn("cosine_similarity", cosineSimilarity($"title_Word2VecFeatures", $"query_Word2VecFeatures"))
      .withColumn("euclidean_distance", euclideanDistance($"title_Word2VecFeatures", $"query_Word2VecFeatures"))
    val newCosine1 = cosine1.na.drop
    //newCosine1.select("title_original_text", "query_original_text", "cosine_similarity", "relevance").orderBy(desc("cosine_similarity")).show(200, false)
    newCosine1.show()
    newCosine1.printSchema()

    val commonterms_SearchVsTitle = udf((filtered_search_words: String, filtered_title_words:String) =>
      if (filtered_search_words.isEmpty || filtered_title_words.isEmpty){
        0
      }
      else{
        var tmp1 = filtered_search_words.split(" ")
        var tmp2 = filtered_title_words.split(" ")
        tmp1.intersect(tmp2).length
      })

    val commonterms_SearchVsDescription = udf((filtered_search_words: String, filtered_description_words:String) =>
      if (filtered_search_words.isEmpty || filtered_description_words.isEmpty){
        0
      }
      else{
        var tmp1 = filtered_search_words.split(" ")
        var tmp2 = filtered_description_words.split(" ")
        tmp1.intersect(tmp2).length
      })

    /*SearchVsTitle*/
    val results = newCosine1.withColumn("common_words_ST_T", commonterms_SearchVsTitle($"query_finished_stem", $"title_finished_stem"))
    /*SearchVsDescription*/
    val results1and2 = results.withColumn("common_words_ST_D", commonterms_SearchVsDescription($"query_finished_stem", $"desc_finished_stem"))
    val results1and2and3 = results1and2.withColumn("common_words_ST_B", commonterms_SearchVsDescription($"query_finished_stem", $"brand_finished_stem"))


    val newConsolidated = results1and2and3
      .withColumn("product_title_len", size(split('title_finished_stem, " ")))
      .withColumn("search_term_len", size(split('query_finished_stem, " ")))
      .withColumn("product_description_len", size(split('desc_finished_stem, " ")))
      .withColumn("brand_len", size(split('brand_finished_stem, " ")))
      .withColumn("ratio_desc_len_search_len", size(split('desc_finished_stem, " "))/size(split('query_finished_stem, " ")))
      .withColumn("ratio_title_len_search_len", size(split('title_finished_stem, " "))/size(split('query_finished_stem, " ")))
      .withColumn("common_words_ST_T", $"common_words_ST_T")
      .withColumn("common_words_ST_D", $"common_words_ST_D")
      .withColumn("common_words_ST_B", $"common_words_ST_B")
      .withColumn("cosine_similarity", $"cosine_similarity")
      .withColumn("euclidean_distance", $"euclidean_distance")

    newConsolidated.show()

    println("to reg")
    /* ======================================================= */
    /* ===================== REGRESSION ====================== */
    /* ======================================================= */

    def train_test_split(data: DataFrame) = {

      val assembler = new VectorAssembler()
        .setInputCols(data.drop("relevance").columns)
        .setOutputCol("features")

      val Array(train, test) = data.randomSplit(Array(0.6, 0.4), seed = 42)

      (assembler.transform(train), assembler.transform(test))
    }

    // Make sure the columns <features> and <label> exist
    //val df = newConsolidated.select("product_title_len", "search_term_len", "product_description_len", "ratio_desc_len_search_len", "ratio_title_len_search_len", "relevance")

    val toDouble = udf[Double, String]( _.toDouble)

    val df = newConsolidated.withColumn("relevance", toDouble(newConsolidated("Relevance"))).select(
      //tf-idf_features are commented out since it was proven during the experiments that they increase MSE
      "product_title_len",
      "search_term_len",
      "product_description_len", "brand_len", "ratio_desc_len_search_len",
      "ratio_title_len_search_len",
      "common_words_ST_T", "common_words_ST_D", "common_words_ST_B",
      "cosine_similarity", "euclidean_distance",
      "title_Word2VecFeatures", //"title_tf-idf_features",
      "query_Word2VecFeatures", //"query_tf-idf_features",
      "desc_Word2VecFeatures",  //"desc_tf-idf_features",
      "brand_Word2VecFeatures", //"brand_tf-idf_features",
      "relevance")
    //val df = newConsolidated.select("search_term_len", "common_words_ST", "common_words_SD", "relevance")
    newConsolidated.unpersist()
    println("dataframe: ")

    df.show()

    val (train_notnormalized, test_notnormalized) = train_test_split(df)

    /*
    //Normalize is commented out since it was proven during the experiments that it increases MSE
    // Normalize each Vector using $L^1$ norm.
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)
 
    /* normalize train set */
    val l1NormData_train = normalizer.transform(train_notnormalized).withColumnRenamed("features", "old_features").withColumnRenamed("normFeatures", "features")
    l1NormData_train.show(false)
    // Normalize each Vector using $L^\infty$ norm.
    val train_normalized = normalizer.transform(train_notnormalized, normalizer.p -> Double.PositiveInfinity).withColumnRenamed("features", "old_features").withColumnRenamed("normFeatures", "features")
    train_normalized.show(false)
 
    /* normalize test set */
    val l1NormData_test = normalizer.transform(test_notnormalized).withColumnRenamed("features", "old_features").withColumnRenamed("normFeatures", "features")
    l1NormData_test.show(false)
    // Normalize each Vector using $L^\infty$ norm.
    val test_normalized = normalizer.transform(test_notnormalized, normalizer.p -> Double.PositiveInfinity).withColumnRenamed("features", "old_features").withColumnRenamed("normFeatures", "features")
    test_normalized.show(false)
    */

    //val train  = train_normalized.select("features", "relevance")
    val train  = train_notnormalized.select("features", "relevance")
    train.show()
    //val test   = test_normalized.select("features", "relevance")
    val test   = test_notnormalized.select("features", "relevance")
    test.show()

    val mse = new RegressionEvaluator()
      .setLabelCol("relevance")
      .setPredictionCol("prediction")
      .setMetricName("mse")


        //it is recommended to run ChiSqSelector without using the Word2Vec features
        /* ChiSqSelector */
        val selector = new ChiSqSelector()
          .setNumTopFeatures(5)
          .setFeaturesCol("features")
          .setLabelCol("relevance")
          .setOutputCol("selectedFeatures")
 
        val result = selector.fit(train).transform(train)
 
        println(s"ChiSqSelector output with top ${selector.getNumTopFeatures} features selected")
        result.show(false)


    /* =============================================================================== */
    /*                               Linear Regression                                 */
    /* =============================================================================== */

    println("Linear Regression training...")
    val lr = new LinearRegression()
      .setMaxIter(10000)
      .setRegParam(0.1)
      .setElasticNetParam(0.0)
      .setLabelCol("relevance")
      .setFeaturesCol("features")

    val lrModel = lr.fit(train)
    val predictionslr = lrModel.transform(test)

    println("Linear Regression Mean Squared Error (MSE) on test data = " + mse.evaluate(predictionslr))

    /* =============================================================================== */
    /*                        Generalized Linear Regression                            */
    /* =============================================================================== */


        println("Generalized Linear Regression training...")
        val glr = new GeneralizedLinearRegression()
          .setFamily("gaussian")
          .setLink("identity")
          .setMaxIter(10000)
          .setRegParam(0.3)
          .setLabelCol("relevance")
          .setFeaturesCol("features")

        // Fit the model
        val glrModel = glr.fit(train)
        val predictionsGlrModel = glrModel.transform(test)

        println(s"Coefficients: ${glrModel.coefficients}")
        println(s"Intercept: ${glrModel.intercept}")
        //println(s"Coefficients: ${predictionsGlrModel.coefficients}")
        //println(s"Intercept: ${predictionsGlrModel.intercept}")

        println("Generalized Linear Regression Mean Squared Error (MSE) on test data = " + mse.evaluate(predictionsGlrModel))


    /* =============================================================================== */
    /*                             Decision Tree Regression                            */
    /* =============================================================================== */

    println("Decision Tree training...")

    val dt = new DecisionTreeRegressor()
      .setLabelCol("relevance")
      .setFeaturesCol("features")

    val dtModel = dt.fit(train)
    val dt_predictions = dtModel.transform(test)
    println("Decision Tree Mean Squared Error (MSE) on test data = " + mse.evaluate(dt_predictions))
    //val fiDTVector: Vector = dtModel.featureImportances
    //println(fiDTVector)
    /* =============================================================================== */
    /*                             Random Forest Regression                            */
    /* =============================================================================== */


    println("Random Forest training...")


    val rf = new RandomForestRegressor()
      .setLabelCol("relevance")
      .setFeaturesCol("features")

    val rfModel = rf.fit(train)
    val rf_predictions = rfModel.transform(test)
    println("Random Forest Mean Squared Error (MSE) on test data = " + mse.evaluate(rf_predictions))
    //val fiRFVector: Vector = rfModel.featureImportances
    //println(fiRFVector)
    /* =============================================================================== */
    /*                        Gradient Boosted Tree Regression                         */
    /* =============================================================================== */

    println("Gradient Boosted Tree training...")

    val gbt = new GBTRegressor()
      .setLabelCol("relevance")
      .setFeaturesCol("features")

    val gbtModel = gbt.fit(train)
    val gbt_predictions = gbtModel.transform(test)
    println("Gradient Boosting Mean Squared Error (MSE) on test data = " + mse.evaluate(gbt_predictions))
    //val fiGBTVector: Vector = gbtModel.featureImportances
    //println(fiGBTVector)
    /* =============================================================================== */
    /*                               Isotonic Regression                               */
    /* =============================================================================== */

    println("Isotonic Regression training...")

    // Trains an isotonic regression model.
    val ir = new IsotonicRegression()
      .setLabelCol("relevance")
      .setFeaturesCol("features")

    val IRmodel = ir.fit(train)
    val ir_predictions = IRmodel.transform(test)

    println(s"Boundaries in increasing order: ${IRmodel.boundaries}\n")
    println(s"Predictions associated with the boundaries: ${IRmodel.predictions}\n")

    // Makes predictions.
    IRmodel.transform(train).show()

    println("Isotonic Regression Mean Squared Error (MSE) on test data = " + mse.evaluate(ir_predictions))

  }
}