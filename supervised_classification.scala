import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{IntegerType, LongType, StringType, StructField, StructType}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.PCA


object supervised_classification {

  def main(args: Array[String]): Unit = {

    // Create the spark session first
    val ss = SparkSession.builder().master("local").appName("tfidfApp").getOrCreate()
    import ss.implicits._  // For implicit conversions like converting RDDs to DataFrames
    val currentDir = System.getProperty("user.dir")  // get the current directory
    val inputFile = "./attributes.csv + ./product_description.csv + ./train.csv"
    //val outputDir = "file://" + currentDir + "/output"

    println("reading from input file: " + inputFile)
    println

    val trainDF = ss.read.format("csv").option("header", "true").load("train.csv")
    trainDF.printSchema()
    val descriptionDF = ss.read.format("csv").option("header", "true").load("product_descriptions.csv")
    descriptionDF.printSchema()
    val attributesDF = ss.read.format("csv").option("header", "true").load("attributes.csv")
    attributesDF.printSchema()

    val newAttributesDF = attributesDF.filter(attributesDF("name")==="MFG Brand Name")
    val newNewAttributesDF = newAttributesDF.select("product_uid","value")

    val consolidated = trainDF.join(descriptionDF, "product_uid").join(newNewAttributesDF, "product_uid")
      /*select columns for df*/
      .select(trainDF("product_uid"), trainDF("product_title"), trainDF("search_term"),
      trainDF("relevance"), descriptionDF("product_description"), newNewAttributesDF("value"))
      /*turn all columns in lower caps*/
      .withColumn("product_description",lower(col("product_description"))).withColumn("product_title", lower(col("product_title")))
      .withColumn("search_term", lower(col("search_term"))).withColumn("value", lower(col("value")))

    /*clear unused dfs*/
    trainDF.unpersist()
    descriptionDF.unpersist()
    attributesDF.unpersist()
    newAttributesDF.unpersist()
    newNewAttributesDF.unpersist()

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
        .replaceAll("inches|inch", "in.")
        .replaceAll("foot|feet|ft", "ft.")
        .replaceAll("pounds|pound|lbs|lb", "lb.")
        .replaceAll("square|sq", "sq.")
        .replaceAll("cubic|cu", "cu.")
        .replaceAll("gallons|gallon|gal", "gal.")
        .replaceAll("ounces|ounce|oz", "oz.")
        .replaceAll("centimeters|cm", "cm.")
        .replaceAll("milimeters|mm", "mm.")
        .replaceAll("degrees|degree|°|deg", "deg.")
        .replaceAll("volts|volt", "volt.")
        .replaceAll("wattage|watts|watt", "watt.")
        .replaceAll("ampere|amps|amperes|amp", "amp.")
        .replaceAll("qquart|quart|qt", "qt.")
        .replaceAll("gallons per minute|gallon per minute|gal per minute|gallons/min|gallon/min", "gal per min.")
        .replaceAll("gallons per hour|gallon per hour|gal per hour|gallons/hour|gallon/hour", "gal per hr.")
        .replaceAll("hrs|hrs.|hours|hour|hr", "hr.")
        .replaceAll("min|mins.|minutes|minute", "min.")

    /*calling removeSpecials*/
    val udf_removeSpecials = udf(removeSpecials)
    val consolidatedRemovedSpecials = consolidated.withColumn("product_description", udf_removeSpecials($"product_description")).withColumn("product_title", udf_removeSpecials($"product_title"))
      .withColumn("search_term", udf_removeSpecials($"search_term")).withColumn("value", udf_removeSpecials($"value"))
    //consolidated.unpersist()
    consolidatedRemovedSpecials.show()

    val tokenizerTitle = new Tokenizer().setInputCol("product_title").setOutputCol("product_title_words")
    val tokenizedTitle = tokenizerTitle.transform(consolidatedRemovedSpecials)
    consolidatedRemovedSpecials.unpersist()
    tokenizedTitle.select("product_title", "product_title_words")
    val removerTitle = new StopWordsRemover()
      .setInputCol("product_title_words")
      .setOutputCol("filtered_title_words")

    val joinSeq = udf { (words: Seq[String]) => words.mkString(" ") }

    val removedStopwordsTitle = removerTitle.transform(tokenizedTitle)
    tokenizedTitle.unpersist()
    val removedStopwordsTitleJoinedSeq = removedStopwordsTitle.withColumn("filtered_title_words", joinSeq($"filtered_title_words"))
    removedStopwordsTitle.unpersist()
    val tokenizerDesc = new Tokenizer().setInputCol("product_description").setOutputCol("product_description_words")
    val tokenizedDesc = tokenizerDesc.transform(removedStopwordsTitleJoinedSeq)
    tokenizedDesc.select("product_description", "product_description_words")
    val removerDesc = new StopWordsRemover()
      .setInputCol("product_description_words")
      .setOutputCol("filtered_description_words")

    val removedStopwordsDesc = removerDesc.transform(tokenizedDesc)
    tokenizedDesc.unpersist()
    val removedStopwordsDescJoinedSeq = removedStopwordsDesc.withColumn("filtered_description_words", joinSeq($"filtered_description_words"))
    removedStopwordsDesc.unpersist()
    val tokenizerSearch = new Tokenizer().setInputCol("search_term").setOutputCol("search_term_words")
    val tokenizedSearch = tokenizerSearch.transform(removedStopwordsDescJoinedSeq)
    removedStopwordsDescJoinedSeq.unpersist()
    tokenizedSearch.select("search_term", "search_term_words")
    val removerSearch = new StopWordsRemover()
      .setInputCol("search_term_words")
      .setOutputCol("filtered_search_words")

    val removedStopwordsSearch = removerSearch.transform(tokenizedSearch)
    tokenizedSearch.unpersist()
    val removedStopwordsSearchJoinedSeq = removedStopwordsSearch.withColumn("filtered_search_words", joinSeq($"filtered_search_words"))
    removedStopwordsSearch.unpersist()
    removedStopwordsSearchJoinedSeq.show(false)
    removedStopwordsSearchJoinedSeq.printSchema()

    /*
        //nGram use
        val ngramSTW = new NGram().setN(1).setInputCol("search_term_words").setOutputCol("ngrams")
        val ngramSTWDF = ngramSTW.transform(removedStopwordsSearchJoinedSeq)
        ngramSTWDF.select("product_uid","search_term","ngrams").show(false)

        val ngramPTW = new NGram().setN(1).setInputCol("product_title_words").setOutputCol("ngrams")
        val ngramPTWDF = ngramPTW.transform(removedStopwordsSearchJoinedSeq)
        ngramPTWDF.select("product_uid","product_title","ngrams").show(false)
    */

    //val DFjoinedSTWandPTW = ngramSTWDF.join(ngramPTWDF, "product_uid")

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

    val countTimesSearchWordsUsed = udf((filtered_search_words: String, filtered_title_words:String, filtered_description_words:String) =>
      if (filtered_search_words.isEmpty || filtered_title_words.isEmpty){
        0
      }
      else{
        var tmp1 = filtered_search_words
        //var tmp2 = filtered_title_words.split(" ")
        var count = 0
        //for (word <- filtered_search_words){
        if (filtered_title_words.contains(filtered_search_words)){
          count += 1
        }
        if (filtered_description_words.contains(filtered_search_words)){
          count += 1
        }
        //}
        count
      })

    /*SearchVsTitle*/
    val results = removedStopwordsSearchJoinedSeq.withColumn("common_words_ST", commonterms_SearchVsTitle($"filtered_search_words", $"filtered_title_words"))
    results.select("common_words_ST").show()
    results.printSchema()
    /*SearchVsDescription*/
    val results2 = removedStopwordsSearchJoinedSeq.withColumn("common_words_SD", commonterms_SearchVsDescription($"filtered_search_words", $"filtered_description_words"))
    results2.select("common_words_SD").show()
    results2.printSchema()
    /*SearchVsTitle + SearchVsDescription*/
    val results1and2 = results.withColumn("common_words_SD", commonterms_SearchVsDescription($"filtered_search_words", $"filtered_description_words"))
    results1and2.printSchema()
    results.unpersist()
    results2.unpersist()

    /* ===== Removed Specials Characters and Stop Words and Common Words (SearchVsTitle and SearchVsDescription) ===== */
    val newConsolidated = results1and2
      .withColumn("product_title_len", size(split('filtered_title_words, " ")))
      .withColumn("search_term_len", size(split('filtered_search_words, " ")))
      .withColumn("product_description_len", size(split('filtered_description_words, " ")))
      .withColumn("ratio_desc_len_search_len", size(split('filtered_description_words, " "))/size(split('filtered_search_words, " ")))
      .withColumn("ratio_title_len_search_len", size(split('filtered_title_words, " "))/size(split('filtered_search_words, " ")))
      .withColumn("common_words_ST", $"common_words_ST")
      .withColumn("common_words_SD", $"common_words_SD")
    results.unpersist()

    /* ======================================================= */
    /* =================== CLASSIFICATION ==================== */
    /* ======================================================= */

    val toDouble = udf[Double, String]( _.toDouble)
    val toInt = udf[Int, Double]( _.toInt)

    val binary_classification = udf( (d: Double) =>
      if(d >= 2.5)
        1.0
      else
        0.0
    )
    val binary_classificationaboveOnepointFive = udf( (d: Double) =>
      if(d >= 1.5)
        1.0
      else
        0.0
    )


    val thirteenclasses_classification = udf( (d: Double) =>
      //13 classes
      if(d >= 3)            12.0
      else if (d >= 2.75)   11.0
      else if (d >= 2.67)   10.0
      else if (d >= 2.5)    9.0
      else if (d >= 2.33)   8.0
      else if (d >= 2.25)   7.0
      else if (d >= 2)      6.0
      else if (d >= 1.75)   5.0
      else if (d >= 1.67)   4.0
      else if (d >= 1.5)    3.0
      else if (d >= 1.33)   2.0
      else if (d >= 1.25)   1.0
      else                  0.0
    )
    val fourclasses_classification = udf( (d: Double) =>
      //4 classes
      if (d>= 2.67)       3.0
      else if(d >= 2.33)  2.0
      else if(d >= 1.67 ) 1.0
      else                0.0
    )
    val threeclasses_classification = udf( (d: Double) =>
      //3 classes
      if(d >= 2.5)        2.0
      else if(d >= 1.67 ) 1.0
      else                0.0
    )

    def train_test_split(data: DataFrame) = {

      val assembler = new VectorAssembler()
        .setInputCols(data.drop("label").columns)
        .setOutputCol("features")

      val Array(train, test) = data.randomSplit(Array(0.6, 0.4), seed = 42)

      (assembler.transform(train), assembler.transform(test))
    }

    val newDF = newConsolidated
      //.withColumn("relevance", binary_classification(toDouble(newConsolidated("Relevance"))))
      .withColumn("relevance", binary_classificationaboveOnepointFive(toDouble(newConsolidated("Relevance"))))
      //.withColumn("relevance", thirteenclasses_classification(toDouble(newConsolidated("Relevance"))))
      //.withColumn("relevance", fourclasses_classification(toDouble(newConsolidated("Relevance"))))
      //.withColumn("relevance", threeclasses_classification(toDouble(newConsolidated("Relevance"))))
      .select("product_title_len", "search_term_len", "product_description_len", "ratio_desc_len_search_len", "ratio_title_len_search_len", "common_words_ST", "common_words_SD", "relevance")
      .withColumnRenamed("relevance", "label")

    val (train_notnormalized, test_notnormalized) = train_test_split(newDF)
    train_notnormalized.drop("product_title_len", "search_term_len", "product_description_len", "ratio_desc_len_search_len", "ratio_title_len_search_len", "common_words_ST", "common_words_SD")
    test_notnormalized.drop("product_title_len", "search_term_len", "product_description_len", "ratio_desc_len_search_len", "ratio_title_len_search_len", "common_words_ST", "common_words_SD")
    train_notnormalized.show(false)
    test_notnormalized.show(false)

    /* Normalize each Vector using $L^1$ norm. */
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)

    /* normalize train set */
    //val l1NormData_train = normalizer.transform(train_notnormalized)
    // Normalize each Vector using $L^\infty$ norm.
    val train_normalized = normalizer.transform(train_notnormalized, normalizer.p -> Double.PositiveInfinity).withColumnRenamed("features", "oldfeatures").withColumnRenamed("normFeatures", "features")
    train_normalized.show(false)

    /* normalize test set */
    //val l1NormData_test = normalizer.transform(test_notnormalized).show(false)
    // Normalize each Vector using $L^\infty$ norm.
    val test_normalized = normalizer.transform(test_notnormalized, normalizer.p -> Double.PositiveInfinity).withColumnRenamed("features", "oldfeatures").withColumnRenamed("normFeatures", "features")
    test_normalized.show(false)

    val train  = train_normalized.select("features", "label")
    train.show()
    val test   = test_normalized.select("features", "label")
    test.show()


    /* PCA */
    println("pca train set...")
    val pca_train = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(3)
      .fit(train)

    val result_train = pca_train.transform(train).select("label", "pcaFeatures").withColumnRenamed("pcaFeatures", "features")
    result_train.show(false)

    println("pca test set...")
    val pca_test = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(3)
      .fit(test)

    val result_test = pca_test.transform(test).select("label", "pcaFeatures").withColumnRenamed("pcaFeatures", "features")
    result_test.show(false)


    println("BEFORE TRAINING")

    /* =============================================================================== */
    /* ========================== Naive Bayes Classifier ============================= */
    /* =============================================================================== */
        //Naive Bayes does not run with PCA features since it does no support negative values
        println("Naive Bayes Classifier training...")

        val NBmodel = new NaiveBayes().fit(train)
        println("before predictionsNB...")
        val predictionsNB = NBmodel.transform(test)
        predictionsNB.printSchema()
        //predictionsNB.take(100).foreach(println)
        //predictionsNB.select("label", "prediction").show(100)
        predictionsNB.show(50)
        println("before evaluatorNB...")
        // Evaluate the model by finding the accuracy
        val evaluatorNB = new MulticlassClassificationEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")
        println("before accuracyNB...")
        val accuracyNB = evaluatorNB.evaluate(predictionsNB)
        println("Accuracy of Naive Bayes: " + accuracyNB)

    /* =============================================================================== */
    /* ========================== One-vs-Rest Classifier ============================= */
    /* =============================================================================== */

        println("One-vs-Rest Classifier training...")

        println("instantiate the base classifier...")
        val OvRCclassifier = new LogisticRegression()
          .setMaxIter(10000)
          .setTol(1E-6)
          .setFitIntercept(true)

        println("instantiate the One Vs Rest Classifier...")
        val ovr = new OneVsRest().setClassifier(OvRCclassifier)

        println("train the multiclass model...")
        val OvRCModel = ovr.fit(result_train) //result_train

        println("score the model on test data...")
        val OvRCpredictions = OvRCModel.transform(result_test) //result_test
        OvRCpredictions.show(false)

        println("obtain evaluator...")
        val OvRCevaluator = new MulticlassClassificationEvaluator()
          .setMetricName("accuracy")

        println("compute the classification error on test data...")
        val OvRCaccuracy = OvRCevaluator.evaluate(OvRCpredictions)
        println(s"Test Error = ${1 - OvRCaccuracy}")
        println("Accuracy of One-vs-Rest Classifier: " + OvRCaccuracy)

        /* =============================================================================== */
        /* ========================= Random Forest Classifier ============================ */
        /* =============================================================================== */

        println("Random Forest Classifier training...")

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        println("labelIndexer creation...")
        val RFClabelIndexer = new StringIndexer()
          .setInputCol("label")
          .setOutputCol("indexedLabel")
          .fit(result_train) //result_train
        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        println("featureIndexer creation...")
        val RFCfeatureIndexer = new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4)
          .fit(result_train) //result_train

        // Split the data into training and test sets (30% held out for testing).
        //val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

        println("train a randomForest model...")
        val rf = new RandomForestClassifier()
          .setLabelCol("indexedLabel")
          .setFeaturesCol("indexedFeatures")
          .setNumTrees(1000)

        println("Convert indexed labels back to original labels...")
        val RFClabelConverter = new IndexToString()
          .setInputCol("prediction")
          .setOutputCol("predictedLabel")
          .setLabels(RFClabelIndexer.labels)

        println("Chain indexers and forest in a Pipeline...")
        val RFCpipeline = new Pipeline()
          .setStages(Array(RFClabelIndexer, RFCfeatureIndexer, rf, RFClabelConverter))

        println("train model. This also runs the indexers...")
        val RFCmodel = RFCpipeline.fit(result_train) //result_train

        println("making predictions...")
        val RFCpredictions = RFCmodel.transform(result_test) //result_test

        println("displaying example rows...")
        RFCpredictions.select("predictedLabel", "label", "features").show(false)

        println("Select (prediction, true label) and compute test error...")
        val RFCevaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("indexedLabel")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")

        println("evaluating accuracy...")
        val RFCaccuracy = RFCevaluator.evaluate(RFCpredictions)
        //println(s"Test Error = ${(1.0 - RFCaccuracy)}")
        println(s"Test Error = ${1.0 - RFCaccuracy}")
        println("Accuracy of Random Forest Classifier: " + RFCaccuracy)

  }

}