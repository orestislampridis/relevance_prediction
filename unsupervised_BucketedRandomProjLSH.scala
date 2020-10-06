import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, MinHashLSH}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, _}

object unsupervised_BucketedRandomProjLSH {
  def main(args: Array[String]): Unit = {

    // Create the spark session first
    val ss = SparkSession.builder().master("local").appName("minhashbucketsApp").getOrCreate()
    import ss.implicits._
    // For implicit conversions like converting RDDs to DataFrames
    val currentDir = System.getProperty("user.dir") // get the current directory

    val newVectorizedTitleDFJoinedSeq = ss.read.parquet("vectorizedTitleDFJoinedSeq.parquet").orderBy(desc("index")).limit(500)
    val newVectorizedQueryDFJoinedSeq = ss.read.parquet("vectorizedQueryDFJoinedSeq.parquet").orderBy(desc("index")).limit(500)

    newVectorizedTitleDFJoinedSeq.show()
    newVectorizedQueryDFJoinedSeq.show()

    val brp = new BucketedRandomProjectionLSH()
      .setBucketLength(2.0)
      .setNumHashTables(3)
      .setInputCol("features")
      .setOutputCol("hashes")

    val model = brp.fit(newVectorizedTitleDFJoinedSeq)

    // Feature Transformation
    val dataset1_LSH = model.transform(newVectorizedTitleDFJoinedSeq)
    val dataset2_LSH = model.transform(newVectorizedQueryDFJoinedSeq)

    // Compute the locality sensitive hashes for the input rows, then perform approximate
    // similarity join.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxSimilarityJoin(transformedA, transformedB, 1.5)`
    val jaccardDF = model.approxSimilarityJoin(dataset1_LSH, dataset2_LSH, 1000, "EuclideanDistance")
      .select(col("datasetA.index").alias("indexA"),
        col("datasetB.index").alias("indexB"), col("datasetA.original_text").alias("textA"),
        col("datasetB.original_text").alias("textB"), col("datasetA.relevance").alias("relevance"),
        col("EuclideanDistance"))

    jaccardDF.show(1000, false)

    //val newJaccardDF = jaccardDF.join(jaccardDF.groupBy("indexA", "indexB").count().where("count = 1").drop("count"), Seq("indexA", "indexB"), "left_anti").show()

    /* euclidean threshold */
    val euclideanPredictor = udf((distance: Double) => {
      if (distance > 5)
        0
      else
        1
    })

    val predictionsDF = jaccardDF.withColumn("EuclideanDistance", euclideanPredictor($"EuclideanDistance"))
      .withColumnRenamed("EuclideanDistance", "prediction").filter(col("indexA") === col("indexB"))

    predictionsDF.show(100, false)

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