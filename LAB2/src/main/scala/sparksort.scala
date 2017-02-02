
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by Mayanka on 09-Sep-15.
  */
object sparksort {

  def main(args: Array[String]) {

    // System.setProperty("hadoop.home.dir","C:\\Users\\Manikanta\\Documents\\UMKC Subjects\\PB\\hadoopforspark");

    val sparkConf = new SparkConf().setAppName("Sparksort").setMaster("local[*]")

    val sc = new SparkContext(sparkConf)

    val in = sc.textFile("input")

    val wc = in.flatMap(line => {
      line.split(" ")
}).map(word => (word, 1)).cache()

val output = wc.reduceByKey(_ + _)
val output2 = output.sortByKey()

output2.saveAsTextFile("finaloutput")

val o = output2.take(10)
println(o)
}
}