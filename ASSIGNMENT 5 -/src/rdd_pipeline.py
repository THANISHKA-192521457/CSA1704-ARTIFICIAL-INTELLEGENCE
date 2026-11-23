"""PySpark RDD implementation for per-sensor mean/max and ten noisiest sensors."""
def main():
    try:
        from pyspark import SparkConf, SparkContext
    except ImportError:
        print("PySpark is optional. Install with: pip install pyspark")
        print("The report contains the required RDD pseudocode and operation classification.")
        return
    conf=SparkConf().setAppName("AirQualityRDD").setMaster("local[*]")
    sc=SparkContext.getOrCreate(conf)
    # Replace this deterministic sample with sc.textFile("path/to/readings.csv").
    readings=[("S1",1,30.0),("S1",2,40.0),("S2",1,20.0),
              ("S2",2,60.0),("S3",1,10.0),("S3",2,90.0)]
    rdd=sc.parallelize(readings,2)
    # map is narrow; reduceByKey is wide (shuffle); cache reused aggregate.
    keyed=rdd.map(lambda x:(x[0],(x[2],x[2],1)))
    agg=keyed.reduceByKey(lambda a,b:(a[0]+b[0],max(a[1],b[1]),a[2]+b[2])).cache()
    stats=agg.mapValues(lambda x:{"mean":x[0]/x[2],"max":x[1]})
    print("Per-sensor mean/max:",stats.collect()) # action triggers DAG
    # Demonstration of top noisy sensors: use max-minus-mean range as simple variability proxy.
    noisy=agg.mapValues(lambda x:x[1]-x[0]/x[2]).sortBy(lambda kv:kv[1],ascending=False).take(10)
    print("Top noisy sensors:",noisy)
    sc.stop()
if __name__=="__main__": main()
