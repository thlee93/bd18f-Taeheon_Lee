#install and run
mvn compile exec:java -Dexec.mainClass=edu.snu.bd.examples.Homework \
    -Dexec.args="--runner=SparkRunner --inputFile=input1.csv --inputDirectory=`pwd`/ --output=`pwd`/spark_output/output" \
    -Pspark-runner
