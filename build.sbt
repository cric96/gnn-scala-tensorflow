name := "gnn-jvm-pof"

version := "0.1"

scalaVersion := "2.12.6"

val tensorflowVersion = "0.4.1"

libraryDependencies ++= Seq(
  //"org.platanios" %% "tensorflow" % tensorflowVersion,
  "org.platanios" %% "tensorflow" % tensorflowVersion classifier "linux-cpu-x86_64"
)
