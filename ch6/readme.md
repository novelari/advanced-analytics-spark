## LSA run requirements

- Download https://mvnrepository.com/artifact/com.databricks/spark-xml_2.11/0.4.1
- move downloaded jar in a common jar directory or create one
- Update conf/spark-defaults.conf
  - spark.driver.extraClassPath    /path/to/jars/*

