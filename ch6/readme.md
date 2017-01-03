## LSA run requirements

- Download https://mvnrepository.com/artifact/com.databricks/spark-xml_2.11/0.4.1
- move downloaded jar in a common jar directory or create one
- Update conf/spark-defaults.conf
  - spark.driver.extraClassPath    /path/to/jars/*
- wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
- unzip stanford-corenlp-full-2015-04-20.zip
- git clone https://github.com/brendano/stanford_corenlp_pywrapper

***Note:*** Stanford CoreNLP python wrapper only works on Unix (Linux and Mac). Does not currently work on Windows.
