#!/usr/bin/env python
""" \
    Created by Pavlo Shcherbatyi (2023).
    This script processes JSONL file with Github Events body payload data
    At this moment, it processes the data with PushEvent type, 
    extracts all commits' messages and converts separated words
    into so-called N-grams (now it's only 3-grams).
"""
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    lower,
    explode,
    regexp_replace,
    collect_list,
    split,
    flatten,
    size,
    concat_ws,
    expr,
)
from pyspark.ml.feature import NGram
from time import strftime

# Here we define all of input parameters of the application
# Would be actually good to implement this as CLI tool in the future
# TODO: convert the script into CLI tool

eventType = "PushEvent"
inputFilePath = "10K.github.jsonl"
ngramFactor = 3
ngramsColumnName = "ngrams"

# Define executors of our script
ctx = SparkContext.getOrCreate()
spark = (
    SparkSession(ctx)
        .builder.master("local[*]")
        .appName("Laboratory Work #1")
        .getOrCreate()
)
ngram = NGram(n=ngramFactor, inputCol="words", outputCol=ngramsColumnName)

# Extract commits' messages and convert them into collection of
# a key-value pair AuthorName:CommitMessageWords
githubEventsDf = (
    spark.read.json(inputFilePath)
        .filter(f"type = '{eventType}'")
        .select(explode("payload.commits").alias("commit"))
        .select(
            lower("commit.author.name").alias("author"),
            lower("commit.message").alias("message"),
        )
        .withColumn("message", regexp_replace("message", "[^a-zA-Z0-9\\s]", ""))
        .withColumn("message", (split("message", "\\s+")))
        .withColumn("message", expr("filter(message, element -> element != '')"))
        .groupBy("author")
        .agg(flatten(collect_list("message")).alias("words"))
)

# Process words into n-grams
result = (
    ngram.transform(githubEventsDf)
        .select("author", ngramsColumnName)
        .filter(size(ngramsColumnName) > 0)
        .withColumn(ngramsColumnName, concat_ws(", ", ngramsColumnName))
)

# Save the result into CSV file
resultFileName = "ngram-{timestamp}".format(timestamp=strftime("%Y%m%d-%H%M%S"))
result.write.option("header", True).option("delimiter", "|").csv(resultFileName)
