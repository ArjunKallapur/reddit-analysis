# Analysis of Sentiment Towards President Trump on r/politics Subreddit

Authors: Arjun Kallapur, Tanmaya Hada

## Introduction
This is our solution to Project 2 for COM SCI 143, taken in Spring 2019 with [Professor Ryan Rosario](http://ryanrosar.io/). For this project, we used a subset of the data collected from the politics subreddit [r/politics](https://www.reddit.com/r/politics/) by the data scientist [Jason Baumgartner](https://pushshift.io/). 

## Steps taken
On a high level, this project consisted of the following steps:
* Taking the text from reddit comments and converting them into text that can be used to train our classifier.
* Use Spark SQL to make a data frame
* Train a classifier using Spark MLLib
* Use obtained data to generate the report

## Files in this repository
* [analysis.py](./analysis.py), the file containing our script to generate the required data plots
* [cleantext.py](./cleantext.py), the file containing our script to convert the text from reddit comments and convert them into a usable form for our classifier
* [reddit_model.py](./reddit_model.py), the file containing our classifier model (courtesy Professor Rosario)
* [report.pdf](./report.pdf), containing our data plots and answers to the questions asked in the spec. 

## Conclusions
As detailed in the report, the overall sentiment on the politics subreddit towards President Trump is negative. 
