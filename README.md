# relevance_prediction

In this project, we tackle a problem that involves text mining, natural language processing, feature extraction from text, machine learning and the use of various data mining and information retrieval techniques. 

The particular task is focused on predicting the quality results of a query at the site of Home Depot. Past queries from users were evaluated on their quality from the users themselves. Thus, for each query we have the search query, the result product and the evaluation of that query (with values that range from 1 to 3). We work in two different flavors of the problem. The supervised case, where the ground truth information is available and the unsupervised case, in which there is no ground truth information to use (no labels). 

We utilize Apache Spark with the Scala programming language, in order to parallelize the work and thus make it run more efficiently.

A PDF which describes the problem, our methodology and our results is also included.

Joint work with [Arampatzis Georgios](https://github.com/alfagama)
