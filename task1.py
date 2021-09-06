from graphframes import *
from pyspark import SparkContext
from pyspark import SparkConf
import sys
import time
from pyspark.sql import SparkSession
from itertools import combinations
import os

def CreateSparkContext():
    # ref: https://spark.apache.org/docs/2.3.0/sql-programming-guide.html#datasets-and-dataframes
    # ref: https://stackoverflow.com/questions/49243719/how-to-access-sparkcontext-from-sparksession-instance
    spark = SparkSession.builder.master('local[3]').appName("task1") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    return spark, sc


if __name__ == '__main__':

    start = time.time()
    os.environ["PYSPARK_SUBMIT_ARGS"] = ("–packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

    spark, sc = CreateSparkContext()
    threshold = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    #print(str(threshold))

    # read input file, remove header
    # group and reduce by userid
    # remove users with ratings less than threshold (reduce size)
    ub_rdd = sc.textFile(input_path)
    header = ub_rdd.first()
    user_busi_rdd = ub_rdd.filter(lambda x: x!=header) \
                        .map(lambda x: (x.split(',')[0], [x.split(',')[1]])) \
                        .reduceByKey(lambda a,b: a+b) \
                        .map(lambda x: (x[0], list(set(x[1])))) \
                        .filter(lambda x: len(x[1]) >= threshold)
    # get the list of users for doing combinations
    user_list = user_busi_rdd.map(lambda x: x[0]).collect()
    # get dict for searching
    user_busi_dict = user_busi_rdd.collectAsMap()

    # for each combination check intersection len >= threshold
    # if true, then save the pair to the list of edges
    # and save each node to the list of vertices
    edges_lst = []
    vertices_lst = []
    for comb in list(combinations(user_list, 2)):
        if len(set(user_busi_dict[comb[0]]).intersection(set(user_busi_dict[comb[1]]))) >= threshold:
            edges_lst.append(tuple([comb[0],comb[1]]))
            # Thoughts from Piazza discussion: save the reverse pair to indicate an undirected graph
            edges_lst.append(tuple([comb[1],comb[0]]))
            # make it tuple! for createDataFrame(), str doesn't work
            vertices_lst.append((comb[0],))
            vertices_lst.append((comb[1],))
    # remove duplicate vertices, no duplicate edges since distinct combs.
    vertices_lst = list(set(vertices_lst))
    #print(edges[:50])
    # create 2 dataframes and make them a graph
    # ref: https://docs.databricks.com/spark/latest/graph-analysis/graphframes/user-guide-python.html
    vertices = spark.createDataFrame(vertices_lst, ["id"])
    edges = spark.createDataFrame(edges_lst, ["src", "dst"])
    g = GraphFrame(vertices, edges)

    # Use LPA given in graphframes
    result = g.labelPropagation(maxIter=5)
    #print(result.collect()[:30]) # Row: (id, label)

    # group and reduce by label
    # sort within communities, sort by size of communities
    communities_lst = result.rdd.map(lambda x: (x[1], [x[0]])) \
                        .reduceByKey(lambda a,b: a+b) \
                        .map(lambda x: sorted(x[1])) \
                        .sortBy(lambda x: (len(x), x)).collect()

    #print(communities_lst[:5])
    with open(output_path, 'w+') as fp:
        for users in communities_lst:
            print_line = "', '".join(users)
            fp.write("'{0}'\n".format(print_line))
    fp.close()
    
    end = time.time()
    print("Duration: "+str(end-start))
