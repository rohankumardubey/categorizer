import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
embed = hub.Module(module_url)

def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  sns.set(font_scale=1.2)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")


def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
  message_embeddings_ = session_.run(
      encoding_tensor, feed_dict={input_tensor_: messages_})
  plot_similarity(messages_, message_embeddings_, 90)

import pandas
import scipy
import math


def load_sts_dataset(filename):
  # Loads a subset of the STS dataset into a DataFrame. In particular both
  # sentences and their human rated similarity score.
  sent_pairs = []
  with tf.gfile.GFile(filename, "r") as f:
    for line in f:
      ts = line.strip().split(",")
      # (sent_1, sent_2, similarity_score)
      sent_pairs.append(ts[0])
  return pandas.DataFrame(sent_pairs, columns=["sent_1"])


def download_and_load_sts_data():
  sts_dataset = "/home/sgoyal/pdl/tf/final/" 

  sts_dev = load_sts_dataset(
      os.path.join(os.path.dirname(sts_dataset), "orders2000.csv"))
  sts_test = load_sts_dataset(
      os.path.join(
          os.path.dirname(sts_dataset), "categ.csv"))

  return sts_dev, sts_test


sts_dev, sts_test = download_and_load_sts_data()
sts_input1 = tf.placeholder(tf.string, shape=(None))
sts_input2 = tf.placeholder(tf.string, shape=(None))

# For evaluation we use exactly normalized rather than
# approximately normalized.
sts_encode1 = tf.nn.l2_normalize(embed(sts_input1), axis=1)
sts_encode2 = tf.nn.l2_normalize(embed(sts_input2), axis=1)
#cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
distanceCos = tf.matmul(sts_encode1, sts_encode2, transpose_b=True)
sim_scores = 1.0 - tf.acos(distanceCos)
categ, indices = tf.math.top_k(sim_scores, 4)
#vals = tf.unstack(categ)

text_a = sts_dev['sent_1'].tolist()
text_b = sts_test['sent_1'].tolist()
#dev_scores = sts_data['sim'].tolist()
def run_sts_benchmark(session):
  """Returns the similarity scores"""
  emba, embb, scores, indices1 = session.run(
      [sts_encode1, sts_encode2, categ, indices],
      feed_dict={
          sts_input1: text_a,
          sts_input2: text_b
      })
  #print (text_a[count], text_b[t[0]])
  return scores, indices1

  
with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  scores, indices, distance = run_sts_benchmark(session)
  count = 0
  for t in scores:
    print (count, ";", text_a[count], ";", text_b[indices[count,0]], ";" ,text_b[indices[count,1]], ";" , text_b[indices[count,2]] , ";" , text_b[indices[count,3]])
    #print (count, t[0], t[1])
    count = count +1 
  #tf.io.write_file(
  #  "filename",
  #  scores,
  #  name=None
  #)	
  #session.run(sts_encode1, feed_dict={sts_input1: text_a}) 
  #session.run(sts_encode2, feed_dict={sts_input2: text_b}) 
  #session.run(distance) 
  
  #print(scores)
