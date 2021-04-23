from generate_vocabulary import generate_vocabulary
from gather_20newsgroups_data import gather_20newsgroups_data
from get_tf_idf import get_tf_idf

gather_20newsgroups_data()
generate_vocabulary('../datasets/20news-bydate/20news-train-processed.txt')
get_tf_idf('../datasets/20news-bydate/20news-train-processed.txt')
