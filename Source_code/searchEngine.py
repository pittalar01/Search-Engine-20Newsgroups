import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy as sp
import nltk.stem
all_data = sklearn.datasets.fetch_20newsgroups(subset='all')
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
# ignoring noisy data
vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english', decode_error='ignore')
vectorized = vectorizer.fit_transform(all_data.data)
num_samples, num_features = vectorized.shape
num_clusters = 50
# performing Kmeans clustering for k=50
km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1, random_state=3)
km.fit(vectorized)
#Problem 1
#new_post = "I've heard that the Saturn V rocket used in the Apollo program was the largest " \
#          "rocket ever launched.   Is this true?  Were there any larger rockets ever " \
#           "designed or built, but never launched?   What was the second-biggest rocket? " \
#           "Thank you in advance for any information you have. "
# Problem 2
# new_post = "Did Bo Jackson play for the Kansas City Royals or the Cleveland Indians?"
# Problem 3
new_post = 'Which came first, the Ford Mustang or the Chevy Camaro? What other early pony cars were there? ' \
           'And why on earth are they called "pony cars"?  That seems a silly name. '
# vectorizing the new post to find relevant posts
new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]
similar_indices = (km.labels_ == new_post_label).nonzero()[0]
similar = []
# calculating the similarity scores
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
    similar.append((dist, all_data.data[i]))
similar = sorted(similar)
#printing relevant datasets
show1 = similar[0]
show2 = similar[1]
show3 = similar[2]
show4 = similar[3]
show5 = similar[4]
print('File1:', show1, '\n')
print('File2:', show2, '\n')
print('File3:', show3, '\n')
print('File4:', show4, '\n')
print('File5:', show5, '\n')
