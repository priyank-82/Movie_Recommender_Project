import numpy as np 
import pickle
import streamlit as st
import pandas as pd
import requests


movies=pd.read_csv('C:\codes\PRML Project\movies (1).csv')
ratings=pd.read_csv('C:/codes/PRML Project/user_ratings.csv')

movies['title'] = movies['title'].str.strip().str[:-7]
#removing the | from the genre and replacing it by space
movies['genres']=movies['genres'].str.replace('|', ' ')

from sklearn.feature_extraction.text import CountVectorizer
#making an obejct of it 
cv=CountVectorizer()
# we have a token in the genres so we will now make tokens out of it 
genres_tokens=cv.fit_transform(movies['genres'].values)

genres_features=cv.get_feature_names_out()

genres_tokens=pd.DataFrame(genres_tokens.toarray(),columns=genres_features.tolist())

genres_tokens['combined']=genres_tokens.values.tolist()

movies['genres']=genres_tokens['combined']

#preprocessing of rating column 

pivot_mat = ratings.pivot(index='movieId',columns='userId',values='rating')

pivot_mat.fillna(0,inplace=True)

vote_movie = [[],[]]
user_votes = [[],[]]
sh = pivot_mat.shape
for i in range(sh[0]):
  r,c = np.unique(pivot_mat.values[i],return_counts=True)
  user_votes[0].append(np.sum(c[1:]))
  user_votes[1].append(pivot_mat.index[i])
for i in range(sh[1]):
  r,c = np.unique(pivot_mat.values[:,i],return_counts=True)
  vote_movie[0].append(np.sum(c[1:]))
  vote_movie[1].append(i+1)


vote_movie = np.array(vote_movie).T
user_votes = np.array(user_votes).T

pivot_mat = pivot_mat.loc[user_votes[:,1][user_votes[:,0] > 10],:]

zc = 0
for i in range(pivot_mat.shape[0]):
  for j in range(pivot_mat.shape[1]):
    if pivot_mat.iloc[i,j] == 0:
      zc+=1


from scipy.sparse import csr_matrix
csr_data = csr_matrix(pivot_mat.values)
pivot_mat.reset_index(inplace=True)



def dist_rec(movie_name,rec):
  try:
    arr = np.array(movies[movies['title'] == movie_name].values[0][2])
  except:
    return "Movie not found"
  
  mov = movies[movies['title'] != movie_name].values
  dis = []
  recommendations=[]
  for i in mov:
    dis.append(np.sqrt((np.sum((np.array(i[2]) - arr)**2)))) # similar to the K-means clustering decision.
  # print("The Recommendations for " + movie_name + " are :\n")
  # for i in range(rec):
  #   print(mov[:,1][np.argmin(dis)])
  #   dis[np.argmin(dis)] = 9999
  for i in range(rec):
    recommendations.append(mov[:,1][np.argmin(dis)])
    dis[np.argmin(dis)] = 9999
  return recommendations


#defining the K-means clustering decision
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

def knn_reccomendation(movie_name,n_movies_to_reccomend):
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = pivot_mat[pivot_mat['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = pivot_mat.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        return "No movies found. Please check your input"

st.header('Movie Recommender System')

option = st.selectbox(
     'Which model would you like to use?',
     ('Genre based', 'KNN-based'))

selected_movie = st.text_input(
    "Type a movie name to get recommendations"
)

number_of_recommendations = st.number_input(
      "Type the number of recommendations to get"
)

if st.button('Show Recommendations'):

  if option=='Genre based':
        movie_recommendations = dist_rec(selected_movie,int(number_of_recommendations))

        st.text(f"Here are {number_of_recommendations} recommendations for {selected_movie}")

        for i in range(int(number_of_recommendations)):
          st.text(f"{i+1}. {movie_recommendations[i]}")

  elif option=='KNN-based':
        movie_recommendations = knn_reccomendation(selected_movie,int(number_of_recommendations))

        st.text(f"Here are {number_of_recommendations} recommendations for {selected_movie}")

        for i in movie_recommendations['Title']:
          st.text(i)



