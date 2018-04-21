import argparse
import csv
import sys
import pandas as pd
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import stats


#Print results function
def output(command, train, algorithm, k, userid, movieid, prediction): 
	print("myrex.command\t\t= " + command)
	print("myrex.training\t\t= " + train)
	print("myrex.algorithm\t\t= " + algorithm)
	print("myrex.k\t\t\t= " + str(k))
	print("myrex.userid\t\t= " + str(userid))
	print("myrex.movieid\t\t= " + str(movieid))
	print("myrex.prediction\t= " + str(prediction))
	
	return


def compute_weights(sim_weights, movie_rating, norm):
	if k == 0:
		sim_weights = sorted(sim_weights.items(), key=lambda x: x[1], reverse=True)
	else:
		sim_weights = sorted(sim_weights.items(), key=lambda x: x[1], reverse=True)[:k]
	movie_rating = movie_rating.loc[movie_rating['userid'].isin([i[0] for i in sim_weights])]
	predicted_rating = 0.0
	weights_sum = 0.0
	
	usr_count=0
	for rating in movie_rating.iterrows():
		weight = [i[1] for i in sim_weights][usr_count]
		usr_weight = [i[0] for i in sim_weights][usr_count]
		rate = movie_rating.loc[movie_rating["userid"]==usr_weight]["rating"].values[0]
		if norm:
			rate = (2*(rate - 1) - 4) / 4
		predicted_rating += rate * weight
		weights_sum += weight
		usr_count+=1
	predicted_rating /= weights_sum
	if norm:
		predicted_rating =  0.5 * (predicted_rating + 1)*4 + 1
	output(command, training_file, algorithm, k, user_id, movie_id, predicted_rating)
	
	

	
def predict():
	df = pd.read_csv(training_file, delim_whitespace=True, header=None, dtype={'ID': object}, names=["userid", "movieid", "rating", "timestamp"])

	# user id | item id | rating | timestamp
	df_user = df.loc[df['userid'] == user_id][["rating", "movieid"]]
	movie_rating = df.loc[df['movieid'] == movie_id]

	if algorithm=="average":
		df = df.loc[df['movieid'] == movie_id]
		predicted_rating = df["rating"].mean()
		output(command, training_file, algorithm, k, user_id, movie_id, predicted_rating)
		
	elif algorithm=="euclid":
		sim_weights = {}
		norm = False
		for i in movie_rating["userid"]:
			if i!=user_id:
				df_other = df.loc[(df['userid'] == i)][["rating", "movieid"]]
				df_both = pd.merge(df_user, df_other, on='movieid', how='inner', suffixes=(user_id, i))
				df_both = df_both.dropna()
				dist = euclidean(df_both.loc[:,("rating" + str(user_id))].values.ravel(), df_both.loc[:,("rating" + str(i))].values.ravel())
				sim_weights[i] = 1.0 / (1.0 + dist)
			
		compute_weights(sim_weights, movie_rating, norm)

		
	elif algorithm=="pearson":
		sim_weights = {}
		norm = True
		df_user = df.loc[df['userid'] == user_id][["rating", "movieid"]]
		
		#Normalize target user ratings
		df_user["rating"] = (2*(df_user["rating"] - 1) - 4) / 4
		print(df_user)
		movie_rating = df.loc[df['movieid'] == args.movie_id]
		
		usr_count=0
		for j in movie_rating["userid"]:	
			if j!=user_id:
				df_other = df.loc[(df['userid'] == j)][["rating", "movieid"]]
				df_other["rating"] = (2*(df_other["rating"] - 1) - 4) / 4
				df_both = pd.merge(df_user, df_other, on='movieid', how='inner', suffixes=(args.user_id, j))
				df_both = df_both.fillna(0)
				dist = stats.pearsonr(df_both.loc[:,("rating" + str(user_id))].values.ravel(), df_both.loc[:,("rating" + str(j))].values.ravel())
				sim_weights[j] = 1.0 / (1.0 + dist[0])	
		compute_weights(sim_weights, movie_rating, norm)	
		
		
	elif algorithm=="cosine":
		sim_weights = {}
		norm = True
		df_user = df.loc[df['userid'] == user_id][["rating", "movieid"]]
		
		#Normalize target user ratings
		df_user["rating"] = (2*(df_user["rating"] - 1) - 4) / 4
		movie_rating = df.loc[df['movieid'] == movie_id]
		
		usr_count=0
		for j in movie_rating["userid"]:	
			if j!=user_id:
				df_other = df.loc[(df['userid'] == j)][["rating", "movieid"]]
				df_other["rating"] = (2*(df_other["rating"] - 1) - 4) / 4
				df_both = pd.merge(df_user, df_other, on='movieid', how='inner', suffixes=(args.user_id, j))
				df_both = df_both.fillna(0)
				dist = cosine(df_both.loc[:,("rating" + str(user_id))].values.ravel(), df_both.loc[:,("rating" + str(j))].values.ravel())
				sim_weights[j] = 1.0 / (1.0 + dist)	
				
		compute_weights(sim_weights, movie_rating, norm)	


def evaluate():
	print("hi")
	

	
#Get input .csv file, px, and results filename

if sys.argv[1]=="predict" and len(sys.argv)!=7:
	print("Must supply 5 arguments for predict command\n")
	sys.exit()
elif sys.argv[1]=="evaluate" and len(sys.argv)!=6:
	print("Must supply 6 arguments for evaluate command\n")
	sys.exit()
elif sys.argv[1]=="predict" and len(sys.argv)==7:	
	command = sys.argv[1]
	training_file = sys.argv[2]
	k = int(sys.argv[3])
	algorithm = sys.argv[4]
	user_id = int(sys.argv[5])
	movie_id = int(sys.argv[6])
	predict()
elif sys.argv[1]=="evaluate" and len(sys.argv)==6:	
	command = sys.argv[1]
	training_file = sys.argv[2]
	k = int(sys.argv[3])
	algorithm = sys.argv[4]
	test_file = sys.argv[5]
	evaluate()
elif int(sys.argv[3]) < 0:
	print("k must be positive")
	sys.exit()
