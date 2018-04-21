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
	sim_weights = sorted(sim_weights.items(), key=lambda x: x[1], reverse=True)[:args.k]
	#print(sim_weights)
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
	output(args.command, args.training_file, args.algorithm, args.k, args.user_id, args.movie_id, predicted_rating)
	
	
#Get input .csv file, px, and results filename
parser = argparse.ArgumentParser()
parser.add_argument("command", nargs='?', help="command")
parser.add_argument("training_file", nargs='?', help="training file")
parser.add_argument("k", nargs='?', type=int, help="K nearest users")
parser.add_argument("algorithm", nargs='?', help="Algorithm")
parser.add_argument("user_id", nargs='?', type=int, help="user ID")
parser.add_argument("movie_id", nargs='?', type=int, help="movie ID")


args = parser.parse_args()
if args.command=="predict" and len(vars(args))!=6:
	print("Must supply 5 arguments for predict command\n")
	sys.exit()
elif args.command=="evaluate" and len(vars(args))!=6:
	
elif args.command=="predict" and len(vars(args))==6:	
	df = pd.read_csv(args.training_file, delim_whitespace=True, header=None, dtype={'ID': object}, names=["userid", "movieid", "rating", "timestamp"])
elif args.k < 0:
	print("k must be positive")
	sys.exit()
	
# user id | item id | rating | timestamp
df_user = df.loc[df['userid'] == args.user_id][["rating", "movieid"]]
movie_rating = df.loc[df['movieid'] == args.movie_id]


if args.algorithm=="average":
	df = df.loc[df['movieid'] == args.movie_id]
	predicted_rating = df["rating"].mean()
	output(args.command, args.training_file, args.algorithm, args.k, args.user_id, args.movie_id, predicted_rating)
	
elif args.algorithm=="euclid":
	sim_weights = {}
	norm = False
	for i in movie_rating["userid"]:
		if i!=args.user_id:
			df_other = df.loc[(df['userid'] == i)][["rating", "movieid"]]
			df_both = pd.merge(df_user, df_other, on='movieid', how='inner', suffixes=(args.user_id, i))
			df_both = df_both.dropna()
			dist = euclidean(df_both.loc[:,("rating" + str(args.user_id))].values.ravel(), df_both.loc[:,("rating" + str(i))].values.ravel())
			sim_weights[i] = 1.0 / (1.0 + dist)
		
	compute_weights(sim_weights, movie_rating, norm)

elif args.algorithm=="pearson":
	sim_weights = {}
	norm = True
	df_user = df.loc[df['userid'] == args.user_id][["rating", "movieid"]]
	#Normalize target user ratings
	df_user["rating"] = (2*(df_user["rating"] - 1) - 4) / 4
	print(df_user)
	movie_rating = df.loc[df['movieid'] == args.movie_id]
	
	usr_count=0
	for j in movie_rating["userid"]:	
		if j!=args.user_id:
			df_other = df.loc[(df['userid'] == j)][["rating", "movieid"]]
			df_other["rating"] = (2*(df_other["rating"] - 1) - 4) / 4
			df_both = pd.merge(df_user, df_other, on='movieid', how='inner', suffixes=(args.user_id, j))
			df_both = df_both.fillna(0)
			dist = stats.pearsonr(df_both.loc[:,("rating" + str(args.user_id))].values.ravel(), df_both.loc[:,("rating" + str(j))].values.ravel())
			sim_weights[j] = 1.0 / (1.0 + dist[0])	
	compute_weights(sim_weights, movie_rating, norm)	
	
elif args.algorithm=="cosine":
	sim_weights = {}
	norm = True
	df_user = df.loc[df['userid'] == args.user_id][["rating", "movieid"]]
	#Normalize target user ratings
	df_user["rating"] = (2*(df_user["rating"] - 1) - 4) / 4
	movie_rating = df.loc[df['movieid'] == args.movie_id]
	
	usr_count=0
	for j in movie_rating["userid"]:	
		if j!=args.user_id:
			df_other = df.loc[(df['userid'] == j)][["rating", "movieid"]]
			df_other["rating"] = (2*(df_other["rating"] - 1) - 4) / 4
			df_both = pd.merge(df_user, df_other, on='movieid', how='inner', suffixes=(args.user_id, j))
			df_both = df_both.fillna(0)
			dist = cosine(df_both.loc[:,("rating" + str(args.user_id))].values.ravel(), df_both.loc[:,("rating" + str(j))].values.ravel())
			sim_weights[j] = 1.0 / (1.0 + dist)	
	compute_weights(sim_weights, movie_rating, norm)	


	

	
	
	
