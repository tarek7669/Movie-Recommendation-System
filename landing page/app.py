from flask import Flask, render_template, request
import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, mean_squared_error
import tabulate
import utils as utils


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Input, Lambda
from tensorflow.keras.layers import Embedding, Flatten, dot
from tensorflow.keras import regularizers
from tensorflow.keras.losses import mse, binary_crossentropy


app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")


@app.route('/prediction', methods=["POST"])
def prediction():

	model = load_model('new_user_model.h5', compile=False)


	# Load Data, set configuration variables
	item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = utils.load_data()

	u_s = 3 # start of columns to use in training, user
	i_s = 1  # start of columns to use in training, items


	scalerItem = StandardScaler()
	scalerItem.fit(item_train)

	scalerUser = StandardScaler()
	scalerUser.fit(user_train)

	scalerTarget = MinMaxScaler((-1, 1))
	scalerTarget.fit(y_train.reshape(-1, 1))


	new_user_id = 5000
	new_rating_ave = 0.0
	action = request.form['action']
	adventure = request.form['adventure']
	animation = request.form['animation']
	children = request.form['children']
	comedy = request.form['comedy']
	crime = request.form['crime']
	documentary = request.form['documentary']
	drama = request.form['drama']
	fantasy = request.form['fantasy']
	horror = request.form['horror']
	mystery = request.form['mystery']
	romance = request.form['romance']
	scifi = request.form['scifi']
	thriller = request.form['thriller']
	new_rating_count = 3

	user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      action, adventure, animation, children,
                      comedy, crime, documentary,
                      drama, fantasy, horror, mystery,
                      romance, scifi, thriller]])



	# generate and replicate the user vector to match the number movies in the data set.
	user_vecs = utils.gen_user_vecs(user_vec,len(item_vecs))

	# scale our user and item vectors
	suser_vecs = scalerUser.transform(user_vecs)
	sitem_vecs = scalerItem.transform(item_vecs)

	# make a prediction
	y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

	# unscale y prediction
	y_pu = scalerTarget.inverse_transform(y_p)

	# sort the results, highest prediction first
	sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
	sorted_ypu   = y_pu[sorted_index]
	sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display

	# model.save('/content/drive/MyDrive/coursera/Unsupervised Learning/new_user_model.h5')

	data = utils.print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount = 10)

	return render_template("prediction.html", data=data)




# def gen_user_vecs(user_vec, num_items):
#     """ given a user vector return:
#         user predict maxtrix to match the size of item_vecs """
#     user_vecs = np.tile(user_vec, (num_items, 1))
#     return user_vecs


# def print_pred_movies(y_p, item, movie_dict, maxcount=10):
#     """ print results of prediction of a new user. inputs are expected to be in
#         sorted order, unscaled. """
#     count = 0
#     disp = [["y_p", "movie id", "rating ave", "title", "genres"]]

#     for i in range(0, y_p.shape[0]):
#         if count == maxcount:
#             break
#         count += 1
#         movie_id = item[i, 0].astype(int)
#         disp.append([np.around(y_p[i, 0], 1), item[i, 0].astype(int), np.around(item[i, 2].astype(float), 1),
#                      movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])

#     table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
#     return table

if __name__=="__main__":
	app.run(debug=True)