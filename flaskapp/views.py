from flaskapp import app
from flask import render_template, request
import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import urllib.parse
import flaskapp.nutrition_data as nd 

#load stored allrecipes databases
recipedb = pd.read_csv('app_data/allrecipes_nutr_labels.csv')
recipedb = recipedb.set_index('recipename')

@app.route('/')
@app.route('/index')
def index():	
	'''Choose random sample of 10 recipes from stored database and display names on sidebar
	Saves db indices for the random sample so that list is maintained when switching to 
	second page'''
	maindishlist = []
	indices = []
	for num in np.random.choice(len(recipedb),size=10,replace=False):
		maindishlist.append(recipedb.iloc[num].name)
		indices.append(num)
	np.save('app_data/maindish_indices',indices)
	return render_template("index.html",maindishes=maindishlist)

@app.route('/example')
def example():
	maindish_indices = np.load('app_data/maindish_indices_ex.npy')
	maindishlist = recipedb.iloc[maindish_indices].index
	return render_template("index.html",maindishes=maindishlist)

@app.route('/firstchoice')
def firstchoice_displfacts():
	'''Display nutritional facts for selected main dish recipe, show list of recommended alternative recipes'''
	#indices for sample of 10 recipes from db from /index for the sidebar 
	maindish_indices = np.load('app_data/maindish_indices.npy')
	maindishlist = recipedb.iloc[maindish_indices].index
	
	#get name of selected recipe and nutrition info 
	recipename = request.args.get('recipename')
	recipename = recipename.replace('_',' ')
	nutrfacts = recipedb.loc[recipename]

	#save db indice for first choice for switching to recipe comparison pages
	np.save('app_data/firstchoice_ind',np.array([recipedb.index.get_loc(recipename)]))

	#plot nutritional info for selected recipe 
	plot_url = nd.plot_nutrinfo(nutrfacts)

	#get class label for selected recipe 
	classlabel = nd.predict(recipename)

	#get list of recommended dishes
	similar_recipes,better_recipes,best_recipes = nd.recommendations(classlabel,recipename)
	
	similar_recipes.to_csv('app_data/similar_recipes.csv')
	better_recipes.to_csv('app_data/better_recipes.csv')
	best_recipes.to_csv('app_data/best_recipes.csv')
		
	return render_template('firstchoice.html',plot_url=plot_url,maindishes=maindishlist,
		okrecs=similar_recipes.index,betterrecs=better_recipes.index,bestrecs=best_recipes.index)


@app.route('/secondchoice')
def secondchoice_comparefacts():
	'''Compare nutritional facts and ingredients of second and first recipe choice'''
	#indices for sample of 10 recipes from db from /index for sidebar
	maindish_indices = np.load('app_data/maindish_indices.npy')
	maindishlist = recipedb.iloc[maindish_indices].index

	#db index for first choice
	ind_firstchoice = np.load('app_data/firstchoice_ind.npy')
	firstrecipename = recipedb.iloc[ind_firstchoice].index[0]
	recipe_firstchoice_nutrfacts = recipedb.iloc[ind_firstchoice]

	#name of selected second choice recipe
	second_recipename = request.args.get('recipename').replace('_',' ')
	recipe_secondchoice_nutrfacts = recipedb.loc[[second_recipename]]

	#plot nutritional info for both recipes
	plot_url = nd.plot_nutrinfo_comp(recipe_firstchoice_nutrfacts,recipe_secondchoice_nutrfacts)
	
	#retrieve saved recipe rec lists 
	similarlist = pd.read_csv('app_data/similar_recipes.csv').set_index('recipename')
	betterlist = pd.read_csv('app_data/better_recipes.csv')
	bestlist = pd.read_csv('app_data/best_recipes.csv')
	if len(betterlist) > 0:
		betterlist = betterlist.set_index('recipename')
	if len(bestlist) > 0:
		bestlist = bestlist.set_index('recipename')
	reclist = pd.concat([similarlist,betterlist,bestlist])

	#compare health and ingred similarity scores
	firstrecipe_hs = recipedb.loc[firstrecipename,'Health Score']
	secondrecipe_hs = recipedb.loc[second_recipename,'Health Score']
	healthscore = round(100*(secondrecipe_hs-firstrecipe_hs)/abs(firstrecipe_hs),2)
	ingredsc = round(reclist.loc[second_recipename].ingred_similarity,2)

	#ingredient list for first and second recipe
	firstlist,secondlist = nd.return_ingredlist(firstrecipename,second_recipename)
	ingredlist = [firstlist,secondlist]

	return render_template('secondchoice.html',plot_url = plot_url,maindishes=maindishlist,
			okrecs=similarlist.index,betterrecs=betterlist.index,bestrecs=bestlist.index,healthscore=healthscore,ingredsc=ingredsc,ingredlist=ingredlist)

@app.route('/about')
def about():
	return render_template('about.html')
