import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import urllib.parse
import seaborn
from scipy.spatial.distance import pdist, squareform
from sklearn.externals import joblib
import spacy

def plot_nutrinfo(nutrfacts):
	'''plot 18 nutrient's % daily values for selected recipe'''

	nutrfacts = nutrfacts*100#remove non-nutrient columns and convert to %
	nutrfacts = nutrfacts.reindex(index=['Total Fat','Saturated Fat','Cholesterol','Sodium','Total Carbohydrates','Sugars',
		'Protein','Dietary Fiber','Vitamin A','Vitamin C','Calcium','Iron','Thiamin','Niacin','Vitamin B6','Magnesium','Folate'])
	img = io.BytesIO()

	plt.figure(figsize=(10,15))
	g = seaborn.barplot(y=nutrfacts.index,x=nutrfacts.values,color='k',alpha=0.3)
	ax=plt.gca()
	ax.tick_params(labelsize=20)
	plt.xlabel('% Daily Value',fontsize=20)
	plt.ylabel('')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_linewidth(3)
	ax.spines['left'].set_linewidth(3)
	ax.axvline(100,linewidth=4,ls=':')
	
	#change x labels to red for nutrients to limit, and green for nutrients to get enough of
	[i.set_color('r') for i in ax.get_yticklabels() if i.get_text() in ['Total Fat','Saturated Fat',
		'Cholesterol','Sodium','Total Carbohydrates','Sugars']]
	[i.set_color('g') for i in ax.get_yticklabels() if i.get_text() in ['Potassium','Dietary Fiber',
		'Protein','Vitamin A','Vitamin C','Calcium','Iron']]

	plt.tight_layout()
	plt.savefig(img,format='png')
	img.seek(0)
	plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())
	
	return plot_url

def predict(recipename):
	'''use the saved GMM model to predict class label for selected recipe'''
	#load recipe db that has been sqrt transformed
	db = pd.read_csv('app_data/db_nutr_sqrt.csv').set_index('recipename')
	db = db.iloc[:,:9]

	#load standard scale, pca, gmm used during training
	pca = joblib.load('app_data/distpca.pkl')
	standardscale = joblib.load('app_data/standardscaler.pkl')
	mdl = joblib.load('app_data/gmm4.pkl')

	nutrfacts = db.loc[recipename]
	nutrfacts = np.array(nutrfacts,ndmin=2)
	nutrfacts = standardscale.transform(nutrfacts)

	db_transformed = standardscale.transform(db.values) 
	db_dist = squareform(pdist(np.append(db_transformed,nutrfacts,axis=0)))
	label = mdl.predict(pca.transform(np.array(db_dist[-1,:-1],ndmin=2)))
	
	return label

def betterclasses(classlabel):
	'''given class label of specific recipe, return three indexes of recipenames in the 
	same, healthier, and even healthier group'''
	
	#load class labels for db and recipe db
	db = pd.read_csv('app_data/allrecipes_nutr_labels.csv').set_index('recipename')
	db_labels = db['labels']

	#ranking of classlabels from least to most healthy
	label_ranks = np.array([2,3,1,0])
	
	#get class labels that rank better than selected recipe's label
	classlabelind = np.where(label_ranks == classlabel)[0][0]
	betterclasslabels = label_ranks[classlabelind:]
	
	#get recipenames for similar, better, best health clusters
	similar = db[(db.labels==betterclasslabels[0])&(db.class_prob>=0.85)].index
	if len(betterclasslabels) > 1:
		better = db[(db.labels==betterclasslabels[1])&(db.class_prob>=0.85)].index
	else:
		better = []
	if len(betterclasslabels) > 2:
		best = db[(db.labels==betterclasslabels[2])&(db.class_prob>=0.85)].index
	else:
		best = []
	
	return similar, better, best

def recommendations(classlabel,recipename):
	'''given class label of specific recipe and recipe name, return list of recipes in the 
	same, healthier, and even healthier groups, ranked by ingredient similarity'''
	
	#names of recipes in a simnilar, better, or even better health class
	similar,better,best = betterclasses(classlabel)

	#load word vector model used to filter recipes by ingred similarity
	nlp = spacy.load('app_data/recipe_ingred_word2vec_lg')

	#load ingred db (cleaned)
	ingredsdb = pd.read_csv('app_data/ingreds_db_cleaned.csv').rename(columns={'Unnamed: 0':'recipename'})
	ingredsdb = ingredsdb.set_index('recipename')
	
	#load recipe db 
	recipedb = pd.read_csv('app_data/allrecipes_nutr_labels.csv').set_index('recipename')

	#get similarity scores to target recipe
	targetingred = nlp(ingredsdb.loc[recipename].ingredients)
	similarity = []
	for num in range(0,len(similar)):
		similarity.append(targetingred.similarity(nlp(ingredsdb.loc[similar[num]].ingredients)))
	similarity = np.array(similarity)
	similar = similar[(-similarity).argsort()[:10]]
	similarity = similarity[(-similarity).argsort()[:10]]
	similar_recipes = recipedb.loc[similar]
	similar_recipes['ingred_similarity'] = similarity

	if len(better) > 0:
		similarity = []
		for num in range(0,len(better)):
			similarity.append(targetingred.similarity(nlp(ingredsdb.loc[better[num]].ingredients)))
		similarity = np.array(similarity)
		better = better[(-similarity).argsort()[:10]]
		similarity = similarity[(-similarity).argsort()[:10]]
		better_recipes = recipedb.loc[better]
		better_recipes['ingred_similarity'] = similarity
	else:
		better_recipes = recipedb.loc[better]

	if len(best) > 0:
		similarity = []
		for num in range(0,len(best)):
			similarity.append(targetingred.similarity(nlp(ingredsdb.loc[best[num]].ingredients)))
		similarity = np.array(similarity)
		best = best[(-similarity).argsort()[:10]]
		similarity = similarity[(-similarity).argsort()[:10]]
		best_recipes = recipedb.loc[best]
		best_recipes['ingred_similarity'] = similarity
	else:
		best_recipes = recipedb.loc[best]

	return similar_recipes,better_recipes, best_recipes

def plot_nutrinfo_comp(firstnutrfacts,secnutrfacts):
	'''plot comparison of nutritional info for two recipes'''
	#combine nutr info and change df to long form 	
	combined_nutrfacts = (pd.concat([firstnutrfacts,secnutrfacts]).iloc[:,:-5]*100).T.reindex(index=['Total Fat',
	'Saturated Fat','Cholesterol','Sodium','Total Carbohydrates','Sugars','Protein','Dietary Fiber','Vitamin A','Vitamin C','Calcium',
	'Iron','Thiamin','Niacin','Vitamin B6','Magnesium','Folate'])
	combined_nutrfacts = combined_nutrfacts.reset_index().melt(id_vars=['index'])
	combined_nutrfacts = combined_nutrfacts.rename(columns={'index':'nutrients'})


	img = io.BytesIO()
	plt.figure(figsize=(10,15))
	g = seaborn.barplot(y="nutrients",x="value",hue="recipename",data=combined_nutrfacts,
		palette=['k','r'],alpha=0.3)
	ax=plt.gca()
	plt.xlabel('% Daily Value',fontsize=20)
	plt.ylabel('')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_linewidth(3)
	ax.spines['left'].set_linewidth(3)
	ax.axvline(100,lw=4,ls=':')
	ax.tick_params(labelsize=20)
	ax.set_yticklabels(labels=combined_nutrfacts['nutrients'])
	ax.legend(loc=1,frameon=False,fontsize=15)
	
	[i.set_color('r') for i in ax.get_yticklabels() if i.get_text() in ['Total Fat','Saturated Fat',
		'Cholesterol','Sodium','Total Carbohydrates','Sugars']]
	[i.set_color('g') for i in ax.get_yticklabels() if i.get_text() in ['Potassium','Dietary Fiber',
		'Protein','Vitamin A','Vitamin C','Calcium','Iron']]
	
	plt.tight_layout()
	plt.savefig(img,format='png')
	img.seek(0)
	plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())

	return plot_url

def return_ingredlist(firstname,secondname):
	'''list of ingredients for first and second recipe'''
	ingreddb = pd.read_csv('app_data/allrecipes_ingreds_db_all.csv').set_index('recipename')
	firstlist = [i for i in ingreddb.loc[firstname,'ingredients']]
	secondlist = [i for i in ingreddb.loc[secondname,'ingredients']]

	return firstlist, secondlist

	
