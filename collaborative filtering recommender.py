
import pandas as pd
rating=pd.read_csv('../data/training_ratings_for_kaggle_comp.csv',names=['user_id', 'movie_id', 'ratings'],usecols=range(3))
movies=pd.read_csv('../data/movies.dat',names=['movie_id','title',],sep='::',usecols=range(2))
ratings=pd.merge(movies,rating)
ratings.head()

user_rating=ratings.pivot_table(index=['user_id'],columns=['title'],values=['ratings'])
user_rating.head()

corrmatrix=user_rating.corr()
corrmatrix.head()
corrmatrix=user_rating.corr(method='pearson',min_periods=100)
corrmatrix.head()
myratings=user_rating.loc[0].dropna()
myratings

simcandidate=pd.Series()
for i in range(0,len(myratings.index)):
    sims=corrmatrix[myratings.index[i]].dropna()
    sims=sims.map(lambda x: x*myratings[i])
    simcandidate=simcandidate.append(sims)
    

simcandidate.sort_values(inplace=True,ascending=False)
print(simcandidate.head(10))

simcandidate=simcandidate.groupby(simcandidate.index).sum()
simcandidate.sort_values(inplace=True,ascending=False)
print(simcandidate.head(10))
filtered=simcandidate.drop(myratings.index[1])
filtered=simcandidate.drop(myratings.index[2])
filtered.head(20)