#data_preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("50_Startups.csv")
X1=dataset.iloc[:, :-1].values #var independante iloc (fonction de pd) vas recuperer les indices dont on aura besoin (: pour toute et :-1 pour toute sauf la dernière) 
y=dataset.iloc[:, -1].values # -1 pour la dernière


stat=dataset.describe() # mini description statistique d'une BD


#gerer les variables categoriques
""""l'orsque les var sont nominale(pas d''existance d'une relation d'ordre) il 
faut effectuer un encodoge par demi variable ou le OneHotEncoder qui 
consiste a creer pour chaque modalite des colones supplementaires"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelEncoder_X=LabelEncoder()
#X[:,0]=labelEncoder_X.fit_transform(X[:,0])

# Extraire la colonne de la variable catégorielle
var_cat= X1[:, 3]
# Redimensionner l'array de la variable catégorielle pour avoir la forme (n_samples, 1)
var_cat_reshaped = var_cat.reshape(-1, 1)
# Créer une instance de l'encodeur OneHotEncoder
encoder = OneHotEncoder()
# Appliquer l'encodage OneHotEncoder à la variable catégorielle
encoded_data = encoder.fit_transform(var_cat_reshaped).toarray()

result = np.concatenate((encoded_data,X1), axis=1)
X= np.delete(result, 6, axis=1)# est ma nouvelle matrice prette a être utilisé

"""pour eviter le problème de multicolinearité supprimons une colonne de la 
matrice issu de la transformation de la var_categoriel""" 
X=X[:,1:]#toute les colonnes sauf la première (notament suprime la col 0)

# division du dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)


#feature scaling ??? pas besoin d'en faire ici en reg_lin_mul

#construction du mdoel
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# nouvelle prediction
y_pred=model.predict(X_test)

#pour la prediction d'une valeur souhaité on fera
model.predict(np.array([[1, 0,142107.34,	91391.77,	366168.42]]))




























