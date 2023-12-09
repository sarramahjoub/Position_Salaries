#!/usr/bin/env python
# coding: utf-8

# In[17]:


#Regression polynomiale
# 1. Importer les librairies
#La manipulation des objets comme les tables
import numpy as np
# les representations graphiques
import matplotlib.pyplot as plt
# la manipulation des fichiers
import pandas as pd
#la regression lineaire
from sklearn.linear_model import LinearRegression
# pour la division des données
from sklearn.model_selection import train_test_split
#Pour l'utilisation des polynomes
from sklearn.preprocessing import PolynomialFeatures
#r au carré r squared mesure d'evaluation pour la regression lineaire
from sklearn.metrics import r2_score


# In[18]:


#2. Importer des données

dataset = pd.read_csv('D:/Position_Salaries.csv')
dataset


# In[19]:


#3. Présenter graphiquement les données
plt.scatter(dataset['Position'],dataset['Salary'],color='red')
plt.show()


# In[20]:


#3. Présenter graphiquement les données
plt.scatter(dataset['Level'],dataset['Salary'],color='red')
plt.show()


# In[21]:


#4. Séparer la variable de décision de la variable cible
x = dataset.iloc [ : , 1:2 ].values
x


# In[22]:


y = dataset.iloc[:, -1].values
y


# In[23]:


#5. Diviser les données entre données de test et données d’apprentissage
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[24]:


#6. Comparer la longueur de x_train et y_train - x_test et y_test
len (x_train)


# In[25]:


len(x_test)


# In[26]:


len(y_train)


# In[27]:


len(y_test)


# In[28]:


7. #Créer un modele de régression linéaire à partir des données d’apprentissage
lin_reg = LinearRegression()


# In[29]:


lin_reg.fit(x_train,y_train)


# In[83]:


#8. Appliquer le modele crée sur les données de test
y_pred = lin_reg.predict(x)


# In[84]:


len(y_pred)


# In[85]:


#9. Comparer les valeurs prédites avec les valeurs réelles


# In[86]:


y_pred


# In[87]:


y_test


# In[88]:


#10. Visualiser la régression linéaire


# In[89]:


plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('La regression linéaire')
plt.xlabel('Le Niveau du poste')
plt.ylabel('Le Salaire')
plt.show()


# In[90]:


#11. Evaluer la performance d'une régression linéaire avec la mesure R-Squared
#12. Evaluer la capacité du modele à apprendre les données
r2= r2_score(y_train,lin_reg.predict(x_train))
r2


# In[91]:


#13. Evaluer la capacité du modele à généraliser
r2= r2_score(y_test,y_pred)
r2


# In[92]:


#14. Appliquer la régression polynomiale de degré 4

# creer un polynome de degré 4
poly_reg = PolynomialFeatures(degree = 8)

# Transformation des données (données polynomiales)
x_poly = poly_reg.fit_transform(x)

#creer une regression lineaire appliquée sur des données polynomiales

lin_reg_2 = LinearRegression ()

# apprentissage du modele
lin_reg_2.fit (x_poly , y)

# Prediction
y_poly_pred = lin_reg_2.predict(x_poly)


# In[93]:


#15. Visualiser la régression polynomiale


# In[94]:


#afficher les données
plt.scatter (x, y , color ='red')
#afficher la regression polynimiale
plt.plot(x, y_poly_pred  ,color='blue')


# In[95]:


x_grid =np.arange (min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))

# Transformation des données (données polynomiales)
x_grid_poly = poly_reg.fit_transform(x_grid)

# Prediction
y_grid_poly_pred = lin_reg_2.predict(x_grid_poly)


# In[96]:


#afficher les données
plt.scatter (x, y , color ='red')
#afficher la regression polynimiale
plt.plot(x_grid, y_grid_poly_pred ,color='blue')


# In[97]:


lin_reg_2.predict(poly_reg.fit_transform([[7.5]]))


# In[98]:


# evaluation de laregression lineaire simple
score=r2_score(y, y_pred)
score


# In[99]:


# evaluation de laregression polynomiale de degré 8
score=r2_score(y, y_poly_pred)
score


# In[ ]:





# In[ ]:





# In[ ]:




