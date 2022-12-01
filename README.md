# 📗 Predicting Readability of Texts Using Machine Learning

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

```diff
+  The dataset was taken from https://www.kaggle.com/c/commonlitreadabilityprize/data. 🙂
```

## Problem Statement 
When we look at the present age, we see a __lot of articles__ and __reading materials__ available by different __authors__ and __bloggers__. As a result, we see text all around us and there is a lot to do with the text. Since there are more and more books and articles being produced every day, it becomes important that one uses the text information and understands them so that they would be able to make the best use of it. 

When there are __millions of documents__ and __publications__ introduced, it is often not possible for an average reader to classify them based on their difficulty. This is because one would have to go over the materials fully and understand them before assigning the difficulty of the text. Therefore, we should think of ways in which we could automate this system which would classify the documents into categories.

![](https://github.com/suhasmaddali/Images/blob/main/patrick-tomasso-Oaqk7qqNh_c-unsplash.jpg)

Since there are a lot of __publications__ and __articles__ being published every day, it sometimes becomes tedious and difficult for the __librarians__ to go over the materials and classify them based on their level of comprehension. As a result, a high-difficulty text might be given to a child who is just about 10 years of age. On the contrary, a low-difficulty text might be given to a highly educated individual who might easily understand the text but lacks much knowledge.

Therefore, it would be of great help to librarians and readers if there are algorithms that could classify the text based on the difficulty without these people having to go through the documents. As a result, this reduces the manpower needed to read the books and also saves a lot of time and effort on the part of humans. 

## Machine Learning and Deep Learning 

* With __machine learning__ and __deep learning__, it is possible to predict the readability of the text and understand some of the important features that determine the difficulty respectively. 
* Therefore, we have to consider a few important parameters when determining the difficulty of different machine learning models respectively.
* We have to take into consideration the difficulty of the text along with other important features such as the number of syllables and the difficulty of the words in order to determine the overall level of the text. 

## Natural Language Processing (NLP)

* We have to use the __natural language processing (NLP)__ when we are dealing with the text respectively.
* Since we have a text, we have to use various processing techniques so that they are considered into forms that could be easy for machine learning purposes.
* Once those values are converted into vectors, we are going to use them by giving them to different machine learning and deep learning models with a different set of layers respectively.
* We would be working with different __machine learning__ and __deep learning algorithms__ and understand some of the important metrics that are needed for the problem at hand. 
* We see that since the target that we are going to be predicting is continuous, we are going to be using the regression machine learning techniques so that we get continuous output.

## Vectorizers

There are various vectorizers that were used to convert a given text into a form of a numeric vector representation so that it could be given to machine learning models for predictions for difficulty. Below are some of the vectorizers used to convert a given text to vectors.

* [__Count Vectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [__Tfidf Vectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
* [__Average Word2Vec (Glove Vectors)__](http://ethen8181.github.io/machine-learning/keras/text_classification/word2vec_text_classification.html)
* [__Tfidf Word2Vec__](https://datascience.stackexchange.com/questions/28598/word2vec-embeddings-with-tf-idf)

## Machine Learning Models

The output variable that we are considering is a continuous variable, therefore, we should be using regression techniques for predictions. Below are some of the machine learning and deep learning models used to predict the difficulty of texts.

* [__Deep Neural Networks__](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
* [__Linear Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [__K - Neighbors Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
* [__PLS Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)
* [__Decision Tree Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
* [__Gradient Boosting Regressor__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

## Outcomes
* __TFIDF Word2Vec__ Vectorizer was the best encoding technique which results in a significant reduction in the __mean absolute error__ respectively. 
* __Gradient Boosted Decision Trees (GBDT)__ were performing the best in terms of the __mean absolute__ and __mean squared error__ of predicting the difficulty of texts.

## Future Scope 
* The best model (__Gradient Boosted Decision Trees__) could be integrated in real-time in Office tools such as __Microsoft Word__ and __Microsoft Presentation__ so that a user can get an indication of the difficulty of his/her sentences.
* Additional text information could be added from other sources such as __Wikipedia__ to further reduce the __mean absolute error__ of the models.  

## 👉 Directions to download the repository and run the notebook 

This is for the Washington Bike Demand Prediction repository. But the same steps could be followed for this repository. 

1. You'll have to download and install Git which could be used for cloning the repositories that are present. The link to download Git is https://git-scm.com/downloads.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(14).png" width = "600"/>
 
2. Once "Git" is downloaded and installed, you'll have to right-click on the location where you would like to download this repository. I would like to store it in the "Git Folder" location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(15).png" width = "600" />

3. If you have successfully installed Git, you'll get an option called "Gitbash Here" when you right-click on a particular location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(16).png" width = "600" />


4. Once the Gitbash terminal opens, you'll need to write "Git clone" and then paste the link to the repository.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(18).png" width = "600" />

5. The link to the repository can be found when you click on "Code" (Green button) and then, there would be an HTML link just below. Therefore, the command to download a particular repository should be "Git clone HTML" where the HTML is replaced by the link to this repository. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(17).png" width = "600" />

6. After successfully downloading the repository, there should be a folder with the name of the repository as can be seen below.

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(19).png" width = "600" />

7. Once the repository is downloaded, go to the start button and search for "Anaconda Prompt" if you have anaconda installed. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(20).png" width = "600" />

8. Later, open the Jupyter notebook by writing "Jupyter notebook" in the Anaconda prompt. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(21).png" width = "600" />

9. Now the following would open with a list of directories. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(22).png" width = "600" />

10. Search for the location where you have downloaded the repository. Be sure to open that folder. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(12).png" width = "600" />

11. You might now run the .ipynb files present in the repository to open the notebook and the python code present in it. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(13).png" width = "600" />

That's it, you should be able to read the code now. Thanks. 

