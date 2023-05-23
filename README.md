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

## Exploratory Data Analysis (EDA)

In this section, we would focus our attention on the data itself and important visualizations that can have an positive impact on determining whether a text is readable or not. Exploratory data analysis (EDA) is an important step in machine learning where we can determine the important features in our data. In addition, we can determine if there are any outliers in the data or not. 

In the dataframe, we find excerpts that contain the actual output or difficulty in the dataframe. We have the output to be the target which is basically the difficult score for the excerpts.

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/Input%20data%20head.jpg"/>

We find there are a few missing values in the data. There are missing values for features such as url_legal and license features. Since these two features do not have a great impact in model predictions of difficulty scores, we can remove them. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/Data%20missing%20values%20plot.jpg"/>

The histogram shows the distribution of output target values which are nothing but the difficulty scores. Since the difficulty scores are normalized, we have output in the form of floating point numbers. This shows the overall concentration of various difficulty scores across a range of values. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/Target%20output%20histogram%202.jpg"/>

Wordcloud figures give a solid representation of the occurrence of various words in the entire text corpus. The size of the words indicate the occurrence of the words. In our corpus, we find words such as "one", "time" and "said" to the most frequently occurring words. This is expected usually as authors tend to use "said" to denote that someone spoke about something in the novel or a book. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/Word%20distribution%20wordcloud.jpg"/>

We have added a good set of features that improve the predictive performance of machine learning models. It could be seen from the pairplots that the number of sentences impact the difficulty score of the overall text. Higher the number of sentences, there is a higher likelihood that the text is difficult. There are other set of features that could be explored with the help of these pair plots. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/New%20features%20pairplot.jpg"/>

Heatmaps are known for their interpretability and can give us a good picture about the correlation between various features in our data. There is an inverse relationship between word length and target. Similarly, there is similar kind of relationship between lemma length and target as well. As discussed earlier, there is a direct relationship between the number of sentences and the difficulty of texts. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/heatmap%20correlation.jpg"/>

Further taking a look at the scatterplot between target and the number of sentences, it is evident to a certain extent that the number of sentences have a positive relationship with text difficulty. While not have a strong correlation, it can have some impact on our model predictions. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/scatterplot%20sentences%20target.jpg"/>

Taking a look at two features preprocessed_essay_length and num_of_lemmas, there seems to be a strong correlation between these features. There is an evidence of strong positive correlation when we also consider the heatmap presented above. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/scatterplot%20essay_length%20num_of_lemmas.jpg"/>

There seems to be a direct positive relationship between the feature text_shortage and word_length. This highlights that having higher length words leads to shortened form of words when applying strategies such as lemmatization and stemming. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/scatterplot%20text_shortage%20word_length.jpg"/>

### Machine Learning Model Results 

We will create graphs to visualize the relationship between the model's predictions and the true labels. These graphs provide an indication of the model's performance on the test data. If the points on the graph form a nearly straight line, it suggests that the model performed exceptionally well. Conversely, if the points are scattered and not aligned, it indicates that the model struggled to identify clear patterns in the data for accurate predictions.

__Neural Networks:__ These models should be able to find complex patterns from the data and highlight some important features that have predictive power. We tend to see that the model does a recent job of accurately predicting the difficulty of text. However, there are some predictions that are way higher or lower than the expected values. We might also test alternate models to determine the accuracy. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/Neural%20network%20predictions.jpg"/>

__K Neighbors Regression:__ This model works with the help of the total neighbors which are defined earlier in order to perform the regression task of predicting the difficulty of texts. There is more scatter as compared to neural networks we have trained and defined earlier. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/K%20neighbors%20regression%20predictions.jpg"/>

__PLS Regression:__ This stands for Partial Least Squares (PLS) regression which uses a combination of principal component analysis (PCA) and multiple linear regression to generate predictions. The model performance is quite similar to neural networks. However, there is more scatter in this approach compared to neural networks. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/PLS%20regression%20plot.jpg"/>

__Decision Tree Regression:__ Looks like a decision tree regressor is not performing as well as the other models we have tested earlier. While a decision tree regressor is a complex model capable of capturing important distinctions and insights, it was unable to perform optimally. Therefore, we should not always be relying on a few set of models for all the machine learning tasks as it would really depend on the dataset used for making the predictions. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/Decision%20tree%20regression%20plot.jpg"/>

__Gradient Boosted Decision Tree Regression:__ This method does not work as well as the other models we have seen so far. The best performing model was the neural network architecture as it was better able to make predictions about the text difficulty. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/Gradient%20boosted%20decision%20tree%20plot.jpg"/>

We have chosen the neural network archicture we defined earlier as it was able to perform optimally. In addition, we used different encoding techniques for the text which is word2vec approach. Below is a graph that shows the mean squared error and how it reduces as a result of epochs run on the neural network. The model seems to be slightly overfitting but it is doing a decent job. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/Neural%20network%20word%20to%20vec%20loss%20plot.jpg"/>

Here is a similar graph but the main difference is that it uses the TFIDF word2vec approach instead of just the word2vec. It has better performance overall as compared to just using word2vec as highlighted in the graph. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/Neural%20network%20tfidf%20loss%20plot.jpg"/>

These are the final predictions of the best model under the best encoding strategies for the text. Overall, there is a less scatter between the predictions and the actual data, indicating a good performance increase compared to the base models. 

<img src = "https://github.com/suhasmaddali/Predicting-Readability-of-Texts-Using-Machine-Learning/blob/main/images/Neural%20network%20predictions%20last.jpg"/>

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

