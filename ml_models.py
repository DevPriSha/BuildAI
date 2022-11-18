

#import stuff
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier, Perceptron, Lasso, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, TheilSenRegressor, HuberRegressor, RANSACRegressor, LinearRegression, SGDRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsRegressor, KNeighborsRegressor 
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier




#ML Models for Supervised Binary Classification

#1. Logistic Regression
def Logistic(X,y,X_Test,y_Test):
    model = LogisticRegression()
    model.fit(X,y)
    return model

#2. Support Vector Machine
def SVM(X,y,X_Test,y_Test):
    model = SVC()
    model.fit(X,y)
    return model

#3. Random Forest
def RandomForest(X,y,X_Test,y_Test):
    model = RandomForestClassifier()
    model.fit(X,y)
    return model

#4. XGBoost
def XGBoost(X,y,X_Test,y_Test):
    model = XGBClassifier()
    model.fit(X,y)
    return model

#5. CatBoost
def CatBoost(X,y,X_Test,y_Test):
    model = CatBoostClassifier()
    model.fit(X,y)
    return model

#6. Passive Aggressive Classifier
def PassiveAggressive(X,y,X_Test,y_Test):
    model = PassiveAggressiveClassifier()
    model.fit(X,y)
    return model

#7. AdaBoost
def AdaBoost(X,y,X_Test,y_Test):
    model = AdaBoostClassifier()
    model.fit(X,y)
    return model

#8. Gradient Boosting
def GradientBoosting(X,y,X_Test,y_Test):
    model = GradientBoostingClassifier()
    model.fit(X,y)
    return model

#9. Extra Trees
def ExtraTrees(X,y,X_Test,y_Test):
    model = ExtraTreesClassifier()
    model.fit(X,y)
    return model

#10. Decision Tree
def DecisionTree(X,y,X_Test,y_Test, max_depth=5):
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X,y)
    return model

#11. KNN
def KNN(X,y,X_Test,y_Test, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X,y)
    return model

#12. Naive Bayes
def NaiveBayes(X,y,X_Test,y_Test):
    model = GaussianNB()
    model.fit(X,y)
    return model

#13. Linear Discriminant Analysis
def LDA(X,y,X_Test,y_Test):
    model = LinearDiscriminantAnalysis()
    model.fit(X,y)
    return model

#14. Quadratic Discriminant Analysis
def QDA(X,y,X_Test,y_Test):
    model = QuadraticDiscriminantAnalysis()
    model.fit(X,y)
    return model

#15. Stochastic Gradient Descent
def SGD(X,y,X_Test,y_Test):
    model = SGDClassifier()
    model.fit(X,y)
    return model

#16. Passive Aggressive Classifier
def PassiveAggressive(X,y,X_Test,y_Test):
    model = PassiveAggressiveClassifier()
    model.fit(X,y)
    return model

#17. Ridge Classifier
def Ridge(X,y,X_Test,y_Test):
    model = RidgeClassifier()
    model.fit(X,y)
    return model

#18. Perceptron
def Perceptron(X,y,X_Test,y_Test):
    model = Perceptron()
    model.fit(X,y)
    return model

#19. Linear Support Vector Machine
def LinearSVM(X,y,X_Test,y_Test):
    model = LinearSVC()
    model.fit(X,y)
    return model

#20. Nearest Centroid
def NearestCentroid(X,y,X_Test,y_Test):
    model = NearestCentroid()
    model.fit(X,y)
    return model

#21. Gaussian Process Classifier
def GaussianProcess(X,y,X_Test,y_Test):
    model = GaussianProcessClassifier()
    model.fit(X,y)
    return model

#22. Bagging Classifier
def Bagging(X,y,X_Test,y_Test):
    model = BaggingClassifier()
    model.fit(X,y)
    return model

#23. MLP Classifier
def MLP(X,y,X_Test,y_Test):
    model = MLPClassifier()
    model.fit(X,y)
    return model

#24. RBF SVM
def RBFSVM(X,y,X_Test,y_Test):
    model = SVC(kernel='rbf')
    model.fit(X,y)
    return model

#25. Polynomial SVM
def PolySVM(X,y,X_Test,y_Test):
    model = SVC(kernel='poly')
    model.fit(X,y)
    return model

#26. Sigmoid SVM
def SigmoidSVM(X,y,X_Test,y_Test):
    model = SVC(kernel='sigmoid')
    model.fit(X,y)
    return model

#27. NuSVC
def NuSVC(X,y,X_Test,y_Test):
    model = NuSVC()
    model.fit(X,y)
    return model

#28. One Class SVM
def OneClassSVM(X,y,X_Test,y_Test):
    model = OneClassSVM()
    model.fit(X,y)
    return model

# Supervised MultiClass Classification Models

#1. Multinomial Naive Bayes
def MultinomialNB(X,y,X_Test,y_Test):
    model = MultinomialNB()
    model.fit(X,y)
    return model

#2. Bernoulli Naive Bayes
def BernoulliNB(X,y,X_Test,y_Test):
    model = BernoulliNB()
    model.fit(X,y)
    return model

#3. Complement Naive Bayes
def ComplementNB(X,y,X_Test,y_Test):
    model = ComplementNB()
    model.fit(X,y)
    return model


# Supervised Regression Models
#1. Linear Regression
def LinearRegression(X,y,X_Test,y_Test):
    model = LinearRegression()
    model.fit(X,y)
    return model

#2. Ridge Regression
def RidgeRegression(X,y,X_Test,y_Test):
    model = Ridge()
    model.fit(X,y)
    return model

#3. Lasso Regression
def LassoRegression(X,y,X_Test,y_Test):
    model = Lasso()
    model.fit(X,y)
    return model

#4. Elastic Net Regression
def ElasticNetRegression(X,y,X_Test,y_Test):
    model = ElasticNet()
    model.fit(X,y)
    return model

#5. Lars Regression
def LarsRegression(X,y,X_Test,y_Test):
    model = Lars()
    model.fit(X,y)
    return model

#6. Lasso Lars Regression
def LassoLarsRegression(X,y,X_Test,y_Test):
    model = LassoLars()
    model.fit(X,y)
    return model

#7. Orthogonal Matching Pursuit Regression
def OrthogonalMatchingPursuitRegression(X,y,X_Test,y_Test):
    model = OrthogonalMatchingPursuit()
    model.fit(X,y)
    return model

#8. Bayesian Ridge Regression
def BayesianRidgeRegression(X,y,X_Test,y_Test):
    model = BayesianRidge()
    model.fit(X,y)
    return model

#9. ARD Regression
def ARDRegression(X,y,X_Test,y_Test):
    model = ARDRegression()
    model.fit(X,y)
    return model

#10. Stochastic Gradient Descent Regression
def SGDRegression(X,y,X_Test,y_Test):
    model = SGDRegressor()
    model.fit(X,y)
    return model

#11. Passive Aggressive Regression
def PassiveAggressiveRegression(X,y,X_Test,y_Test):
    model = PassiveAggressiveRegressor()
    model.fit(X,y)
    return model

#12. TheilSen Regression
def TheilSenRegression(X,y,X_Test,y_Test):
    model = TheilSenRegressor()
    model.fit(X,y)
    return model

#13. Huber Regression
def HuberRegression(X,y,X_Test,y_Test):
    model = HuberRegressor()
    model.fit(X,y)
    return model

#14. RANSAC Regression
def RANSACRegression(X,y,X_Test,y_Test):
    model = RANSACRegressor()
    model.fit(X,y)
    return model

#15. TheilSen Regression
def TheilSenRegression(X,y,X_Test,y_Test):
    model = TheilSenRegressor()
    model.fit(X,y)
    return model

#16. Linear Support Vector Regression
def LinearSVR(X,y,X_Test,y_Test):
    model = LinearSVR()
    model.fit(X,y)
    return model

#17. NuSVR
def NuSVR(X,y,X_Test,y_Test):
    model = NuSVR()
    model.fit(X,y)
    return model

#18. K Neighbors Regression
def KNeighborsRegression(X,y,X_Test,y_Test):
    model = KNeighborsRegressor()
    model.fit(X,y)
    return model

#19. Radius Neighbors Regression
def RadiusNeighborsRegression(X,y,X_Test,y_Test):
    model = RadiusNeighborsRegressor()
    model.fit(X,y)
    return model

#20. Decision Tree Regression
def DecisionTreeRegression(X,y,X_Test,y_Test):
    model = DecisionTreeRegressor()
    model.fit(X,y)
    return model

#21. Extra Tree Regression
def ExtraTreesRegression(X,y,X_Test,y_Test):
    model = ExtraTreesRegressor()
    model.fit(X,y)
    return model

#22. Random Forest Regression
def RandomForestRegression(X,y,X_Test,y_Test):
    model = RandomForestRegressor()
    model.fit(X,y)
    return model


classifiers = {"Logistic Regression":LogisticRegression, "K Neighbors Classifier":KNeighborsClassifier, "Decision Tree Classifier":DecisionTreeClassifier, "Extra Tree Classifier":ExtraTreesClassifier, "Random Forest Classifier":RandomForestClassifier, "AdaBoost Classifier":AdaBoostClassifier, "Gradient Boosting Classifier":GradientBoostingClassifier, "Bagging Classifier":BaggingClassifier, "Linear Discriminant Analysis":LinearDiscriminantAnalysis, "Quadratic Discriminant Analysis":QuadraticDiscriminantAnalysis, "Gaussian Naive Bayes":GaussianNB, "Linear SVM":LinearSVM, "RBF SVM":RBFSVM, "Polynomial SVM":PolySVM, "Sigmoid SVM":SigmoidSVM, "NuSVC":NuSVC, "One Class SVM":OneClassSVM, "Multinomial Naive Bayes":MultinomialNB, "Bernoulli Naive Bayes":BernoulliNB, "Complement Naive Bayes":ComplementNB}

regressors = { "Linear Regression":LinearRegression, "Ridge Regression":RidgeRegression, "Lasso Regression":LassoRegression, "Elastic Net Regression":ElasticNetRegression, "Lars Regression":LarsRegression, "Lasso Lars Regression":LassoLarsRegression, "Orthogonal Matching Pursuit Regression":OrthogonalMatchingPursuitRegression, "Bayesian Ridge Regression":BayesianRidgeRegression, "ARD Regression":ARDRegression, "Stochastic Gradient Descent Regression":SGDRegression, "Passive Aggressive Regression":PassiveAggressiveRegression, "TheilSen Regression":TheilSenRegression, "Huber Regression":HuberRegression, "RANSAC Regression":RANSACRegression, "Linear Support Vector Regression":LinearSVR, "NuSVR":NuSVR, "K Neighbors Regression":KNeighborsRegression, "Radius Neighbors Regression":RadiusNeighborsRegression, "Decision Tree Regression":DecisionTreeRegression, "Extra Tree Regression":ExtraTreesRegression, "Random Forest Regression":RandomForestRegression}

#function to get model name as string and run the function from the dictionary
def get_model(model_name, X, y, X_Test, y_Test):
    model = classifiers[model_name](X,y,X_Test,y_Test)
    return model