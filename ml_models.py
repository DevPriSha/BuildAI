

#import stuff
from sklearn.linear_model import LogisticRegression, SGDClassifier,PassiveAggressiveClassifier, RidgeClassifier, Perceptron, Lasso, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, TheilSenRegressor, HuberRegressor, RANSACRegressor, LinearRegression, SGDRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor, Ridge
from sklearn.svm import SVC, LinearSVC, NuSVC, OneClassSVM, LinearSVR, NuSVR, SVR
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
#import Linear SVR


#ML Models for Supervised Binary Classification

#1. Logistic Regression
def Logistic1(X,y,X_Test,y_Test):
    model = LogisticRegression()
    model.fit(X,y)
    return model

#2. Support Vector Machine
def SVM1(X,y,X_Test,y_Test):
    model = SVC()
    model.fit(X,y)
    return model

#3. Random Forest
def RandomForest1(X,y,X_Test,y_Test):
    model = RandomForestClassifier()
    model.fit(X,y)
    return model

#4. XGBoost
def XGBoost1(X,y,X_Test,y_Test):
    model = XGBClassifier()
    model.fit(X,y)
    return model

#5. CatBoost
def CatBoost1(X,y,X_Test,y_Test):
    model = CatBoostClassifier()
    model.fit(X,y)
    return model

#6. Passive Aggressive Classifier
def PassiveAggressive1(X,y,X_Test,y_Test):
    model = PassiveAggressiveClassifier()
    model.fit(X,y)
    return model

#7. AdaBoost
def AdaBoost1(X,y,X_Test,y_Test):
    model = AdaBoostClassifier()
    model.fit(X,y)
    return model

#8. Gradient Boosting
def GradientBoosting1(X,y,X_Test,y_Test):
    model = GradientBoostingClassifier()
    model.fit(X,y)
    return model

#9. Extra Trees
def ExtraTrees1(X,y,X_Test,y_Test):
    model = ExtraTreesClassifier()
    model.fit(X,y)
    return model

#10. Decision Tree
def DecisionTree1(X,y,X_Test,y_Test, max_depth=5):
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X,y)
    return model

#11. KNN
def KNN1(X,y,X_Test,y_Test, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X,y)
    return model

#12. Naive Bayes
def NaiveBayes1(X,y,X_Test,y_Test):
    model = GaussianNB()
    model.fit(X,y)
    return model

#13. Linear Discriminant Analysis
def LDA1(X,y,X_Test,y_Test):
    model = LinearDiscriminantAnalysis()
    model.fit(X,y)
    return model

#14. Quadratic Discriminant Analysis
def QDA1(X,y,X_Test,y_Test):
    model = QuadraticDiscriminantAnalysis()
    model.fit(X,y)
    return model

#15. Stochastic Gradient Descent
def SGD1(X,y,X_Test,y_Test):
    model = SGDClassifier()
    model.fit(X,y)
    return model

#17. Ridge Classifier
def Ridge1(X,y,X_Test,y_Test):
    model = RidgeClassifier()
    model.fit(X,y)
    return model

#18. Perceptron
def Perceptron1(X,y,X_Test,y_Test):
    model = Perceptron()
    model.fit(X,y)
    return model

#19. Linear Support Vector Machine
def LinearSVM1(X,y,X_Test,y_Test):
    model = LinearSVC()
    model.fit(X,y)
    return model

#20. Nearest Centroid
def NearestCentroid1(X,y,X_Test,y_Test):
    model = NearestCentroid()
    model.fit(X,y)
    return model

#21. Gaussian Process Classifier
def GaussianProcess1(X,y,X_Test,y_Test):
    model = GaussianProcessClassifier()
    model.fit(X,y)
    return model

#22. Bagging Classifier
def Bagging1(X,y,X_Test,y_Test):
    model = BaggingClassifier()
    model.fit(X,y)
    return model

#23. MLP Classifier
def MLP1(X,y,X_Test,y_Test):
    model = MLPClassifier()
    model.fit(X,y)
    return model

#24. RBF SVM
def RBFSVM1(X,y,X_Test,y_Test):
    model = SVC(kernel='rbf')
    model.fit(X,y)
    return model

#25. Polynomial SVM
def PolySVM1(X,y,X_Test,y_Test):
    model = SVC(kernel='poly')
    model.fit(X,y)
    return model

#26. Sigmoid SVM
def SigmoidSVM1(X,y,X_Test,y_Test):
    model = SVC(kernel='sigmoid')
    model.fit(X,y)
    return model

#27. NuSVC
def NuSVC1(X,y,X_Test,y_Test):
    model = NuSVC()
    model.fit(X,y)
    return model

#28. One Class SVM
def OneClassSVM1(X,y,X_Test,y_Test):
    model = OneClassSVM()
    model.fit(X,y)
    return model

# Supervised MultiClass Classification Models

#1. Multinomial Naive Bayes
def MultinomialNB1(X,y,X_Test,y_Test):
    model = MultinomialNB()
    model.fit(X,y)
    return model

#2. Bernoulli Naive Bayes
def BernoulliNB1(X,y,X_Test,y_Test):
    model = BernoulliNB()
    model.fit(X,y)
    return model

#3. Complement Naive Bayes
def ComplementNB1(X,y,X_Test,y_Test):
    model = ComplementNB()
    model.fit(X,y)
    return model


# Supervised Regression Models
#1. Linear Regression
def LinearRegression1(X,y,X_Test,y_Test):
    model = LinearRegression()
    model.fit(X,y)
    return model

#2. Ridge Regression
def RidgeRegression1(X,y,X_Test,y_Test):
    model = Ridge()
    model.fit(X,y)
    return model

#3. Lasso Regression
def LassoRegression1(X,y,X_Test,y_Test):
    model = Lasso()
    model.fit(X,y)
    return model

#4. Elastic Net Regression
def ElasticNetRegression1(X,y,X_Test,y_Test):
    model = ElasticNet()
    model.fit(X,y)
    return model

#5. Lars Regression
def LarsRegression1(X,y,X_Test,y_Test):
    model = Lars()
    model.fit(X,y)
    return model

#6. Lasso Lars Regression
def LassoLarsRegression1(X,y,X_Test,y_Test):
    model = LassoLars()
    model.fit(X,y)
    return model

#7. Orthogonal Matching Pursuit Regression
def OrthogonalMatchingPursuitRegression1(X,y,X_Test,y_Test):
    model = OrthogonalMatchingPursuit()
    model.fit(X,y)
    return model

#8. Bayesian Ridge Regression
def BayesianRidgeRegression1(X,y,X_Test,y_Test):
    model = BayesianRidge()
    model.fit(X,y)
    return model

#9. ARD Regression
def ARDRegression1(X,y,X_Test,y_Test):
    model = ARDRegression()
    model.fit(X,y)
    return model

#10. Stochastic Gradient Descent Regression
def SGDRegression1(X,y,X_Test,y_Test):
    model = SGDRegressor()
    model.fit(X,y)
    return model

#11. Passive Aggressive Regression
def PassiveAggressiveRegression1(X,y,X_Test,y_Test):
    model = PassiveAggressiveRegressor()
    model.fit(X,y)
    return model

#12. TheilSen Regression
def TheilSenRegression1(X,y,X_Test,y_Test):
    model = TheilSenRegressor()
    model.fit(X,y)
    return model

#13. Huber Regression
def HuberRegression1(X,y,X_Test,y_Test):
    model = HuberRegressor()
    model.fit(X,y)
    return model

#14. RANSAC Regression
def RANSACRegression1(X,y,X_Test,y_Test):
    model = RANSACRegressor()
    model.fit(X,y)
    return model

#15. TheilSen Regression
def TheilSenRegression1(X,y,X_Test,y_Test):
    model = TheilSenRegressor()
    model.fit(X,y)
    return model

#16. Linear Support Vector Regression
def LinearSVR1(X,y,X_Test,y_Test):
    model = LinearSVR()
    model.fit(X,y)
    return model

#17. NuSVR
def NuSVR1(X,y,X_Test,y_Test):
    model = NuSVR()
    model.fit(X,y)
    return model

#18. K Neighbors Regression
def KNeighborsRegression1(X,y,X_Test,y_Test):
    model = KNeighborsRegressor()
    model.fit(X,y)
    return model

#19. Radius Neighbors Regression
def RadiusNeighborsRegression1(X,y,X_Test,y_Test):
    model = RadiusNeighborsRegressor()
    model.fit(X,y)
    return model

#20. Decision Tree Regression
def DecisionTreeRegression1(X,y,X_Test,y_Test):
    model = DecisionTreeRegressor()
    model.fit(X,y)
    return model

#21. Extra Tree Regression
def ExtraTreesRegression1(X,y,X_Test,y_Test):
    model = ExtraTreesRegressor()
    model.fit(X,y)
    return model

#22. Random Forest Regression
def RandomForestRegression1(X,y,X_Test,y_Test):
    model = RandomForestRegressor()
    model.fit(X,y)
    return model


#create a list of all the functions
classifiers = {"LogisticRegression":Logistic1, "SVM": SVM1, "Random Forest": RandomForest1, "XGBoost": XGBoost1, "CatBoost": CatBoost1, "Passive Aggressive": PassiveAggressive1, "AdaBoost": AdaBoost1, "Gradient Boosting": GradientBoosting1, "K Neighbors": KNN1, "Decision Tree": DecisionTree1, "Extra Trees": ExtraTrees1, "Bagging": Bagging1, "Linear SVC": LinearSVM1, "NuSVC": NuSVC1, "One Class SVM": OneClassSVM1, "Multinomial Naive Bayes": MultinomialNB1, "Bernoulli Naive Bayes": BernoulliNB1, "Complement Naive Bayes": ComplementNB1}

regressors = {"LinearRegression":LinearRegression1, "Ridge":RidgeRegression1, "Lasso":LassoRegression1, "ElasticNet":ElasticNetRegression1, "Lars":LarsRegression1, "LassoLars":LassoLarsRegression1, "OrthogonalMatchingPursuit":OrthogonalMatchingPursuitRegression1, "BayesianRidge":BayesianRidgeRegression1, "ARDRegression":ARDRegression1, "PassiveAggressiveRegressor":PassiveAggressiveRegression1, "TheilSenRegressor":TheilSenRegression1, "HuberRegressor":HuberRegression1, "RANSACRegressor":RANSACRegression1, "LinearSVR":LinearSVR1, "NuSVR":NuSVR1, "KNeighborsRegressor":KNeighborsRegression1, "RadiusNeighborsRegressor":RadiusNeighborsRegression1, "DecisionTreeRegressor":DecisionTreeRegression1, "ExtraTreesRegressor":ExtraTreesRegression1, "RandomForestRegressor":RandomForestRegression1}

#function to get model name as string and run the function from the dictionary
def get_model_classifier(model_name, X, y, X_Test, y_Test):
    model = classifiers[model_name](X,y,X_Test,y_Test)
    return model

def get_model_regressor(model_name, X, y, X_Test, y_Test):
    model = regressors[model_name](X,y,X_Test,y_Test)
    return model

if __name__ == "__main__":
    print(len(classifiers), "Classifiers Loaded")
    print(len(regressors), "Regressors Loaded")
    print("Classifiers: ", classifiers.keys())
    print("Regressors: ", regressors.keys())