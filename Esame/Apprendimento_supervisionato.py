import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # Modificato: importato DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  learning_curve, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.utils import resample

# Function to show learning curve for each model
def plot_learning_curves(
        regressionModel, X, y, regressionModelName):
    # Calculate the learning curve for the given regression model
    train_sizes, train_scores, test_scores = learning_curve(
        regressionModel, 
        X, y, 
        cv=10, 
        scoring='accuracy', 
        random_state=42,
        n_jobs=-1
    )
    # Converts the negative mean squared error to a positive mean error
    mean_train_errors =1- np.mean(train_scores, axis=1)
    mean_test_errors =1- np.mean(test_scores, axis=1)
    #calcola varianza e deviazione standard
    std_train_errors = np.std(train_scores, axis=1)
    std_test_errors = np.std(test_scores, axis=1)
    std_train_err=np.var(train_scores, axis=1)
    std_test_err=np.var(test_scores, axis=1)
    # Stampa i risultati
    print("deviazione standard Training")
    print(std_train_err.mean())
    print("deviazione standard Test")
    print(std_test_err.mean())
    print("varianza Training")
    print(std_train_errors.mean())
    print("varianza Test")
    print(std_test_errors.mean())
    plt.figure()
    plt.plot(train_sizes, mean_train_errors, 'o-', color='r', label='Training Error')
    plt.plot(train_sizes, mean_test_errors, 'o-', color='g', label='Test Error')
    plt.xlabel('Training examples')
    plt.ylabel('Mean Error')
    plt.legend(loc='best')
    plt.title(regressionModelName + ' Learning Curves')
    plt.show()


# Function that returns the best hyperparameters for each model
def returnBestHyperparameters(dataset, targetColumn):

    X = dataset.drop(targetColumn, axis=1)
    y = dataset[targetColumn]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modificato: utilizzo di DecisionTreeRegressor
    reg = DecisionTreeClassifier()
    reg1 = RandomForestClassifier()

    DecisionTreeHyperparameters = {
        'DecisionTree__max_depth': [10,20,30],
        'DecisionTree__min_samples_split': [2,5,10,20,50],
        'DecisionTree__min_samples_leaf': [1,2,5,10,20],
        'DecisionTree__criterion': [ 'gini', 'entropy','log_loss'],
    }
    #aggiungi il randomForestRegressor
    RandomForestRegressorHyperparameters = {
        'RandomForest__n_estimators': [30,50,75],
        'RandomForest__max_depth': [30,50],
        'RandomForest__min_samples_split': [2,5,10,20],
        'RandomForest__min_samples_leaf': [1,2,5,10,20],
        'RandomForest__criterion': ['gini', 'entropy','log_loss'],
        'RandomForest__n_jobs': [-1]
    }
    
    gridSearchCV_reg = GridSearchCV(Pipeline([('DecisionTree', reg)]), DecisionTreeHyperparameters, cv=5, scoring='accuracy',verbose=2, n_jobs=-1)
    gridSearchCV_reg1 = GridSearchCV(Pipeline([('RandomForest', reg1)]), RandomForestRegressorHyperparameters, cv=5, scoring='accuracy',verbose=2, n_jobs=-1)
    

    gridSearchCV_reg.fit(X_train, y_train)
    gridSearchCV_reg1.fit(X_train, y_train)
    
    bestParameters = {
        'DecisionTree__max_depth': gridSearchCV_reg.best_params_['DecisionTree__max_depth'],
        'DecisionTree__min_samples_split': gridSearchCV_reg.best_params_['DecisionTree__min_samples_split'],
        'DecisionTree__min_samples_leaf': gridSearchCV_reg.best_params_['DecisionTree__min_samples_leaf'],
        'DecisionTree__criterion': gridSearchCV_reg.best_params_['DecisionTree__criterion'],
        'RandomForest__max_depth': gridSearchCV_reg1.best_params_['RandomForest__max_depth'],
        'RandomForest__min_samples_split': gridSearchCV_reg1.best_params_['RandomForest__min_samples_split'],
        'RandomForest__min_samples_leaf': gridSearchCV_reg1.best_params_['RandomForest__min_samples_leaf'],
        'RandomForest__criterion': gridSearchCV_reg1.best_params_['RandomForest__criterion'],
        'RandomForest__n_estimators': gridSearchCV_reg1.best_params_['RandomForest__n_estimators']
    }
    print("\033[94m" + str(bestParameters) + "\033[0m")
    return bestParameters


# Function to train the model using cross-validation
def trainModelKFold(dataSet, targetColumn):
    model = {
        'DecisionTreeClassifier': {
            'accuracy_list': [],
            'precision_list': [],
            'recall_list': [],
            'f1': []
        },
        'RandomForestClassifier': {
            'accuracy_list': [],
            'precision_list': [],
            'recall_list': [],
            'f1': []
        }
    }
    
    bestParameters = returnBestHyperparameters(dataSet, targetColumn)
    
    if targetColumn not in dataSet.columns:
        raise KeyError(f"The column '{targetColumn}' is not present in the dataset.")
    
    X = dataSet.drop(targetColumn, axis=1)
    y = dataSet[targetColumn]
    
    DecisionTreeModel = DecisionTreeClassifier(
        max_depth=bestParameters['DecisionTree__max_depth'],
        min_samples_split=bestParameters['DecisionTree__min_samples_split'],
        min_samples_leaf=bestParameters['DecisionTree__min_samples_leaf'],
        criterion=bestParameters['DecisionTree__criterion']
    )
    RandomForestModel = RandomForestClassifier(
        n_estimators=bestParameters['RandomForest__n_estimators'],
        max_depth=bestParameters['RandomForest__max_depth'],
        min_samples_split=bestParameters['RandomForest__min_samples_split'],
        min_samples_leaf=bestParameters['RandomForest__min_samples_leaf'],
        criterion=bestParameters['RandomForest__criterion']
    )
    
    # Use KFold for cross-validation
    kFold = KFold(n_splits=10,random_state=42, shuffle=True)
    
    model['DecisionTreeClassifier']['accuracy_list'] = cross_val_score(DecisionTreeModel, X, y, cv=kFold, scoring='accuracy', n_jobs=-1)
    model['DecisionTreeClassifier']['precision_list'] = cross_val_score(DecisionTreeModel, X, y, cv=kFold, scoring='precision_macro', n_jobs=-1)
    model['DecisionTreeClassifier']['recall_list'] = cross_val_score(DecisionTreeModel, X, y, cv=kFold, scoring='recall_macro', n_jobs=-1)
    model['DecisionTreeClassifier']['f1'] = cross_val_score(DecisionTreeModel, X, y, cv=kFold, scoring='f1_macro', n_jobs=-1)
    model['RandomForestClassifier']['accuracy_list'] = cross_val_score(RandomForestModel, X, y, cv=kFold, scoring='accuracy', n_jobs=-1)
    model['RandomForestClassifier']['precision_list'] = cross_val_score(RandomForestModel, X, y, cv=kFold, scoring='precision_macro', n_jobs=-1)
    model['RandomForestClassifier']['recall_list'] = cross_val_score(RandomForestModel, X, y, cv=kFold, scoring='recall_macro', n_jobs=-1)
    model['RandomForestClassifier']['f1'] = cross_val_score(RandomForestModel, X, y, cv=kFold, scoring='f1_macro', n_jobs=-1)

    # Fit the model
    RandomForestModel.fit(X, y)
    importance = RandomForestModel.feature_importances_
    # Stampa l'importanza delle feature
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()
    #Stampa i nomi delle feature
    print(dataSet.columns)
    # Modificato: cambiato il nome del modello nella funzione di visualizzazione
    plot_learning_curves(DecisionTreeModel, X, y, "Decision Tree Classifier")
    plot_learning_curves(RandomForestModel, X, y, "Random Forest Classifier")

    #Stampa dei risultati e delle diverse metriche
    print("Decision Tree Classifier")
    print("Accuracy: ", model['DecisionTreeClassifier']['accuracy_list'])
    print("Precision: ", model['DecisionTreeClassifier']['precision_list'])
    print("Recall: ", model['DecisionTreeClassifier']['recall_list'])
    print("F1: ", model['DecisionTreeClassifier']['f1'])
    print("Random Forest Classifier")
    print("Accuracy: ", model['RandomForestClassifier']['accuracy_list'])
    print("Precision: ", model['RandomForestClassifier']['precision_list'])
    print("Recall: ", model['RandomForestClassifier']['recall_list'])
    print("F1: ", model['RandomForestClassifier']['f1'])
    #stampa le medie delle metriche di valutazione
    print("Decision Tree Classifier")
    print("Accuracy: ", model['DecisionTreeClassifier']['accuracy_list'].mean())
    print("Precision: ", model['DecisionTreeClassifier']['precision_list'].mean())
    print("Recall: ", model['DecisionTreeClassifier']['recall_list'].mean())
    print("F1: ", model['DecisionTreeClassifier']['f1'].mean())
    print("Random Forest Classifier")
    print("Accuracy: ", model['RandomForestClassifier']['accuracy_list'].mean())
    print("Precision: ", model['RandomForestClassifier']['precision_list'].mean())
    print("Recall: ", model['RandomForestClassifier']['recall_list'].mean())
    print("F1: ", model['RandomForestClassifier']['f1'].mean())

    return model


# Data preprocessing function
def preprocessData(dataset):
    dataset = dataset.drop('Unnamed: 0', axis=1)
    dataset = dataset.drop('totalRent', axis=1)
    dataset = dataset.drop('ID', axis=1)
    dataset = dataset.drop('regio1', axis=1)
    dataset = dataset.drop('Tipologia_Propieta',axis=1)
    dataset = dataset.drop('street', axis=1)
    dataset = dataset.drop('houseNumber', axis=1)
    dataset = dataset.drop('streetPlain', axis=1)
    dataset = dataset.drop('geo_krs', axis=1)
    dataset = dataset.drop('petsAllowed', axis=1)
    dataset = dataset.dropna()
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset = dataset.dropna()
    dataset['regio2'] = dataset['regio2'].astype('category').cat.codes
    dataset['typeOfFlat'] = dataset['typeOfFlat'].astype('category').cat.codes
    dataset['condition'] = dataset['condition'].astype('category').cat.codes
    dataset['interiorQual'] = dataset['interiorQual'].astype('category').cat.codes
    dataset['Tipologia_Affitti'] = dataset['Tipologia_Affitti'].astype('category').cat.codes
    dataset['livingSpace'] = pd.qcut(dataset['livingSpace'], q=5, labels=False)
    dataset['yearConstructed'] = pd.qcut(dataset['yearConstructed'], q=5, labels=False)
    dataset=pulizia_dataset(dataset)
    dataset=oversampling(dataset)
    print(dataset['Tipologia_Affitti'].value_counts())
    dataset=dataset.dropna()
    return dataset

def oversampling(dataset):
    df_majority = dataset[dataset['Tipologia_Affitti'] == 1]
    df_minor1 = dataset[dataset['Tipologia_Affitti'] == 0]
    df_minor2 = dataset[dataset['Tipologia_Affitti'] == 2]
    # Sottocampiona la classe maggioritaria
    df_minority_upsampled1 = resample(df_minor1,
                                      replace=True,  # Sostituzione per permettere il campionamento ripetuto
                                      n_samples=len(df_majority),  # Eguaglia la classe maggioritaria
                                      random_state=42)  # Per la riproducibilità
    df_minority_upsampled2 = resample(df_minor2,
                                      replace=True,  # Sostituzione per permettere il campionamento ripetuto
                                      n_samples=len(df_majority),  # Eguaglia la classe maggioritaria
                                      random_state=42)  # Per la riproducibilità

    # Unisci i dati rimanenti
    dataset = pd.concat([df_majority, df_minority_upsampled1, df_minority_upsampled2])
    #fau un shuffle delle righe del dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    return dataset


def pulizia_dataset(df):
    df=df.drop('Unnamed: 0.1', axis=1)
    df=df.drop('newlyConst', axis=1)
    df=df.drop('yearConstructed', axis=1)
    df=df.drop('hasKitchen', axis=1)
    df=df.drop('noParkSpaces', axis=1)
    df=df.drop('numberOfFloors', axis=1)
    return df