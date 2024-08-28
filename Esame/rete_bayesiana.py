import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.utils import resample

def build_bayesian_network(percorso_file_dataset):
    # Carica il dataset
    df = pd.read_csv(percorso_file_dataset)

    # Seleziona le caratteristiche desiderate
    features = ['balcony', 'condition', 'interiorQual', 'hasKitchen', 
                'lift', 'livingSpace', 'noRooms', 'floor', 'Tipologia_Affitti', 'newlyConst','typeOfFlat']
    df = df[features]
    print(df.head(10))
    # Discretizza le variabili categoriche
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.Categorical(df[col]).codes

    # Discretizza le variabili booleane
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    #discretizza il livingspace in 5 intervalli di grandezza uguale
    df['livingSpace'] = pd.qcut(df['livingSpace'], q=5, labels=False)
    df =oversampling(df)
    #fai lo shuffle delle righe dl dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    #sostituire tutti 
    #elimina valori nulli
    df = df.dropna()

    # Ricerca della struttura ottimale
    hc_k2 = HillClimbSearch(df)
    k2_model = hc_k2.estimate(scoring_method='k2score')

    # Creazione della rete bayesiana
    model = BayesianNetwork(k2_model.edges())
    model.fit(df, estimator=MaximumLikelihoodEstimator, n_jobs=-1)

    # Stampa i nodi e gli archi del modello
    print("Nodi (variabili) nel modello:", model.nodes())
    print("Archi (relazioni) nel modello:", model.edges())

    # Restituisce il modello e il test set
    return model

def crea_previsione(model : BayesianNetwork):
    examples = crea_esempio(model)
    print("Esempio creato:")
    print(examples)
    inference = VariableElimination(model)
    evidence = examples.iloc[0].to_dict()
    print("Evidenza utilizzata per la query:", evidence)
    result=inference.query(variables=['Tipologia_Affitti'], evidence=evidence)
    print(result)


def crea_esempio(model):
    return model.simulate(n_samples=1).drop(columns=['Tipologia_Affitti'])

def plot_bayesian_network(model):
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    
    pos = nx.spring_layout(G)  # Cambia il layout se necessario
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=2000, node_color='red', font_size=10, font_weight='bold', edge_color='grey')
    plt.title("Bayesian Network Layout")
    plt.show()

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