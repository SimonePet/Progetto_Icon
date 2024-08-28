import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator


def regola_del_gomito(data):
    inertia = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='random', n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    
    kl = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
    plt.plot(k_range, inertia, marker='o')
    plt.xlabel('Numero di Cluster')
    plt.ylabel('Inertia')
    plt.title('Regola del Gomito')
    plt.show()
    
    return kl.elbow


def apprendimento_non_supervisionato(nome_datsaset):
    # Supponiamo che i seguenti attributi rappresentino quanto una casa sia pronta per essere abitata
    features = ['yearConstructed', 'balcony', 'condition', 'interiorQual', 'hasKitchen', 'lift', 'garden', 'livingSpace', 'noRooms','typeOfFlat', 'floor', 'numberOfFloors', 'totalRent','regio2']

    # Carica il dataset
    df = pd.read_csv(nome_datsaset)

    # Seleziona solo le colonne di interesse
    df_features = df[features]

    # Se ci sono caratteristiche categoriali, le codifichiamo
    df_features1 = pd.get_dummies(df_features, drop_first=True)

    # Normalizzazione delle caratteristiche
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_features1)

    optimal_clusters = regola_del_gomito(df_scaled)
    print(f"Il numero ottimale di cluster è {optimal_clusters}")

    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', n_init=10,random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

    # Oppure per visualizzare la distribuzione per ogni colonna numerica in base al gruppo
    for column in df_features.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Cluster', y=column, data=df)
            plt.title(f'Distribuzione di {column} per Cluster')
            plt.show()

    # Visualizza la distribuzione dei cluster con un grafico a torta
    df['Cluster'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    plt.title('Distribuzione dei cluster')
    plt.show()
    df.to_csv("Dataset//dataset_clusterizzato.csv")


