import Prolog_ragionamento_logico as prl
import pandas as pd
import apprendimento_non_supervisionato as ans
#import web_semantico as ws
import rete_bayesiana as rb
import Apprendimento_supervisionato as app


percorso_file_dataset = "Dataset//immo_data.csv"
percorso_KB = "Dataset//kb.pl"
df = pd.read_csv(percorso_file_dataset)

#funzione che elimina le colonne superflue e le righe con valori nulli
df_finito = prl.pulizia_dataset(df)
print(df_finito['totalRent'].mean())
#crea la base di conoscenza
prl.crea_base_di_conoscenza(df_finito,percorso_KB)

#utilizzo di Prolog per crerare nuove colonne nel datasets
df_finito=prl.inferenza_dati(percorso_KB, df_finito)
df_finito.to_csv("Dataset//dataset_con_inferenza.csv")

#aggiungo la popolazione delle citta attraverso l'uso del web semantico
#unique_values_dict = {'regio2':list(df_finito['regio2'].unique())}
#dizionario_finale=ws.get_population_wikidata(unique_values_dict)
#df_finito['population'] = df_finito['regio2'].map(dizionario_finale)
#df_finito.to_csv("dataset_aggiornato.csv")

#utilizzo di KMeans per raggruppare i dati
ans.apprendimento_non_supervisionato("Dataset//dataset_con_inferenza.csv")

#costruisco la rete bayesiana per conoscere eventuali correlazioni tra le variabili
model=rb.build_bayesian_network("Dataset//dataset_clusterizzato.csv")
rb.plot_bayesian_network(model)
rb.crea_previsione(model)


#Addestro e valuto i modelli
#dataSet = pd.read_csv("Dataset//dataset_clusterizzato.csv")
#dataSet=app.preprocessData(dataSet)
#differentialColumn = "Tipologia_Affitti"
#print("Sono nel main")
#model= app.trainModelKFold(dataSet, differentialColumn)
