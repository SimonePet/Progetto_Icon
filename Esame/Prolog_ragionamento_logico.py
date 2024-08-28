import csv
from pyswip import Prolog  # type: ignore
import pandas as pd  # type: ignore
import numpy as np # type: ignore

# Percorso al file CSV
def pulizia_dataset(df):
    #Eliminazione colonne superflue 
    df = df.drop('date', axis=1)
    df = df.drop('serviceCharge', axis=1)
    df = df.drop('heatingType', axis=1)
    df = df.drop('telekomHybridUploadSpeed', axis=1)
    df = df.drop('telekomUploadSpeed', axis=1)
    df = df.drop('electricityBasePrice', axis=1)
    df = df.drop('picturecount', axis=1)
    df = df.drop('pricetrend', axis=1)
    df = df.drop('scoutId', axis=1)
    df = df.drop('firingTypes', axis=1)
    df = df.drop('yearConstructedRange', axis=1)
    df = df.drop('baseRentRange', axis=1)
    df = df.drop('thermalChar', axis=1)
    df = df.drop('noRoomsRange', axis=1)
    df = df.drop('livingSpaceRange', axis=1)
    df = df.drop('electricityKwhPrice', axis=1)
    df = df.drop('energyEfficiencyClass', axis=1)
    df = df.drop('lastRefurbish', axis=1)
    df = df.drop('heatingCosts', axis=1)
    df = df.drop('telekomTvOffer', axis=1)
    df = df.drop('baseRent', axis=1)
    df = df.drop('geo_bln', axis=1)
    df = df.drop('regio3', axis=1)
    df = df.drop('geo_plz', axis=1)
    df = df.drop('description', axis=1)
    df = df.drop('facilities', axis=1)
    #eliminazione delle righe che contengono valori nulli
    df = df.dropna()
    df=df[df['numberOfFloors']!=730]    
    df=df[df['totalRent']<=10000]
    df=df[df['totalRent']>=400]
    df=df[df['regio1']=='Nordrhein_Westfalen']
    df.loc[:,'ID']=range(1,(df.shape[0])+1)
    print("Dataset pulito")
    return df


# Funzione per scrivere fatti nel file Prolog
def scrivi_fatto_su_file(fact, percorso_file_kb):
    with open(percorso_file_kb, 'r', encoding='utf-8') as file:
        contenuto_esistente = file.read()

    if fact not in contenuto_esistente:
        with open(percorso_file_kb, 'a', encoding='utf-8') as file:
            file.write(f"{fact}.\n")

# Funzione per scrivere informazioni sugli affitti nel file Prolog
def scrivi_info_affitti(data_set, percorso_file_kb):
    with open(percorso_file_kb, "w", encoding="utf-8") as file:  # Sovrascrive il file (lo svuota)
        for index, row in data_set.iterrows():
            ID = row['ID']
            regio1 = row['regio1']
            newly_const = row['newlyConst']
            balcony = row['balcony']
            total_rent = row['totalRent']
            year_constructed = row['yearConstructed']
            no_park_spaces = row['noParkSpaces']
            has_kitchen = row['hasKitchen']
            cellar = row['cellar']
            house_number = row['houseNumber']
            living_space = row['livingSpace']
            condition = row['condition']
            interior_qual = row['interiorQual']
            pets_allowed = row['petsAllowed']
            street = row['street']
            street_plain = row['streetPlain']
            lift = row['lift']
            type_of_flat = row['typeOfFlat']
            no_rooms = row['noRooms']
            floor = row['floor']
            number_of_floors = row['numberOfFloors']
            garden = row['garden']
            regio2 = row['regio2']
            fatto_prolog = f"proprieta_affitto({ID}, '{regio1}', {total_rent}, {living_space}, {no_rooms}, '{type_of_flat}', {newly_const}, {balcony}, {has_kitchen}, {cellar},'{house_number}' , '{condition}', '{interior_qual}', {pets_allowed}, '{street}', '{street_plain}', {lift}, {floor}, {number_of_floors}, {garden}, {year_constructed}, {no_park_spaces}, '{regio2}')"            
            scrivi_fatto_su_file(fatto_prolog, percorso_file_kb)

# Funzione per scrivere regole nel file Prolog
def scrivi_regole(percorso_file_kb):  
    regole = """
        media_affitti_citta(Citta, Media) :-
            findall(PrezzoAffitto, proprieta_affitto(_, _, PrezzoAffitto, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, Citta), ListaAffitti),
            length(ListaAffitti, NumeroAffitti),  
            NumeroAffitti > 0, 
            sum_list(ListaAffitti, SommaAffitti),
            Media is SommaAffitti / NumeroAffitti.

        affitto_economico(ID,Citta, PrezzoAffitto,Media) :-
            proprieta_affitto(ID, _, PrezzoAffitto, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, Citta),
            media_affitti_citta(Citta, Media),
            PrezzoAffitto < Media - 150.

        affitto_costose(ID,Citta, PrezzoAffitto,Media) :-
            proprieta_affitto(ID, _, PrezzoAffitto, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, Citta),
            media_affitti_citta(Citta, Media),
            PrezzoAffitto > Media + 100.

        affitto_medie(ID,Citta, PrezzoAffitto,Media) :-
            proprieta_affitto(ID, _, PrezzoAffitto, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, Citta),
            media_affitti_citta(Citta, Media),
            PrezzoAffitto < Media + 100,
            PrezzoAffitto > Media - 150.

        monolocale(ID) :- 
            proprieta_affitto(ID, _, _, Living_space, No_rooms, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _),
            No_rooms =:= 1,
            Living_space < 50.

        bilocale(ID) :- 
            proprieta_affitto(ID, _, _, Living_space, No_rooms, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _),
            No_rooms =:= 2,
            Living_space >= 50,
            Living_space < 70.

        trilocale(ID) :- 
            proprieta_affitto(ID, _, _, Living_space, No_rooms, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _),
            No_rooms =:= 3,
            Living_space >= 70,
            Living_space < 100.

        proprieta_per_famiglie(ID) :- 
            proprieta_affitto(ID, _, _, Living_space, No_rooms, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _),
            No_rooms > 4,
            Living_space >= 100.

    """


    with open(percorso_file_kb, 'a') as file:
        file.write(regole)
    print("KB costruita")

# Funzione per creare la Knowledge Base
def crea_base_di_conoscenza(local_df, percorso_file_kb):
    scrivi_info_affitti(local_df, percorso_file_kb)
    scrivi_regole(percorso_file_kb)

# Funzione per effettuare ragionamento logico usando fatti e regole della KB
def inferenza_dati(percorso_file_kb, df):
    dizionario_propieta = {}
    dizionario_dimensione = {}

    try:
        # Crea un'istanza di Prolog
        prolog = Prolog()
        prolog.consult(percorso_file_kb)
    except Exception as e:
        print(f"Errore durante il caricamento della KB: {e}")

    results = prolog.query("affitto_costose(ID,Città, PrezzoAffitto,Media)")
    for result in results:
        dizionario_propieta[result['ID']] = "costoso"

    results = prolog.query("affitto_economico(ID,Città, PrezzoAffitto,Media)")
    for result in results:
        dizionario_propieta[result['ID']] = "economico"


    results = prolog.query("affitto_medie(ID,Città, PrezzoAffitto,Media)")
    for result in results:
        dizionario_propieta[result['ID']] = "medio"

    # Query per i monolocali
    results = prolog.query("monolocale(ID)")
    for result in results:
        dizionario_dimensione[result['ID']] = "monolocale"

    # Query per i bilocali
    results = prolog.query("bilocale(ID)")
    for result in results:
        dizionario_dimensione[result['ID']] = "bilocale"


    # Query per i trilocali
    results = prolog.query("trilocale(ID)")
    for result in results:
        dizionario_dimensione[result['ID']] = "trilocale"

    # Query per le proprietà per famiglie
    results = prolog.query("proprieta_per_famiglie(ID)")
    for result in results:
        dizionario_dimensione[result['ID']] = "per famiglie"

    df['Tipologia_Affitti'] = df['ID'].map(dizionario_propieta)
    print(df['Tipologia_Affitti'].value_counts())
    df['Tipologia_Propieta'] = df['ID'].map(dizionario_dimensione)
    print("Inferenza dati completata e colonne aggiunte al dataset")
    return df

