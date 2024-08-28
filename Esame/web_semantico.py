from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import time

from SPARQLWrapper import SPARQLWrapper, JSON

import time
from SPARQLWrapper import SPARQLWrapper, JSON

def get_population_wikidata(dizionario):
    for city in dizionario['regio2']:
        city1 = city.split("_")[0]
        print(f"Ricerca della popolazione di {city1} su Wikidata")
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setReturnFormat(JSON)

        query = f"""
            SELECT ?population WHERE {{
            ?city rdfs:label "{city1}"@en ;
                  wdt:P1082 ?population .
            }} LIMIT 1
        """
        time.sleep(1)  # Per evitare di sovraccaricare il server di Wikidata
        sparql.setQuery(query)
        results = sparql.query().convert()

        if 'results' in results and results['results']['bindings']:
            population = results['results']['bindings'][0]['population']['value']
            print(f"La popolazione di {city} è {population}")
            dizionario[city] = population
        else:
            dizionario[city] = "Data not available"

    return dizionario



