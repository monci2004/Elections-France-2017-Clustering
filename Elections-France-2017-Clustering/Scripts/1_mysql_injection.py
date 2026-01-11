import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus  # <--- AJOUTE CET IMPORT

# 1. Configuration de la connexion
user = 'root'
raw_password = 'MOn_MOT_DE_PASSE_ICI_avec des character special'  
host = 'localhost'
db_name = 'projet_elections'

# ON ENCODE LE MOT DE PASSE POUR GeRER DES CAS SPECIALES ET DES CHARACTER  
encoded_password = quote_plus(raw_password)

# On cree le moteur de connexion avec le mot de passe encode

connection_str = f"mysql+pymysql://{user}:{encoded_password}@{host}/{db_name}"
engine = create_engine(connection_str)

# 2. Chargement du CSV
print("Lecture du CSV...")
df = pd.read_csv("Presidentielle2017_par_departement.csv", sep=",") 

# 3. Envoi vers MySQL
print("ecriture dans MySQL...")
try:
    df.to_sql('votes', con=engine, if_exists='replace', index=False)
    print("Succes ! Tes donnees sont maintenant dans MySQL.")
except Exception as e:
    print("Erreur lors de l'ecriture :", e)