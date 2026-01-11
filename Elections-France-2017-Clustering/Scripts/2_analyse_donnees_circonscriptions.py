import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# =============================================================================
# eTAPE 1 : CONNEXION ET CHARGEMENT (EXTRACT)
# =============================================================================
# Configuration de la base de donnees
user = 'root'
password = quote_plus('MOn_MOT_DE_PASSE_ICI_avec des character special') 
host = 'localhost'
db_name = 'projet_elections'

# Creation du lien avec MySQL
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db_name}")

print("--- 1. CHARGEMENT DES DONNeES ---")
# On recupere tout le contenu de la table 'votes_circonscription'
df_raw = pd.read_sql("SELECT * FROM votes_circonscription", engine)
print(f"Donnees chargees : {df_raw.shape[0]} circonscriptions.")


# =============================================================================
# eTAPE 2 : NETTOYAGE ET NORMALISATION (TRANSFORM)
# =============================================================================
print("\n--- 2. PRePARATION ---")

# --- MODIFICATION MAJEURE ICI (ID UNIQUE) ---
# On combine CodeDpt et CodeCirco pour avoir un identifiant unique (ex: "67-1")
# On convertit en string pour etre sur de concatener du texte
df_raw['CodeDpt'] = df_raw['CodeDpt'].astype(str)
df_raw['CodeCirco'] = df_raw['CodeCirco'].astype(str)

df_raw['ID_Unique'] = df_raw['CodeDpt'] + "-" + df_raw['CodeCirco']

# On utilise cet ID comme index
df = df_raw.set_index('ID_Unique')
print(f"Index cree (exemple) : {df.index[0]}")

# On ne garde que les colonnes chiffrees des candidats
candidats = ['Arthaud', 'Asselineau', 'Cheminade', 'DupontAignan', 
             'Fillon', 'Hamon', 'Lasalle', 'LePen', 'Macron', 'Melenchon', 'Poutou']

# Securite : on ne prend que les colonnes qui existent vraiment
cols_presentes = [c for c in candidats if c in df.columns]
df_votes_circonscriptions = df[cols_presentes]

# CRUCIAL : On transforme les votes bruts en POURCENTAGES (%)
df_clean = df_votes_circonscriptions.div(df_votes_circonscriptions.sum(axis=1), axis=0) * 100

# On remplace les eventuels trous (NaN) par 0
df_clean = df_clean.fillna(0)


# =============================================================================
# eTAPE 3 : CALCULS MATHeMATIQUES
# =============================================================================
print("\n--- 3. CALCULS EN COURS ---")

# A. Pour le Dendrogramme (CAH)
print("- Calcul de la hierarchie (Linkage)...")
Z = linkage(df_clean, method='ward', metric='euclidean')

# B. Pour la methode du Coude (K-Means)
print("- Test de 1 a 15 groupes pour le K-Means...")
inertia = []
K_range = range(1, 15)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_clean)
    inertia.append(kmeans.inertia_)


# =============================================================================
# eTAPE 4 : CReATION DE LA FIGURE (VISUALISATION)
# =============================================================================
print("\n--- 4. AFFICHAGE DES GRAPHIQUES ---")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12)) 
fig.suptitle('Analyse de Clustering des Votes 2017 (Circonscriptions)', fontsize=16)

# --- GRAPHIQUE 1 (HAUT) : LE DENDROGRAMME ---
ax1.set_title("Classification Ascendante Hierarchique (CAH)")
ax1.set_ylabel("Distance")
# Note : On ajoute 'truncate_mode' pour afficher un resume, sinon c'est illisible avec 577 lignes
dendrogram(Z, truncate_mode='lastp', p=30, labels=df_clean.index, leaf_rotation=90, leaf_font_size=8, ax=ax1)
ax1.axhline(y=150, c='red', linestyle='--', label="Seuil de coupure")
ax1.legend()

# --- GRAPHIQUE 2 (BAS) : LA MeTHODE DU COUDE ---
ax2.set_title("Methode du Coude (Choix du nombre K optimal)")
ax2.set_xlabel("Nombre de groupes (K)")
ax2.set_ylabel("Inertie (WSS)")
ax2.plot(K_range, inertia, marker='o', linestyle='-', color='blue')
ax2.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# =============================================================================
# eTAPE 5 : INTERPReTATION ET ZOOM ALSACE
# =============================================================================
print("\n--- 5. ANALYSE DES PROFILS ---")

k_final = 4
kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)

# On ajoute une colonne 'Groupe' a nos donnees
df_clean['Groupe'] = kmeans_final.fit_predict(df_clean[cols_presentes])

# Affichage des moyennes
print(f"Moyenne des votes par Groupe (K={k_final}) :")
print(df_clean.groupby('Groupe')[cols_presentes].mean().round(1))

# --- AJOUT SPECIAL : ZOOM SUR L'ALSACE (Dpt 67 et 68) ---
print("\n" + "="*40)
print(" ReSULTATS SPeCIAUX : ALSACE (67 & 68) ")
print("="*40)

# On filtre les index qui commencent par '67' ou '68'
# L'index est de type string ("67-1"), donc startswith marche tres bien
index_str = df_clean.index.astype(str)
is_alsace = index_str.str.startswith(('67', '68'))
df_alsace = df_clean[is_alsace]

if not df_alsace.empty:
    print(f"Trouve {len(df_alsace)} circonscriptions en Alsace.")
    
    # Repartition des groupes
    repartition = df_alsace['Groupe'].value_counts()
    print("\nRepartition des Groupes en Alsace :")
    print(repartition)
    
    # Groupe dominant
    groupe_dominant = repartition.idxmax()
    print(f"\n>> CONCLUSION : L'Alsace est majoritairement dans le GROUPE {groupe_dominant}.")
    
    # Analyse du vote de ce groupe
    moyenne_groupe = df_clean[df_clean['Groupe'] == groupe_dominant][cols_presentes].mean()
    top3 = moyenne_groupe.sort_values(ascending=False).head(3)
    print(f"Ce Groupe {groupe_dominant} vote principalement pour :")
    print(top3.round(1))
else:
    print("Aucune circonscription alsacienne trouvee (codes 67/68).")


# =============================================================================
# eTAPE 6 : COMPARAISON CAH vs K-MEANS
# =============================================================================
print("\n--- 6. COMPARAISON CAH vs K-MEANS ---")

# 1. etiquettes CAH
labels_cah = fcluster(Z, t=k_final, criterion='maxclust')

# 2. etiquettes K-Means
labels_kmeans = df_clean['Groupe']

# 3. Comparaison
ari = adjusted_rand_score(labels_cah, labels_kmeans)
print(f"SCORE DE SIMILARITe (Rand Index) : {ari:.3f}")

if ari > 0.7:
    print(">> Conclusion : Les deux methodes donnent des resultats tres coherents !")
else:
    print(">> Conclusion : Les methodes trouvent des nuances differentes.")
    
    


print("\n--- FIN DU SCRIPT ---")

# =============================================================================
# eTAPE 7 : EXPORT POUR POWER BI 
# =============================================================================
print("\n--- 7. EXPORT DES DONNeES ---")

# 1. On prepare la source pour qu'elle ait le meme Index que le tableau final
df_raw_indexed = df_raw.set_index('ID_Unique')

# 2. On cree la copie d'export
df_export = df_clean.copy()

# 3. On recupere le Nom du Departement 
col_nom_dpt = 'Dpt' 

if col_nom_dpt in df_raw_indexed.columns:
    df_export['Nom_Departement'] = df_raw_indexed[col_nom_dpt]
else:
    # Si on ne trouve pas le nom, on met le code par securite
    print(f"ATTENTION : Colonne '{col_nom_dpt}' introuvable. On utilise le code.")
    df_export['Nom_Departement'] = df_export.index.str.split('-').str[0]

# 4. CRUCIAL POUR POWER BI : Creation d'une colonne de geolocalisation
# Power BI place mal les departements s'il n'y a pas le pays.
# On cree : "Ain, France", "Aisne, France", etc.
df_export['Localisation_PowerBI'] = df_export['Nom_Departement'].astype(str) + ", France"

# 5. Sauvegarde
nom_fichier = 'resultats_clustering_powerbi.csv'
df_export.to_csv(nom_fichier, sep=',', encoding='utf-8')

print(f"Fichier '{nom_fichier}' genere.")
print("Verification d'une ligne au hasard :")
print(df_export[['Groupe', 'Nom_Departement', 'Localisation_PowerBI']].sample(1))