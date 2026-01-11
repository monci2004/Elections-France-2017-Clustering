import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

# =============================================================================
# ETAPE 1 : CONNEXION ET CHARGEMENT (EXTRACT)
# =============================================================================
# Configuration de la base de donnees
user = 'root'
password = quote_plus('MOn_MOT_DE_PASSE_ICI_avec des character special') # Gere les caracteres speciaux
host = 'localhost'
db_name = 'projet_elections'

# Creation du lien avec MySQL
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db_name}")

print("--- 1. CHARGEMENT DES DONNEES ---")
# On recupere tout le contenu de la table 'votes'
df_raw = pd.read_sql("SELECT * FROM votes", engine)
print(f"Donnees chargees : {df_raw.shape[0]} departements.")


# =============================================================================
# ETAPE 2 : NETTOYAGE ET NORMALISATION 
# =============================================================================
print("\n--- 2. PREPARATION ---")
# On utilise le nom du departement comme index (etiquette)
df = df_raw.set_index('Dpt')

# On ne garde que les colonnes des candidats
candidats = ['Arthaud', 'Asselineau', 'Cheminade', 'DupontAignan', 
             'Fillon', 'Hamon', 'Lasalle', 'LePen', 'Macron', 'Melenchon', 'Poutou']
df_votes = df[candidats]

# CRUCIAL : On transforme les votes bruts en POURCENTAGES (%)
# Cela permet de comparer les convictions politiques sans etre biaise par la population.
# axis=0 : divise les colonnes / axis=1 : par la somme de la ligne
df_clean = df_votes.div(df_votes.sum(axis=1), axis=0) * 100

# On remplace les eventuels trous (NaN) par 0 pour eviter les erreurs
df_clean = df_clean.fillna(0)


# =============================================================================
# eTAPE 3 : CALCULS MATHeMATIQUES (Avant de dessiner)
# =============================================================================
print("\n--- 3. CALCULS EN COURS ---")

# A. Pour le Dendrogramme (CAH)
# On calcule la matrice de distances avec la methode 'Ward' (minimise la variance intra-classe)
print("- Calcul de la hierarchie (Linkage)...")
Z = linkage(df_clean, method='ward', metric='euclidean')

# B. Pour la methode du Coude (K-Means)
# On lance l'algo 15 fois pour voir comment l'erreur (inertie) diminue
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

# On prepare une grande fenetre avec 2 graphiques l'un sous l'autre (nrows=2, ncols=1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12)) 
fig.suptitle('Analyse de Clustering des Votes 2017', fontsize=16)

# --- GRAPHIQUE 1 (HAUT) : LE DENDROGRAMME ---
ax1.set_title("Classification Ascendante Hierarchique (CAH)")
ax1.set_ylabel("Distance (Dissimilarite)")
# On dessine l'arbre dans la case du haut (ax=ax1)
dendrogram(Z, labels=df_clean.index, leaf_rotation=90, leaf_font_size=8, ax=ax1)
# Ligne rouge indicative pour suggerer une coupure a 150 (environ 3-4 groupes)
ax1.axhline(y=150, c='red', linestyle='--', label="Seuil de coupure")
ax1.legend()

# --- GRAPHIQUE 2 (BAS) : LA MeTHODE DU COUDE ---
ax2.set_title("Methode du Coude (Choix du nombre K optimal)")
ax2.set_xlabel("Nombre de groupes (K)")
ax2.set_ylabel("Inertie (WSS)")
# On dessine la courbe bleue avec des points ronds
ax2.plot(K_range, inertia, marker='o', linestyle='-', color='blue')
ax2.grid(True) # Ajoute une grille pour faciliter la lecture

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuste les marges proprement
plt.show() # Affiche la fenetre 


# =============================================================================
# eTAPE 5 : INTERPReTATION DES ReSULTATS 
# =============================================================================
print("\n--- 5. ANALYSE DES PROFILS ---")

# On fixe le nombre de groupes a 4 (d'apres l'observation du Coude souvent K=4)
k_final = 4
kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)

# On ajoute une colonne 'Groupe' a nos donnees
df_clean['Groupe'] = kmeans_final.fit_predict(df_clean)

# On affiche la moyenne des votes pour chaque groupe (arrondi a 1 chiffre apres la virgule)
print(f"Moyenne des votes par Groupe (K={k_final}) :")
print(df_clean.groupby('Groupe').mean().round(1))

# On affiche quels departements sont dans quel groupe
print("\n--- Detail des groupes ---")
for i in range(k_final):
    noms_dpt = df_clean[df_clean['Groupe'] == i].index.tolist()
    print(f"\n GROUPE {i} ({len(noms_dpt)} dpts) :")
    print(noms_dpt[:10], "..." if len(noms_dpt) > 10 else "") # Affiche les 10 premiers
    
    
    
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score

# =============================================================================
# eTAPE 6 : COMPARAISON ET VALIDATION (Le Bonus du PDF)
# =============================================================================
print("\n--- 6. COMPARAISON CAH vs K-MEANS ---")

# 1. On recupere les etiquettes de groupes de la CAH (en coupant l'arbre a 4 groupes)
# Note : 't=k_final' signifie qu'on veut le meme nombre de groupes que K-Means
labels_cah = fcluster(Z, t=k_final, criterion='maxclust')

# 2. On a deja les etiquettes K-Means dans df_clean['Groupe']
labels_kmeans = df_clean['Groupe']

# 3. Tableau Croise (Matrice de confusion)
# Cela permet de voir : "Est-ce que le Groupe 1 de la CAH correspond au Groupe 1 du K-Means ?"
print("Tableau de comparaison (Lignes=CAH, Colonnes=K-Means) :")
print(pd.crosstab(labels_cah, labels_kmeans, rownames=['CAH'], colnames=['K-Means']))

# 4. Calcul de l'indice de Rand Ajuste (ARI)
# 1.0 = Identique | 0.0 = Aleatoire
ari = adjusted_rand_score(labels_cah, labels_kmeans)
print(f"\nSCORE DE SIMILARITe (Rand Index) : {ari:.3f}")

if ari > 0.7:
    print(">> Conclusion : Les deux methodes donnent des resultats tres coherents !")
else:
    print(">> Conclusion : Les methodes trouvent des nuances differentes.")
    
    
    
    
    
    
    
# ==================================================================================================================================
# eTAPE 7 : EXPORT FINAL (ici jai fait des correction parceque POWERBI naffiche pas tout les departement correctement)
# ==================================================================================================================================
print("\n--- 7. EXPORT DES DONNeES ---")

df_export = df_clean.copy()
df_export.index = df_export.index.astype(str).str.strip()
df_export['Nom_Departement'] = df_export.index

# 1. NETOYAGE AVEC LA ReGLE DE BASE : "Nom + France"
df_export['Localisation_PowerBI'] = df_export['Nom_Departement'] + ", France"

# 2. CORRECTIONS SPeCIFIQUES (On ecrase les valeurs par defaut)

# --- A. LE CAS WALLIS (On vise la ville de Mata-Utu) ---
# C'est la seule facon d'eviter la Suisse (Valais) ou Belfort (Territoire), cest ici ou POWERBI CONFENDRE ENTRE DES DEPARTEMENT ET DES AUTRE VILLES
nom_wallis_index = "Wallis et Futuna"
if nom_wallis_index in df_export.index:
    df_export.loc[nom_wallis_index, "Localisation_PowerBI"] = "Mata-Utu, Wallis and Futuna"
elif "Wallis-et-Futuna" in df_export.index:
    df_export.loc["Wallis-et-Futuna", "Localisation_PowerBI"] = "Mata-Utu, Wallis and Futuna"

# --- B. LES CONFLITS MeTROPOLE (Lot, Vienne, Belfort) ---
if "Lot" in df_export.index:
    df_export.loc["Lot", "Localisation_PowerBI"] = "Lot Department, France"

if "Vienne" in df_export.index:
    df_export.loc["Vienne", "Localisation_PowerBI"] = "Vienne Department, France"

# On securise Belfort pour qu'il ne vole pas la vedette
if "Territoire de Belfort" in df_export.index:
    df_export.loc["Territoire de Belfort", "Localisation_PowerBI"] = "Territoire-de-Belfort, France"

# --- C. LES AUTRES CLASSIQUES ---
corrections_classiques = {
    "Nord": "Nord Department, France",
    "Somme": "Somme Department, France",
    "Cher": "Cher Department, France",
    "Aube": "Aube Department, France",
    "Loire": "Loire Department, France",
    "Rhone": "Rhone Department, France",
    "Rhone": "Rhone Department, France",
    "Paris": "Paris Department, France",
    # Outre-mer (Noms anglais sans ", France")
    "Polynesie francaise": "French Polynesia",
    "Nouvelle-Caledonie": "New Caledonia",
    "Guyane": "French Guiana",
    "Guyane francaise": "French Guiana",
    "La Reunion": "Reunion",
    "La Reunion": "Reunion",
    "Saint-Pierre-et-Miquelon": "Saint Pierre and Miquelon",
    "Mayotte": "Mayotte",
    "Martinique": "Martinique",
    "Guadeloupe": "Guadeloupe"
}

for nom, loc in corrections_classiques.items():
    if nom in df_export.index:
        df_export.loc[nom, "Localisation_PowerBI"] = loc

# 3. SAUVEGARDE
nom_fichier = 'resultats_clustering_DEPARTEMENTS_final.csv'
df_export.to_csv(nom_fichier, sep=',', encoding='utf-8')

print(f"Fichier '{nom_fichier}' genere.")
print("--- Verification ---")
print("Wallis est envoye a :")
check = df_export[df_export['Nom_Departement'].str.contains("Wallis")][['Localisation_PowerBI']]
print(check)