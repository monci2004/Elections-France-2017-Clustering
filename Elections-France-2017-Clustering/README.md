# 🗳️ Analyse des Élections Présidentielles France 2017 (Clustering)

![Tableau de bord Power BI](images/carte_resultats.png)

## 📄 Description
Ce projet vise à analyser les comportements électoraux en France par départements et circonscriptions.
L'objectif est d'utiliser le **Machine Learning (K-Means & CAH)** pour regrouper les territoires ayant des profils de vote similaires, sans a priori politique, et de visualiser ces fractures territoriales.

## 🛠️ Technologies Utilisées
* **Langage :** Python 3.9
* **Analyse de données :** Pandas, NumPy, Scikit-learn (Clustering)
* **Base de données :** MySQL (Extraction via SQLAlchemy)
* **Visualisation :** Power BI (Tableau de bord interactif) & Matplotlib

## 🚀 Fonctionnalités Clés
1.  **Nettoyage de données (ETL) :** Traitement des votes bruts, gestion des valeurs manquantes et normalisation en pourcentages.
2.  **Clustering Non-Supervisé :**
    * Détection du nombre optimal de groupes via la méthode du Coude (Elbow Method).
    * Segmentation des départements en 4 profils types (K-Means).
3.  **Géolocalisation Avancée :** Correction des données géographiques pour Power BI (traitement spécifique des codes ISO pour Wallis-et-Futuna, DOM-TOM, et préfectures).
4.  **Storytelling :** Visualisation interactive des clusters via Radar Charts (profils types) et Cartes Choroplèthes.

## 📊 Aperçu des Résultats & Insights
Le modèle a identifié 4 groupes distincts (Clusters) :
* **Groupe 0 (Conservateur/Nationaliste) :** Forte prédominance dans le Nord-Est et le Sud-Est.
* **Groupe 1 (Centriste/Urbain) :** Vote majoritaire dans les grandes métropoles (Paris, Lyon, Bordeaux).
* **Groupe 2 (Gauche/Insoumis) :** Présent dans le Sud-Ouest et certaines banlieues.
* **Groupe 3 (Spécificités Outre-mer) :** Dynamiques locales propres aux DOM-TOM.

### 💡 Cas d'étude : L'Alsace
L'algorithme a correctement identifié une homogénéité politique forte dans le Bas-Rhin (67) et le Haut-Rhin (68), classés dans le même cluster conservateur, validant la pertinence du modèle par rapport à la réalité historique de la région.

## ⚙️ Installation et Exécution
1. Cloner le projet :
   ```bash
   git clone [https://github.com/](https://github.com/)moncif2004/Elections-France-2017-Clustering.git