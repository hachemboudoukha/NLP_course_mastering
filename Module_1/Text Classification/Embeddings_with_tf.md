# Utilisation des Embeddings avec l'API Feature Column

Plutôt que de créer manuellement une matrice d'embeddings, TensorFlow propose une API dédiée pour intégrer et entraîner automatiquement ces couches au sein d'un modèle LSTM.

## A. L'API Feature Column
L'utilisation de `tf.feature_column.embedding_column` permet d'automatiser la création de la matrice d'embeddings. Elle sera entraînée en même temps que le reste du modèle LSTM pour optimiser la représentation vectorielle des mots.



---

## B. Colonnes Catégorielles Séquentielles
Comme nous travaillons avec des séquences (données avec des étapes temporelles), nous utilisons une extension spécifique de l'API.

### 1. Sélection de la colonne
Il existe plusieurs fonctions selon le type de traitement, mais la plus courante pour des mots déjà transformés en identifiants entiers est :
* **`sequence_categorical_column_with_identity`** : Utilise directement les IDs uniques des mots.

### 2. Paramètres requis
* **Clé (String) :** Un nom arbitraire qui servira d'identifiant dans le dictionnaire de données.
* **Taille du vocabulaire :** Le nombre total de mots uniques.
* **Dimension de l'embedding :** Généralement fixée à la racine quatrième de la taille du vocabulaire ($\sqrt[4]{vocab\_size}$).

---

## C. Conversion en Couche d'Entrée (Sequence Input Layer)
La fonction `sequence_input_layer` est l'étape finale qui transforme les données brutes en vecteurs denses utilisables par le LSTM.



### Fonctionnement du processus :
1. **Dictionnaire d'entrée :** On crée un dictionnaire où la clé est le nom défini précédemment et la valeur est le lot de séquences tokenisées.
2. **Liste de colonnes :** On passe une liste contenant notre colonne d'embedding.
3. **Sortie :** La fonction renvoie un tuple contenant :
    * Les **séquences d'embeddings** (prêtes pour le LSTM).
    * Les **longueurs de séquences** (utile pour gérer le padding).

---

### Résumé du flux de travail
| Étape | Composant | Rôle |
| :--- | :--- | :--- |
| **1. Identité** | `sequence_categorical_column` | Définit la source des données (IDs de mots). |
| **2. Embedding** | `embedding_column` | Définit la taille des vecteurs de sortie. |
| **3. Couche** | `sequence_input_layer` | Exécute la transformation réelle en tenseurs. |