# Résumé du Chapitre : Modèles Skip-gram et CBOW

## 1. Deux Approches pour les Embeddings

Pour entraîner un modèle à créer des vecteurs de mots (embeddings), nous utilisons des paires d'entraînement basées sur la relation entre un mot cible (**target**) et ses voisins (**context**).

### A. Le Modèle Skip-gram
Le modèle **Skip-gram** tente de prédire les mots du contexte à partir d'un mot cible.
* **Structure des paires** : `(target, context_word)`.
* **Fonctionnement** : Chaque fenêtre de contexte génère plusieurs paires d'entraînement.
* **Exemple** : Pour "le chat *mange* sa souris" (cible = mange), on crée :
    * `(mange, le)`, `(mange, chat)`, `(mange, sa)`, `(mange, souris)`.

### B. Le Modèle CBOW (Continuous Bag of Words)
Le modèle **CBOW** fait l'inverse : il tente de prédire le mot cible à partir de l'ensemble des mots du contexte.
* **Structure des paires** : `([context_words], target)`.
* **Fonctionnement** : On utilise la **moyenne** des vecteurs de contexte pour prédire la cible. Une fenêtre ne génère qu'une seule paire d'entraînement.
* **Exemple** : `([le, chat, sa, souris], mange)`.



---

## 2. Comparaison : Skip-gram vs CBOW

| Caractéristique | Skip-gram | CBOW |
| :--- | :--- | :--- |
| **Paires d'entraînement** | Target $\rightarrow$ Context | Context $\rightarrow$ Target |
| **Vitesse d'entraînement** | Plus lent (génère plus de paires) | Plus rapide |
| **Volume de données** | Fonctionne bien avec peu de données | Nécessite plus de données |
| **Précision** | Meilleur pour les **mots rares** | Meilleur pour les **mots fréquents** |

