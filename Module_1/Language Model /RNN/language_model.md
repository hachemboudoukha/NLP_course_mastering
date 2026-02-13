# Language Model


## Fondamentaux 

### Probabilités de mots

L'objectif principal d'un modèle de langue est d'assigner des probabilités aux mots dans une séquence.

* **Classification multiclasse** : Le calcul de la probabilité d'un mot en fonction de ceux qui le précèdent est fondamentalement une tâche de classification où chaque mot du vocabulaire représente une classe distincte.
* **Contexte** : La probabilité de chaque mot est conditionnée par les mots précédents dans la séquence.

---

## Structure des données : input et target

Pour entraîner un modèle, nous ne créons pas de simples paires mot-cible, mais des séquences d'entrée et de cible de longueurs égales.

* **Séquence d'entrée (Input)** : La portion de texte fournie au modèle.
* **Séquence cible (Target)** : Il s'agit de la séquence d'entrée décalée d'un mot vers la droite.
* **Mécanisme de prédiction** : Le modèle tente de prédire chaque mot de la cible en se basant sur le préfixe correspondant dans l'entrée.

### Exemple de paires préfixe-cible

Pour la séquence "she bought a book", les paires apprises sont :

* `["she"]`  "bought"
* `["she", "bought"]`  "a"
* `["she", "bought", "a"]`  "book"

---

## Gestion de la longueur (Maximum Length)

Le paramètre `max_length` est crucial pour calibrer le modèle :

* **Performance** : Limiter la longueur augmente la vitesse d'entraînement.
* **Généralisation** : Cela aide à éviter le surapprentissage (overfitting) sur des dépendances textuelles rares présentes dans de très longues phrases.




---