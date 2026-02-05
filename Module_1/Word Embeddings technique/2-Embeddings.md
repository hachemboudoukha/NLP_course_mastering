# Word Embeddings (Plongements Lexicaux)

## 1. Pourquoi utiliser des Embeddings ?

* **Le problème des IDs entiers** : Les identifiants numériques (ex: 14 pour "lait", 103 pour "céréales") ne capturent aucune relation de sens. Mathématiquement, la distance entre 14 et 103 n'a aucun rapport avec la similarité des mots.
* **La solution vectorielle** : Les **embeddings** convertissent chaque mot en un vecteur dans un espace multidimensionnel. 
    * Les mots utilisés dans des contextes similaires se retrouvent **proches géométriquement** dans cet espace.
    * Cela permet au modèle de comprendre que "chat" est plus proche de "chien" que de "voiture".



---

## 2. Dimensionnalité et Taille des Vecteurs

Le choix de la dimension des vecteurs (le nombre de valeurs par mot) est crucial :
* **Vecteurs larges** : Capturent mieux les relations complexes mais consomment plus de ressources et risquent le surapprentissage (*overfitting*).
* **Règle empirique** : Une bonne base de départ est de régler la dimension sur la racine quatrième de la taille du vocabulaire :
$$\text{dim} = \sqrt[4]{\text{vocab\_size}}$$

---

## 3. Le concept de Cible et Contexte (Target-Context)

Les relations entre les mots sont apprises en observant quels mots entourent un mot donné.
* **Mot Cible (Target)** : Le mot central que l'on étudie.
* **Fenêtre de contexte (Context Window)** : Les mots adjacents. 
* **Taille de la fenêtre** : Doit être un nombre **impair** pour permettre une symétrie (ex: une taille de 5 signifie 2 mots à gauche, le mot cible, et 2 mots à droite).



---

# Implémentation : Préparation des données d'entraînement

Pour entraîner un modèle d'embedding, nous devons extraire les cibles et leurs indices de contexte.

### Fonctions Utilitaires

```python
import tensorflow as tf

def get_target_and_size(sequence, target_index, window_size):
    """Récupère le mot cible et calcule la demi-fenêtre."""
    target_word = sequence[target_index]
    # Calcul de la portée symétrique (arrondi à l'inférieur)
    half_window_size = window_size // 2
    return target_word, half_window_size

def get_window_indices(sequence, target_index, half_window_size):
    """Calcule les limites de la fenêtre avec sécurité pour les bords."""
    # max(0, ...) empêche de sortir par la gauche (index négatif)
    left_incl = max(0, target_index - half_window_size)
    # min(len, ...) empêche de sortir par la droite
    right_excl = min(len(sequence), target_index + half_window_size + 1)
    return (left_incl, right_excl)