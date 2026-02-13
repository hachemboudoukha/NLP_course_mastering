# Implémentation LSTM

## 1. Empilement de Couches (Stacked RNNs)
Pour améliorer les performances sur des jeux de données complexes, il est possible d'empiler plusieurs couches de cellules LSTM.

* **Principe :** La sortie de la première couche devient l'entrée de la couche suivante pour chaque étape temporelle.
* **États séparés :** Chaque couche maintient son propre état (mémoire) ; il n'y a pas de connexion récurrente directe entre les cellules de couches différentes.
* **Hiérarchie des caractéristiques :** Plus on ajoute de couches, plus le modèle peut extraire des motifs abstraits et complexes.



---

## 2. Implémentation avec TensorFlow
La fonction principale utilisée est `tf.keras.layers.RNN` (ou directement `tf.keras.layers.LSTM`).

### Configuration Clé
* **Cellules :** On peut passer une cellule unique (`LSTMCell`) ou un groupe empilé (`StackedRNNCells`).
* **Dimensions de sortie :** La sortie possède généralement trois dimensions : `[Batch Size, Sequence Length, Hidden Units]`.
* **Optimisation des séquences :** * Les séquences réelles ont souvent des longueurs variables et contiennent du **padding** (remplissage).
    * L'argument `input_length` permet au RNN d'ignorer les calculs inutiles sur les parties "remplies", ce qui accélère considérablement l'entraînement.

---

## 3. Calcul de la Perte (Loss) et Logits
L'entraînement d'un modèle de langage est traité comme une **classification multiclasse**.

### Conversion en Logits
On utilise une couche **Dense** (entièrement connectée) finale pour transformer la sortie du LSTM en **logits**. Chaque logit correspond à un mot du vocabulaire.

### Le Masque de Padding (Padding Mask)
Il est crucial de ne pas pénaliser le modèle pour ses prédictions sur les jetons de remplissage (padding).

1. **Création du masque :** Un tenseur de 0 et de 1 de la même taille que les cibles (1 pour un vrai mot, 0 pour du padding).
2. **Application :** On multiplie la perte calculée par ce masque.
3. **Résultat :** Les erreurs sur les parties inutiles de la séquence sont mises à zéro et ignorées par l'algorithme d'optimisation.



---

### Comparaison des fonctions de perte
| Type de Label | Fonction TensorFlow |
| :--- | :--- |
| **One-hot vectors** | `tf.nn.softmax_cross_entropy_with_logits` |
| **Index de classe (Sparse)** | `tf.nn.sparse_softmax_cross_entropy_with_logits` |