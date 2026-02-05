# Similarité Cosinus (Cosine Similarity)

## 1. Mesurer la Similarité entre Vecteurs

Une fois que les mots sont convertis en vecteurs (embeddings), nous avons besoin d'une mesure pour comparer leur proximité sémantique. La méthode la plus utilisée en NLP est la **similarité cosinus**.

* **Le Principe** : On ne mesure pas la distance "euclidienne" (la ligne droite entre deux points), mais l'**angle** entre deux vecteurs.
* **L'Avantage** : Cela permet de comparer des mots même si leurs vecteurs n'ont pas la même magnitude (longueur). On se concentre uniquement sur leur direction dans l'espace multidimensionnel.



---

## 2. Interprétation de la Corrélation

La similarité cosinus produit un score situé entre **-1** et **1** :

| Score | Signification | Exemple |
| :--- | :--- | :--- |
| **1** | Vecteurs identiques (Parfaite corrélation) | "Orange" et "Orange" |
| **Proche de 1** | Mots très liés ou synonymes | "Orange" et "Jus" |
| **0** | Vecteurs orthogonaux (Aucune corrélation) | "Chocolat" et "Clôture" |
| **Négatif** | Sens opposés (Antonymes) | "Bon" et "Mauvais" |



---

## 3. Normalisation L2

Pour calculer facilement la similarité cosinus avec un produit scalaire, les vecteurs doivent être **normalisés**. La normalisation L2 ramène la longueur de chaque vecteur à **1**. 

Dans TensorFlow, on utilise `tf.math.l2_normalize`. Pour une matrice, il est crucial de spécifier `axis=1` pour normaliser chaque mot individuellement.

---

# Implémentation : La fonction `compute_cos_sims`

L'objectif est de comparer un mot cible à l'intégralité du vocabulaire contenu dans la matrice d'embedding.

### Code Python (TensorFlow)

```python
import tensorflow as tf

class EmbeddingModel(object):
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)
        self.embedding_matrix = None

    def forward(self, target_ids):
        # Initialisation et récupération de la matrice d'embedding
        initial_bounds = 0.5 / self.embedding_dim
        initializer = tf.random.uniform(
            [self.vocab_size, self.embedding_dim],
            minval=-initial_bounds,
            maxval=initial_bounds)
        
        self.embedding_matrix = tf.compat.v1.get_variable('embedding_matrix',
            initializer=initializer)
        
        return tf.compat.v1.nn.embedding_lookup(self.embedding_matrix, target_ids)
    
    def compute_cos_sims(self, word, training_texts):
        # 1. Identification du mot cible
        self.tokenizer.fit_on_texts(training_texts)
        word_id = self.tokenizer.word_index[word]
        word_embedding = self.forward([word_id]) # Récupère le vecteur du mot
        
        # 2. Normalisation L2 (Vecteur et Matrice complète)
        normalized_embedding = tf.math.l2_normalize(word_embedding)
        normalized_matrix = tf.math.l2_normalize(self.embedding_matrix, axis=1)
        
        # 3. Calcul de la similarité (Produit scalaire via multiplication matricielle)
        # transpose_b=True permet de multiplier (1, dim) par (dim, vocab_size)
        cos_sims = tf.linalg.matmul(
            normalized_embedding, 
            normalized_matrix,
            transpose_b=True
        )
        
        return cos_sims