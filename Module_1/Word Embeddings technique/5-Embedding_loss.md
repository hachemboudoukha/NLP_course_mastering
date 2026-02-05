# Perte d'Embedding (Embedding Loss)

## 1. Pourquoi l'échantillonnage de candidats (Candidate Sampling) ?

Calculer une perte **Softmax** sur l'intégralité d'un dictionnaire est extrêmement gourmand en ressources (trop de calculs pour chaque mot du vocabulaire). Pour optimiser l'entraînement, on utilise l'échantillonnage de candidats : on ne calcule la perte que sur un petit sous-ensemble de mots.

Il existe deux fonctions de perte principales pour cela :
* **Sampled Softmax** : Une approximation du Softmax qui n'utilise que le mot correct et quelques mots choisis au hasard.
* **NCE Loss (Noise-Contrastive Estimation)** : Transforme le problème en une **classification binaire**. Le modèle doit distinguer le vrai mot de contexte du "bruit" (mots aléatoires).



---

## 2. La logique de la NCE Loss

L'idée est d'entraîner le modèle à :
1.  Attribuer une **probabilité élevée** au vrai mot de contexte (label positif).
2.  Attribuer une **probabilité faible** aux mots échantillonnés aléatoirement (labels négatifs/bruit).

Cette approche est plus simple et plus rapide, ce qui en fait le standard pour l'entraînement des embeddings. Pour l'évaluation finale (test), on utilise généralement une perte classique (Softmax ou Sigmoid Cross Entropy) pour obtenir une précision réelle.

---

# Implémentation : La fonction `calculate_loss`

L'objectif est d'utiliser `tf.nn.nce_loss` pour calculer l'erreur entre nos vecteurs d'entrée (embeddings) et les mots de contexte réels.

### Code Python (TensorFlow)

```python
import tensorflow as tf

class EmbeddingModel(object):
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    def get_bias_weights(self):
        """Initialise les paramètres de poids et de biais pour la couche de sortie."""
        weights_initializer = tf.zeros([self.vocab_size, self.embedding_dim])
        bias_initializer = tf.zeros([self.vocab_size])
        
        weights = tf.compat.v1.get_variable('weights', initializer=weights_initializer)
        bias = tf.compat.v1.get_variable('bias', initializer=bias_initializer)
        return weights, bias
    
    def calculate_loss(self, embeddings, context_ids, num_negative_samples):
        # 1. Récupération des poids et biais de sortie
        weights, bias = self.get_bias_weights()
        
        # 2. Calcul des pertes individuelles avec NCE
        nce_losses = tf.nn.nce_loss(
            weights=weights,
            biases=bias,
            labels=context_ids,
            inputs=embeddings,
            num_sampled=num_negative_samples,
            num_classes=self.vocab_size
        )
        
        # 3. Calcul de la perte moyenne pour le batch (Overall Loss)
        overall_loss = tf.math.reduce_mean(nce_losses)
        
        return overall_loss