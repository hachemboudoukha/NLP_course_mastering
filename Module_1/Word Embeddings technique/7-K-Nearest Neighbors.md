# Résumé du Chapitre : K-Plus Proches Voisins (K-Nearest Neighbors)

## 1. Analyser la Proximité entre Mots

Le concept de **K-Nearest Neighbors (KNN)** permet d'identifier les $K$ mots du vocabulaire dont les vecteurs sont les plus proches (ceux ayant la similarité cosinus la plus élevée) d'un mot donné.

* **Utilité** : C'est le meilleur outil pour évaluer la qualité d'un modèle d'embedding. 
* **Diagnostic** : 
    * Si les voisins de "ordinateur" sont "logiciel" et "clavier", le modèle a bien appris.
    * Si les voisins sont "cascade" et "océan", l'entraînement est probablement défaillant.
* **Influence du Corpus** : Les voisins changent selon le texte utilisé. Dans un corpus militaire, le mot "code" sera proche de "signal", alors que dans un corpus technique, il sera proche de "programme".



---

## 2. Processus de Calcul

Pour trouver les voisins, on suit trois étapes logiques :
1. **Calcul des similarités** : On compare le mot cible avec TOUS les autres mots du vocabulaire via la similarité cosinus.
2. **Nettoyage des dimensions** : Le résultat du calcul matriciel possède souvent une dimension superflue (ex: `[1, 5000]`). On utilise `tf.squeeze` pour obtenir un vecteur simple (`[5000]`).
3. **Extraction du Top K** : On utilise une fonction de tri efficace pour récupérer uniquement les $K$ valeurs les plus hautes.

---

# Implémentation : La fonction `k_nearest_neighbors`

Cette fonction utilise les outils de TensorFlow pour extraire les mots les plus similaires à partir de la matrice d'embedding.

### Code Python (TensorFlow)

```python
import tensorflow as tf

class EmbeddingModel(object):
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    def compute_cos_sims(self, word, training_texts):
        """Calcule les similarités cosinus (voir chapitre précédent)."""
        # ... (Logique de calcul des similarités)
        # Retourne un tenseur de forme (1, vocab_size)
        pass

    def k_nearest_neighbors(self, word, k, training_texts):
        # 1. Obtenir toutes les similarités pour le mot cible
        cos_sims = self.compute_cos_sims(word, training_texts)
        
        # 2. Supprimer la dimension de taille 1 (Squeeze)
        # Transforme (1, vocab_size) en (vocab_size)
        squeezed_cos_sims = tf.squeeze(cos_sims)
        
        # 3. Récupérer les K plus grandes valeurs
        # Retourne un tuple : (valeurs_de_similarité, indices_des_mots)
        top_k_output = tf.math.top_k(squeezed_cos_sims, k)
        
        return top_k_output