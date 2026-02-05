# La Matrice d'Embedding

## 1. Création de Variables avec `get_variable`

Contrairement aux couches denses classiques, nous devons ici gérer manuellement la matrice contenant les vecteurs.
* **`tf.compat.v1.get_variable`** : C'est la fonction clé pour créer la matrice d'embedding.
    * **Réutilisabilité** : Si une variable avec le même nom (ex: `'embedding_matrix'`) existe déjà, TensorFlow la récupère au lieu d'en créer une nouvelle. Cela permet de conserver les poids appris d'une itération à l'autre.
    * **Paramètres** : On définit principalement le nom, la forme (`shape`) et le type de données (`dtype`).

---

## 2. Initialisation et Distribution Uniforme

L'initialisation détermine les valeurs de départ de nos vecteurs avant l'entraînement.
* **`tf.random.uniform`** : On utilise souvent une distribution uniforme pour initialiser les poids.
* **Règle de calcul** : Pour une dimension d'embedding $d$, les bornes recommandées sont :
$$\text{bounds} = \pm \frac{0.5}{d}$$
Cela garantit que les valeurs initiales ne sont ni trop grandes (explosion du gradient) ni trop petites.

---

## 3. Recherche d'Embedding (Lookup)

Une fois la matrice créée, le passage "Forward" consiste à extraire les vecteurs correspondant aux mots de notre batch.
* **`tf.nn.embedding_lookup`** : Cette fonction agit comme un indexeur ultra-rapide. Au lieu de faire des multiplications matricielles complexes, elle va chercher directement la ligne $i$ de la matrice correspondant à l'ID du mot.



---

# Implémentation : La fonction `forward`

L'objectif est de créer la matrice lors du premier appel, puis de récupérer les vecteurs pour les `target_ids` fournis.

### Fonctions Utilitaires

```python
import tensorflow as tf

def get_initializer(embedding_dim, vocab_size):
    """Calcule les bornes et génère l'initialiseur aléatoire."""
    # Division flottante pour obtenir une précision décimale
    initial_bounds = 0.5 / embedding_dim
    
    # Création de la distribution uniforme
    initializer = tf.random.uniform(
        (vocab_size, embedding_dim),
        minval=-initial_bounds,
        maxval=initial_bounds
    )
    return initializer