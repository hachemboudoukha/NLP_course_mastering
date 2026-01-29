# Résumé du Chapitre : Vocabulaire et Tokenisation en NLP

## 1. Concepts Fondamentaux du Vocabulaire

* **Corpus de texte** : Il s'agit de l'ensemble des documents ou textes utilisés pour une tâche spécifique (par exemple, un ensemble d'articles de presse).
* **Vocabulaire** : C'est l'ensemble des mots uniques présents dans le corpus. Bien que le vocabulaire puisse être basé sur les caractères, l'approche par **mots** est la plus courante.
* **Tokenisation** : Processus consistant à diviser un texte en unités individuelles appelées **tokens** (généralement des mots). Cela permet de transformer une chaîne de caractères brute en une liste exploitable par un algorithme.

---

## 2. Utilisation de l'objet Tokenizer (TensorFlow/Keras)

L'outil principal utilisé est la classe `tf.keras.preprocessing.text.Tokenizer`. Elle automatise plusieurs étapes cruciales :

* **Indexation** : Chaque mot du vocabulaire est associé à un identifiant entier unique, attribué selon la fréquence d'apparition (les mots les plus fréquents ont les index les plus bas).
* **fit_on_texts** : Cette méthode analyse le corpus pour créer le dictionnaire interne (le vocabulaire).
* **texts_to_sequences** : Cette méthode transforme une liste de textes en listes de nombres (vecteurs), remplaçant chaque mot par son nombre correspondant.

---

## 3. Paramètres et Gestion des Exceptions

Le `Tokenizer` offre des options pour affiner le traitement des données :

* **Filtrage** : Par défaut, la ponctuation et les majuscules sont supprimées pour normaliser le texte.
* **OOV (Out-Of-Vocabulary)** : Lorsqu'un nouveau texte contient un mot absent du vocabulaire initial, il est normalement ignoré. Le paramètre `oov_token` permet de remplacer ces mots inconnus par un jeton spécial (ex: "OOV") afin de conserver la structure de la phrase.
* **num_words** : Ce paramètre limite la taille du vocabulaire aux $N$ mots les plus fréquents. C'est essentiel pour réduire la complexité du modèle et éviter le surapprentissage sur des mots rares.

---

# Implémentation : La fonction `tokenize_text_corpus`



### Code Python (TensorFlow)

```python
import tensorflow as tf

class EmbeddingModel(object):
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Initialisation du Tokenizer avec la limite de mots définie
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    def tokenize_text_corpus(self, texts):
        # 1. Analyse les textes pour construire l'index des mots (le dictionnaire)
        self.tokenizer.fit_on_texts(texts)
        
        # 2. Convertit les mots en séquences d'entiers basées sur cet index
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        return sequences