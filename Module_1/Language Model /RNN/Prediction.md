# Prédiction avec le modèle LSTM

## A. Calcul des Probabilités (Softmax)
Une fois que le modèle a généré des **logits** (valeurs brutes) via la couche finale, il faut les transformer en probabilités exploitables.

* **Fonction Softmax :** Appliquée sur la **dernière dimension** des logits (celle du vocabulaire).
* **Interprétation des dimensions :**
    1. **Dimension 1 :** Séquences dans le batch.
    2. **Dimension 2 :** Étapes temporelles (chaque mot de la phrase).
    3. **Dimension 3 :** Probabilité pour chaque mot du vocabulaire.



---

## B. Prédictions de Mots (Argmax)
Pour obtenir le mot final prédit par le modèle à chaque étape, on sélectionne celui qui a la probabilité la plus élevée.

* **Méthode :** On utilise la fonction `tf.argmax` sur la dimension du vocabulaire.
* **Résultat :** L'indice retourné correspond directement à l'identifiant (entier) du mot dans notre dictionnaire de tokenisation.

> **Logique de prédiction :** À l'étape $t$, le modèle prédit le mot le plus probable pour l'étape $t+1$ en se basant sur tout le contexte précédent.

---

## C. État de l'art : Au-delà de l'LSTM
Bien que les LSTM soient très performants pour traiter le langage naturel par rapport aux réseaux classiques, ils ne représentent plus le sommet de la technologie actuelle.

* **Limites de l'LSTM :** Traitement séquentiel lent et difficulté à capturer des relations très complexes dans d'immenses volumes de données.
* **Les Transformers :** C'est l'architecture dominante aujourd'hui (utilisée par des modèles comme GPT). Contrairement aux LSTM, ils traitent tous les mots d'une phrase en même temps grâce au mécanisme d'**Attention**.



---

### Synthèse du flux de données
1. **Entrée :** Séquence d'identifiants de mots.
2. **Traitement :** Passage dans les couches LSTM (éventuellement empilées).
3. **Sortie brute :** Logits via une couche Dense.
4. **Probabilités :** Softmax.
5. **Prédiction finale :** Argmax (le mot le plus probable).