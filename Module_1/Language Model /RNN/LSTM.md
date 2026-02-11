#  Réseaux de Neurones aux LSTM


## 2. Réseaux de Neurones Récurrents (RNN)
Contrairement aux réseaux classiques, les RNN possèdent une **mémoire interne**. Ils sont indispensables pour les données séquentielles.





### Avantages et Inconvénients
| Points Forts | Points Faibles |
| :--- | :--- |
| Modélise les dépendances entre échantillons. | Problème de disparition du gradient (*Vanishing Gradient*). |
| Utilisable avec des couches convolutionnelles. | Très difficile à entraîner sur de longues séquences. |

---

## 3. Long Short-Term Memory (LSTM)
L'LSTM est une version évoluée du RNN qui permet de conserver des informations sur de très longues périodes en réglant le problème du gradient.

### Le système des "Portes" (Gates)
La cellule LSTM régule le flux d'information via trois mécanismes :

1. **Forget Gate (Oubli) :** Utilise une fonction *Sigmoid* pour décider quelles informations de la mémoire passée ($C_{t-1}$) doivent être supprimées (0 = oublier, 1 = garder).
2. **Input Gate (Entrée) :** Détermine quelles nouvelles données sont importantes. La *Sigmoid* filtre et la *tanh* donne un poids d'importance aux valeurs.
3. **Output Gate (Sortie) :** Décide quelle partie de la mémoire interne est envoyée en sortie vers l'étape suivante.


