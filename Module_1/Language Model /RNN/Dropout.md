# Régularisation et Dropout dans les RNN

## A. Pourquoi régulariser ?
Lorsqu'un RNN possède de nombreux paramètres (grand nombre d'unités cachées ou plusieurs couches), il risque le **surapprentissage (overfitting)**. Il apprend alors par cœur les données d'entraînement mais échoue à généraliser sur de nouvelles données. La régularisation permet de limiter ce risque.

## B. Le fonctionnement du Dropout
Le **Dropout** est une technique de régularisation consistant à "désactiver" aléatoirement des neurones pendant l'entraînement.

* **En Feed-forward :** On met à zéro certains neurones cachés pour forcer le réseau à ne pas dépendre d'un seul groupe de neurones.
* **Dans les RNN :** Le dropout est appliqué aux **entrées** ($x_i$) et/ou aux **sorties** de chaque cellule à chaque étape temporelle. 



### Paramètres clés (TensorFlow/DropoutWrapper)
Pour implémenter cela, on utilise souvent un `DropoutWrapper`. Les deux paramètres principaux sont :

| Argument | Description | Valeur par défaut |
| :--- | :--- | :--- |
| **input_keep_prob** | Probabilité de **conserver** l'entrée à chaque étape. | 1.0 (pas de dropout) |
| **output_keep_prob** | Probabilité de **conserver** la sortie à chaque étape. | 1.0 (pas de dropout) |

> **Note pratique :** Une valeur de **0.5** est souvent un bon point de départ pour l'entraînement. Cela signifie que la moitié des connexions sont aléatoirement ignorées, forçant le modèle à devenir plus robuste.