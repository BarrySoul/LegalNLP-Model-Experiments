# LegalNLP-Model-Experiments

Expériences de traitement automatique du langage naturel appliquées aux textes juridiques et aux documents législatifs.

## Contenu

- `ex1/` : classification de textes juridiques avec LEDGAR, notamment le fine-tuning LoRA et le linear probing.
- `ex2/` : génération de résumés avec BillSum, notamment le fine-tuning complet, LoRA et une approche de type BERTSum.
- `ex3/` : génération et explication de prédictions sur LEDGAR avec des approches zero-shot, few-shot et LoRA.
- `docs/rapport.pdf` : rapport du projet.

## Installation

Python 3.10 ou plus récent est recommandé. Créez un environnement virtuel, puis installez PyTorch pour votre système depuis [pytorch.org](https://pytorch.org/get-started/locally/) et les autres dépendances :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

Sous Windows, activez l'environnement avec `.venv\Scripts\activate`.

## Exécution

Depuis la racine du projet, lancez un script avec Python, par exemple :

```bash
python ex1/distilbert_linear_probing.py
```

Chaque expérience peut télécharger les jeux de données et modèles Hugging Face nécessaires à sa première exécution. Les besoins en mémoire et en calcul dépendent du modèle choisi ; un GPU est recommandé pour plusieurs expériences.
