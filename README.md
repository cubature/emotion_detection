# Émotion Détection (Sentiment Analyse)

C'est un projet pour entraîner un modèle de réseau neurone pour l'émotion détection en niveau de texte et la création d'un corpus pour l'entraînement.

## Géneration d'un corpus

La géneration d'un corpus contient la collection des données brutes et les traitements des données. 

### Theory

#### Collection des données brutes

Dans notre cas, les données à collecter sont des phrases. Et comme notre objectif est de créer un chatbot, les phrases courtes nous intéressent plus. Pour entraîner un modèle à détecter l'émotion dans une phrase, nous avons besoin d'un label émotion pour chaque phrase. Et pour garder la qualité de l'entraînement, un corpus qui a plus de 100,000 phrases est mieux. Donc il va être comme : 

> Une phrase court. label émotion  
> ...  
> ...  
> ...\*100,000+

Créer un corpus à partir de 0 est très difficile. Normalement, nous utilisons les corpus existant sur internet. Nous pouvons les collecter par *web crawler* ou d'autres façons. Mais nous n'avons pas trouvé un grand corpus français avec émotion. Donc nous voulons créer un corpus basé sur des corpus sur internet. Il y a quelques propositions : Traduire un grand corpus anglais avec émotion, combiner des petit corpus français avec émotion, transformer un corpus français vocal en un corpus texte, et ajouter des labels émotions dans un grand corpus français, etc. La proposition plus executable est la première. 

La traduction un corpus anglais en un corpus français a besoin beaucoup de travaux. Pour commencer tôt l'entraînement du réseau neurone, nous pouvons utiliser le corpus anglais directement. Il n'y a pas beaucoup de différences entre entraîner un modèle pour détecter l'émotion des phrases anglais et un modèle pour français, qui est correspondant à la structure du réseau neurone.

#### Traitements des données

Même si nous avons collecté un grand corpus, nous ne pouvons pas l'utiliser directement sans le traiter. Des bons traitements peuvent améliorer beaucoup le résultat de l'entraînement. Normalement, les traitements contiennent :

1. Supprimer les données inutiles (e.g. des noms d'utilisateurs, les ponctuations).
2. Rendre tous les lettres en minuscules, sinon après le word embedding, un mot peut être traité comme deux mots, e.g. *NOM* et *nom*.
3. La lemmatisation. Transformer tous les mots en mode original, e.g. transformer un verb en infinitif. Sinon *est* et *soit* vont être traité comme deux mots différents.
4. Groupe de mots. Parfois des mots groupés ont un sens qui est différent que ceux quand ils sont séparés. 
5. ...

### Utilisation des codes

Aujourd'hui, nous avons fait seulemnt le premier et le deuxième traitement. Et le corpus original que nous avons utilisé est `/data/corpus_tlkh/text_emotion_original.txt`.

Le fichier `/data/corpus_tlkh/generate_corpus.py` est le script pour faire les traitements des données. Les traitements sont réalisés à l'aide de regex, par un librairie de Python, *re*. Il supprime les mots inutiles, supprime les ponctuations, groupe les 13 émotions en 5 classes `happy, neutral, sad, hate, anger` et sépare le corpus original en 3 fichiers : 

* train.txt : 70% de corpus original pour entraîner le modèle de réseau neurone
* eval.txt : 15% de corpus original pour évaluer le modèle de réseau neurone
* test.txt : 15% de corpus original pour tester le modèle de réseau neurone

Ces 3 fichiers sont stocké dans le dossier `/data/corpus_tlkh/processed/`. Les lignes dans ces 3 fichiers sont dans le format comme : 

`we did what we had to do;sad`

Une phrase est à la gauche de ";", par contre un label d'émotion est à la droite. 

## Entraînement du réseau neurone

### Word embedding

Après les traitements de données, nous obtenons un corpus concise. Mais une machine ne peut pas encore comprendre des phrases dans le corpus. Avant l'entraînement, une démarche est très important, c'est le **word embedding**. 

Word embedding transforme les mots en vecteurs, puis la machine peut les utiliser. 

`/data/corpus_tlkh/word_embedding.py` est le script qui fait le word embedding. D'ailleurs, il transforme les 5 émotions en chiffres (mais pas en vecteurs), et les stocke dans `/data/corpus_tlkh/embedded/`. 

### Structure du réseau neurone

`/rnn_model.py` définie la structure du réseau neurone (surtout de la ligne 42 à 116). Comme nous avons utiliser un réseau de Bi-LSTM, le corpus doivent être entrées par ordre et par ordre à l'inverse. 

Maintenant, la structure du réseau est simple, il a une couche de neurones. Dans cette couche, les neurones peuvent être séparées en deux groupes, l'un est pour les entrées par ordre, l'autre est pour les entrées par ordre à l'inverse. 

### Entraînement

`/data_helper.py` aide charger le corpus à entraîner. Les phrases ont toujours des longeurs différens, mais ce sont difficile à traiter les entrées qui ont des longeurs différents. Donc dans `/data_helper.py`, nous décidons un longeur maximum de tous les phrases. Et puis tous les pharses sont rallongé au longeur maximum (Maintenant, tous les phrases sont des listes de vecteurs grâce à word embeding, donc *rallonger* signifie d'ajouter des vecteurs de 0s). Après le *rallonger*, une entrée de plusieur phrases devient une matrice, qui est plus facile à traiter.

Mais quand nous voulons lancer un entraînement, nous n'avons pas besoin de lancer `/data_helper.py`, mais lancer `/emotion_detecter.py`. Dans `/emotion_detecter.py`, il charge le corpus par `/data_helper.py`, crée un réseau neurone basé sur `/rnn_model.py` et utilise le corpus pour entraîner le réseau neurone. Les options d'entraînement sont définies dans `/emotion_detecter.py`.

Après l'entraînement, nous pouvons lancer `/use_emotion_detecter.py` pour utiliser le modèle entraîné stocké dans `/runs/checkpoints` et le word embedding modèle dans `/data/corpus_tlkh/embed_model/` pour une émotion détection. 

## Conclusion

Les démarches globales sont : 

1. Collecter un corpus et le stocker dans `/data/corpus_tlkh/`.
2. Traiter le corpus par `/data/corpus_tlkh/generate_corpus.py`.
	* obtenir trois fichiers, `train.txt, eval.txt, test.txt` (corpus traité), stockés dans `/data/corpus_tlkh/processed/`.
3. Faire le word embedding par `/data/corpus_tlkh/`.
	* obtenir `train.pkl, eval.pkl, test.pkl` (corpus en vecteurs), stockés dans `/data/corpus_tlkh/embedded/`.
	* obtenir `word2vec_model` (modèle de word embedding, dictionnaire de mot - vecteur), stocké dans `/data/corpus_tlkh/embed_model/`.
	* obtenir `emotion_embeddings.pkl` (dictionnaire de emotion - numéro de classe), stocké dans `/data/corpus_tlkh/embed_model/`.
4. (optionnel) Lancer `/data_helper.py` pour verifier le corpus est bien traité. 
5. Lancer `/emotion_detecter.py` pour commencer un entraînement.
	* obtenir des modèles pendant l'entraînement, stockés dans `/runs/checkpoints/`.
	* obtenir des sommaires (pas important pour l'instant), stockés dans `runs/summaries/`.
6. Lancer `/use_emotion_detecter.py` pour une émotion détection. La phrase à détecter est écrit dans `use_emotion_detecter.py`, ligne 15. Le résultat va être affiché sur l'écran. 