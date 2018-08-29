# Émotion Détection (Sentiment Analyse)

C'est un projet pour entraîner un modèle de réseau neurone pour l'émotion détection en niveau de texte et la création d'un corpus pour l'entraînement.

## Géneration d'un corpus

La géneration d'un corpus contient la collection des données brutes et le traitement des données. 

### Collection des données brutes

Dans notre cas, les données à collecter sont des phrases. Et comme notre objectif est de créer un chatbot, les phrases courtes nous intéressent plus. Pour entraîner un modèle à détecter l'émotion dans une phrase, nous avons besoin d'un label émotion pour chaque phrase. Et pour garder la qualité de l'entraînement, un corpus qui a plus de 100,000 phrases est mieux. Donc il va être comme : 

> Une phrase court. label émotion
> ...
> ...
> ...\*100,000+

Créer un corpus à partir de 0 est très difficile. Normalement, nous utilisons les corpus existant sur internet. Nous pouvons les collecter par *web crawler* ou d'autres façons. Mais nous n'avons pas trouvé un grand corpus français avec émotion. Donc nous voulons créer un corpus basé sur des corpus sur internet. Il y a quelques propositions : Traduire un grand corpus anglais avec émotion, combiner des petit corpus français avec émotion, transformer un corpus français vocal en un corpus texte, et ajouter des labels émotions dans un grand corpus français, etc. La proposition plus executable est la première. 

La traduction un corpus anglais en un corpus français a besoin beaucoup de travaux. Pour commencer tôt l'entraînement du réseau neurone, nous pouvons utiliser le corpus anglais directement. Il n'y a pas beaucoup de différences entre entraîner un modèle pour détecter l'émotion des phrases anglais et un modèle pour français, qui est correspondant à la structure du réseau neurone.

### Traitement des données

Même si nous avons collecté un grand corpus, nous ne pouvons pas l'utiliser directement sans le traiter. Un bon traitement peut améliorer beaucoup le résultat de l'entraînement. 



## Entraînement du réseau neurone