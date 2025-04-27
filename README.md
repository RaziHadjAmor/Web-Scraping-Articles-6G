# Web-Scraping-Articles-6G
__Projet en Management des Systèmes d'Information (MSI) en BUT2 Informatique parcours Technologie de l'information__

## Description du projet
Ce projet est une application qui récupère des articles scientifiques depuis l’API d’arXiv en fonction d’un mot-clé donné (en l'occurence sur la 6G ici). 
Les articles extraits sont ensuite stockés dans un fichier Excel (XLSX) et affichés sous forme d’un tableau interactif dans une interface Streamlit.

## Comment exécuter le code 

Il faudra au préalable ajouter une clé API de Groq sur le fichier extraction.py (à la ligne 7 entre les guillemets)
Récupérez votre clé API Groq en suivant le lien https://console.groq.com/keys

Apres avoir télécharger les fichiers, exécuter app.py sur votre terminal avec la commande suivante :
```
streamlit run app.py
```
ou si ça ne fonctionne pas, exécutez la commande suivante :
```
python -m streamlit run app.py
```
