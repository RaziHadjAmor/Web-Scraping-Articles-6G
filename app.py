from src.extraction import *
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt


articles = grep_articles("6g", 15)
df = pd.DataFrame(articles)
df.to_excel("articles.xlsx")

# Charger les données à partir du fichier Excel
df = pd.read_excel("articles.xlsx")  # On charge le fichier xlsx qui contient les informations sur les articles

# Filtres Sidebar 
st.sidebar.header("Options de filtres : ")
with st.expander("Faire défiler pour voir les données") :
    Titre_filtre = st.sidebar.multiselect("Filtrer les articles que vous souhaitez :", options=df['Titre'].unique(), default=df['Titre'].unique())
Annee_filtre = st.sidebar.multiselect("Filtrer les dates :", options=df['Année'].unique(), default=df['Année'].unique())

# Application des filtres sur le dataframe ---> df devient donnees_filtres
donnees_filtres = df[(df["Titre"].isin(Titre_filtre)) & (df["Année"].isin(Annee_filtre))]

# Affichage du titre de la page avec Streamlit
st.title("Tableau de bord : Articles sur la 6G")  # Crée un titre principal pour l'application

# Affichage des données filtrées
st.subheader("DataFrame obtenu après l'extraction des articles")
st.info("Double-cliquer sur une case pour voir tout son contenu", icon="ℹ️")

with st.expander("Faire défiler pour voir les données") : # Permet de défiler le tableau
    st.dataframe(donnees_filtres)   # Affiche le DataFrame sous forme de tableau interactif dans l'application

# Bouton de téléchargement
donnees_filtres.to_excel("donnees.xlsx", index=False, sheet_name="Feuille1")
st.download_button(label="Télécharger le fichier .xlsx", data=open("donnees.xlsx", "rb"), file_name="donnees.xlsx", icon="⬇️")

st.subheader("Exemple de diagramme et d'une courbe")
st.write("Nombre d'articles produits par année")

# Création d'un graphique pour visualiser la fréquence des dates de publication
st.bar_chart(donnees_filtres['Année'].value_counts())  # Crée un graphique à barres qui montre la fréquence de publication des articles
st.line_chart(donnees_filtres['Année'].value_counts()) # Courbe montrant l'évolution du nb d'articles produits séléctionnés par l'API en fonction de l'année de publication

# Création d'un nuage de mots des mots-clé les plus fréquents dans les articles 
st.subheader("Nuage de mots des mots-clé les plus fréquents")
st.info("La taille du mot varie en fonction de sa fréquence", icon="ℹ️")

# Vérification si les données sont vides
if donnees_filtres.empty:
    st.error("Aucune donnée sélectionnée. Sélectionnez les données dans la barre latérale.", icon="🚨")
    st.stop()

# Concaténation des titres et résumés
text = " ".join(donnees_filtres['Titre'].astype(str) + " " + donnees_filtres['Abstract'].astype(str))
# Génération du nuage de mots
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
# Affichage du nuage de mots
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Equipe 3 :  Joel - Ilyes - Razi - Vincent")