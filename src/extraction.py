import urllib.request as libreq
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
import json

GROQ_API_KEY = ""  # clé API Groq (remplacer par votre propre clé API Groq)

def grep_articles(keyword=None, max_articles=None):
    """
    Effectue une recherche d'articles sur arxviv suivant le mot-clé et le nombre max d'articles passés en paramètre.
    :param keyword: Le mot clés de la recherche, pour utiliser plusieurs mot-clés séparés les par un espace
    :param max_articles: Le nombre d'articles que vous souhaitez, par défaut 10
    :return: une liste d'articles sous forme de dictionnaires contenant les informations des articles.
    """
    if keyword is None:
        print("Mot-clé manquant, veuillez ajouter un objet de recherche")
        return None
     # Récupération des données depuis l'API arxiv
    if max_articles is None:
        url = f"http://export.arxiv.org/api/query?search_query=all:{keyword}"
    else:
        url = f"http://export.arxiv.org/api/query?search_query=all:{keyword}&max_results={max_articles}" 
     # si on veut les articles les plus récents dans l'ordre décroissant on peut ajouter &sortBy=submittedDate&sortOrder=descending 
     # sinon l'API récupère par défaut les articles qu'il considère comme les plus pertinants

    with libreq.urlopen(url) as response:
        # Forcer l'encodage en UTF-8 lors de la lecture
        r = response.read().decode('utf-8')

    # Enregistrer le XML avec encodage UTF-8
    with open("articles.xml", 'w', encoding='utf-8') as f:
        f.write(r)

    parsed = parse_articles()
    return process_articles_with_groq(parsed)

def parse_articles():
    """
    Parcours le fichier xml contenant les articles recherchés pour en extraire une liste.
    :return: Une liste d'articles sous forme de dictionnaires contenant les informations des articles
    """
    articles = []
    n = 0
    parser = ET.XMLParser(encoding="utf-8")
    tree = ET.parse('articles.xml', parser=parser)
    grosse_racine = tree.getroot()
    # Création de la liste d'articles
    for article in grosse_racine.findall('{http://www.w3.org/2005/Atom}entry'):
        articles.append({
            "id": article.find('{http://www.w3.org/2005/Atom}id').text,
            "Titre": article.find('{http://www.w3.org/2005/Atom}title').text,
            "Auteurs": [],
            "published": article.find('{http://www.w3.org/2005/Atom}published').text,
            "Année": article.find('{http://www.w3.org/2005/Atom}published').text,
            "Keywords": "",
            "Abstract": article.find('{http://www.w3.org/2005/Atom}summary').text,
            "Summary": "",
            "Problem": "",
            "Solution": "",
            "Topic": "",
            "pdf": "",
            "doi": ""
        })
        # Ajoute les différents auteurs de l'article au dictionnaire associé
        for auteur in article.findall('{http://www.w3.org/2005/Atom}author'):
            articles[n]["Auteurs"].append(auteur.find('{http://www.w3.org/2005/Atom}name').text)
       # Ajoute le lien du pdf de l'article
        for link in article.findall('{http://www.w3.org/2005/Atom}link'):
            if link.get('title') == "pdf":
                articles[n]["pdf"] = link.get('href')
            if link.get('title') == "doi":
                articles[n]["doi"] = link.get('href')
        # Formate la date de l'article
        articles[n]["published"] = format_date(articles[n]["published"])
        
        # Récupère l'année
        articles[n]["Année"] = getAnnee(articles[n]["Année"])
        n += 1
    return articles

def format_date(date):
    """
    Formate la date passée en paramètre. Ex : 13 Sep 2020
    :param date: date à formater
    :return: la nouvelle date formatée
    """
    new_date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
    s = new_date.strftime("%d")
    s += " " + new_date.strftime("%b")
    s += " " + new_date.strftime("%Y")
    return s


def getAnnee(date):
    """
    Récupère seulement l'année de la date passée en paramètre. Ex : 2020
    :param date: date à modifier
    :return: l'année de la date
    """
    new_date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
    y = new_date.strftime("%Y")
    return y


def send_request_to_groq(article):
    """
    Sends a request to the Groq API for a single article.
    Returns the raw response text.
    """
    url = 'https://api.groq.com/openai/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GROQ_API_KEY}'
    }

    system_message = {
        "role": "system",
        "content": (
            "You are an AI assistant that extracts structured information from a research article. "
            "For the given article, produce EXACTLY five lines in EXACTLY this order:\n\n"
            "Keywords: <list of keywords separated by commas>\n"
            "Summary: <concise summary>\n"
            "Problem: <short sentence describing the problem>\n"
            "Solution: <short sentence describing the solution>\n"
            "Topic: This topic covers <short paragraph>\n\n"
            "Do not include any additional text."
        )
    }

    user_message = {
        "role": "user",
        "content": (
            "Please extract the following information from the text below:\n"
            "1) Keywords\n2) Summary\n3) Problem\n4) Solution\n5) Topic\n\n"
            f"Text:\n{article['Abstract']}"
        )
    }

    data = {
        "model": "llama-3.3-70b-versatile",  # Ajustez le modèle si nécessaire
        "messages": [system_message, user_message]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print(f"Groq API error: {response.status_code}")
        return ""
    
    try:
        response_data = response.json()
        raw_content = response_data['choices'][0]['message']['content'].strip()
        # Affichage pour le débogage
        print("=== RAW CONTENT for one article ===")
        print(raw_content)
        print("===================================")
        return raw_content
    except Exception as e:
        print(f"Groq API error: {str(e)}")
        return ""

def process_articles_with_groq(articles):
    """
    Processes each article individually with the Groq API and adds
    'Keywords', 'Summary', 'Problem', 'Solution', and 'Topic' fields to each article.
    """
    print("Processing articles with Groq API...")
    for i, article in enumerate(articles):
        raw_content = send_request_to_groq(article)
        if not raw_content:
            print(f"No response for article {i+1}.")
            continue
        
        sections = raw_content.split("\n")
        if len(sections) < 5:
            print(f"Unexpected format for article {i+1}: got only {len(sections)} lines.")
            continue
        
        article["Keywords"] = sections[0].replace("Keywords:", "").strip()
        article["Summary"]  = sections[1].replace("Summary:", "").strip()
        article["Problem"]  = sections[2].replace("Problem:", "").strip()
        article["Solution"] = sections[3].replace("Solution:", "").strip()
        article["Topic"]    = sections[4].replace("Topic:", "").strip()
    
    return articles
