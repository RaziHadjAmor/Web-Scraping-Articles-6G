import os
import pytest
from unittest.mock import patch, MagicMock
from extraction import grep_articles, parse_articles, format_date, getAnnee, send_request_to_groq, process_articles_with_groq

# Teste la fonction grep_articles en simulant une requête HTTP à arXiv
@patch("extraction.libreq.urlopen")
def test_grep_articles_success(mock_urlopen):
    """Vérifie que grep_articles récupère correctement les articles depuis l'API arXiv."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"""<?xml version='1.0' encoding='utf-8'?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <id>1234.5678</id>
            <title>Test Article</title>
            <author><name>John Doe</name></author>
            <published>2023-05-10T12:34:56Z</published>
            <summary>Abstract of test article</summary>
            <link title="pdf" href="http://example.com/pdf"/>
        </entry>
    </feed>"""
    mock_urlopen.return_value.__enter__.return_value = mock_response

    # grep_articles écrit un fichier articles.xml dans le répertoire courant.
    articles = grep_articles("test", 1)
    # Nettoyage du fichier créé
    if os.path.exists("articles.xml"):
        os.remove("articles.xml")

    assert len(articles) == 1
    assert articles[0]["Titre"] == "Test Article"
    assert articles[0]["Auteurs"] == ["John Doe"]
    assert articles[0]["pdf"] == "http://example.com/pdf"

# Teste la fonction parse_articles avec un XML vide
def test_parse_articles_empty(tmp_path, monkeypatch):
    """Vérifie que parse_articles renvoie une liste vide lorsqu'aucun article n'est présent dans le XML."""
    empty_xml = "<feed xmlns='http://www.w3.org/2005/Atom'></feed>"
    # Création d'un fichier temporaire articles.xml contenant le XML vide
    file_path = tmp_path / "articles.xml"
    file_path.write_text(empty_xml, encoding='utf-8')
    # Change le répertoire courant pour que parse_articles() ouvre le fichier temporaire
    monkeypatch.chdir(tmp_path)
    articles = parse_articles()  # La fonction ne prend aucun argument
    assert len(articles) == 0

# Teste le formatage correct d'une date ISO
def test_format_date_valid():
    """Vérifie que format_date convertit une date ISO en format lisible."""
    assert format_date("2023-05-10T12:34:56Z") == "10 May 2023"

# Teste le comportement de format_date avec une date invalide
def test_format_date_invalid():
    """Vérifie que format_date lève une ValueError pour une date invalide."""
    with pytest.raises(ValueError):
        format_date("invalid-date")

# Teste l'extraction de l'année d'une date ISO avec getAnnee
def test_getAnnee_valid():
    """Vérifie que getAnnee extrait correctement l'année d'une date ISO."""
    assert getAnnee("2023-05-10T12:34:56Z") == "2023"

# Teste l'envoi d'une requête à l'API Groq et le traitement de la réponse
@patch("extraction.requests.post")
def test_send_request_to_groq_success(mock_post):
    """Vérifie que send_request_to_groq extrait correctement les informations d'un article via l'API Groq."""
    # Création d'une réponse factice sans utiliser MagicMock pour les attributs critiques
    class DummyResponse:
        def __init__(self):
            self.status_code = 200
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": "Keywords: AI, Research\nSummary: Short summary\nProblem: Define problem\nSolution: Provide solution\nTopic: Topic details"
                    }
                }]
            }
    mock_post.return_value = DummyResponse()
    article = {"Abstract": "This is a test abstract."}
    result = send_request_to_groq(article)
    assert "Keywords:" in result

# Teste le traitement des articles avec les données retournées par l'API Groq
@patch("extraction.send_request_to_groq", return_value="Keywords: AI, Research\nSummary: Short summary\nProblem: Define problem\nSolution: Provide solution\nTopic: Topic details")
def test_process_articles_with_groq(mock_send_request):
    """Vérifie que process_articles_with_groq remplit correctement les champs des articles après traitement avec l'API Groq."""
    articles = [{"Abstract": "Test abstract"}]
    processed_articles = process_articles_with_groq(articles)
    assert processed_articles[0]["Keywords"] == "AI, Research"
    assert processed_articles[0]["Summary"] == "Short summary"
    assert processed_articles[0]["Problem"] == "Define problem"
    assert processed_articles[0]["Solution"] == "Provide solution"
    assert processed_articles[0]["Topic"] == "Topic details"
