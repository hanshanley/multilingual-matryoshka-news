'''
Utility function file.

'''
import langdetect
from bs4 import BeautifulSoup
import re

# Function to clean HTML and JavaScript from text
def clean_html_js(text):
    # Remove script tags and their content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    # Remove inline JavaScript
    text = re.sub(r'function\s*\w*\(.*?\)\s*{[^}]*}', '', text, flags=re.DOTALL)
    text = re.sub(r'var\s*\w*\s*=\s*[^;]*;', '', text, flags=re.DOTALL)
    text = re.sub(r'setInterval\([^)]*\)', '', text, flags=re.DOTALL)
    # Remove include virtual
    text = re.sub(r'include\s*virtual="[^"]*"', '', text, flags=re.DOTALL)
    # Remove conditional comments
    text = re.sub(r'\[if\s*!IE\][^\[]*\[endif\]', '', text, flags=re.DOTALL)
    # Remove remaining unwanted characters
    cleaned_text = re.sub(r'\\n', ' ', text)
    cleaned_text = re.sub(r'\\t', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Function to process and normalize the text
def process(article):
    normalized_string = unicodedata.normalize('NFKD', article)
    normalized_string = normalized_string.replace('’','\'').replace('‘','\'').replace('“','\"').replace('”','\"')
    # Remove URLs
    text = re.sub(r'https?://\S+', ' ', normalized_string)
    cleaned_text = re.sub('(http)(.+?)(?:\s)', ' ', text)
    cleaned_text = re.sub('<.*?>', '', cleaned_text) 
    cleaned_text = cleaned_text.replace('\t',' ').replace('\\t',' ').replace('\n','').replace('\\n','').replace('\r','').replace('\\r','')
    return clean_html_js(remove_extraneous_javascript(cleaned_text.strip()))

# Function to remove extraneous JavaScript from text
def remove_extraneous_javascript(text):
    patterns = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Match <script>...</script> blocks
        r'window\..*?;',  # Match window.<something>;
        r'addEventListener\([^\)]*\)',  # Match addEventListener(...)
        r'function\s*\(.*?\)\s*{.*?}',  # Match function definitions
        r'\bif\b\s*\(.*?\)\s*{.*?}',  # Match if conditions
        r'\bvar\b\s+[^\n;]+;',  # Match var declarations
        r'\bconst\b\s+[^\n;]+;',  # Match const declarations
        r'\blet\b\s+[^\n;]+;',  # Match let declarations
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    # Remove remaining unwanted characters
    text = re.sub(r'\n\s*\n', '\n', text)
    cleaned_text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    cleaned_text = re.sub(r'<style[^>]*>.*?</style>', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'{.*?}', '', cleaned_text, flags=re.DOTALL)
    return text.strip()
