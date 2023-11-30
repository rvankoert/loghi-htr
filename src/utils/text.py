# Imports

# > Standard library
import re


# Text processing functions
def remove_tags(text):
    return re.sub(r'[␃␅␄␆]', '', text)


def preprocess_text(text):
    text = text.strip().replace('', '')
    text = remove_tags(text)
    return text


def simplify_text(text):
    lower_text = text.lower()
    simple_text = re.sub(r'[^a-zA-Z0-9]', '', lower_text)
    return lower_text, simple_text
