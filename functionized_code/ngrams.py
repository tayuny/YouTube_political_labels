import re

def clean_punctuation(text: str) -> str:
    '''
    Remove punctuation characters {., !, ?, -, ', "} from a string.
    text (str): string to clean.
    Return cleaned text (str).
    '''
    regex = r'[.!?\-\'"]'
    return re.sub(regex, '', text)


def make_ngrams(text: str, n: int) -> dict:
    '''
    Parse text into substrings of length n and return the frequency of each.
    text (str): string to parse into substrings.
    n (int): length of substring.
    Return mapping of substrings to frequencies (dict).
    '''
    # Initialize a mapping of n-grams to frequencies.
    grams = {}
    # Consider each character in the string.
    for i in range(len(text) - n + 1):
        # Get the n-gram at this index.
        gram = text[i:i + n]
        # Add this n-gram to the dictionary if it does not yet exist.
        if not grams.get(gram):
            grams[gram] = 0
        # Increment the count of this n-gram by one.
        grams[gram] += 1
    # Return the mapping of n-grams to frequencies.
    return grams


def dict_to_list(d: dict) -> list:
    '''
    Convert a dictionary into a list of key, value pairs.
    d (dict): mapping of key, value pairs to convert.
    Return dictionary as a list (list of tuples).
    '''
    return [(key, value) for key, value in d.items()]