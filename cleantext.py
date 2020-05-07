#!/usr/bin/env python3

"""Clean comment text for easier parsing."""

from __future__ import print_function

import re
import string
import argparse

__author__ = "Arjun Kallapur"
__email__ = "arjun.kallapur@gmail.com"

# Depending on your implementation,
# this data may or may not be useful.
# Many students last year found it redundant.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}


# You may need to write regular expressions.

def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings 
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """

    # YOUR CODE GOES BELOW:

    # Remove newlines and tabs
    text = re.sub(r"[\n\t]", " ", text)
    # Remove tabs encoded as multiple spaces
    text = re.sub(r"\s+", " ", text)
    # For handling links, markup subreddits
    text = re.sub(r"[\[\]]", "", text)
    text = re.sub(r"\(http[^\)]*\)", "", text)
    text = re.sub(r"http[^\s]*\s", " ", text)
    text = text.lower()
    i = 0
    tokens = []
    punctuation = ",.!?:;"
    temp = ""
    for c in text:
        if not c.isalpha() and c not in punctuation and c != ' ':
            continue
        else:
            temp += c
    text = temp
    temp = ""
    while i < len(text):
        if text[i] in punctuation:
            if len(temp) > 0:
                tokens.append(temp)
                temp = ""
            tokens.append(str(text[i]))
        elif text[i] == ' ':
            if len(temp) > 0:
                tokens.append(temp)
                temp = ""
        else:
            temp = temp + text[i]
        i += 1
    if len(temp) > 0:
        tokens.append(temp)

    finalStr = ""
    for st in tokens:
        finalStr = finalStr + st + ' '
    finalStr = finalStr[:-1]

    unilist = []
    for st in tokens:
        if st[0] not in punctuation:
            unilist.append(st)

    dublist = []
    i = 0
    while i + 1 < len(tokens):
        contains_punc = False
        sliced = tokens[i:i + 2]
        for c in punctuation:
            if str(c) in sliced:
                contains_punc = True
                break
        if not contains_punc:
            dublist.append(sliced[0] + '_' + sliced[1])
        i += 1
    i = 0
    triplist = []
    while i + 2 < len(tokens):
        contains_punc = False
        sliced = tokens[i:i + 3]
        for c in punctuation:
            if str(c) in sliced:
                contains_punc = True
                break
        if not contains_punc:
            triplist.append(sliced[0] + '_' + sliced[1] + '_' + sliced[2])
        i += 1
    uniStr = ""
    dubStr = ""
    triStr = ""
    for tok in unilist:
        uniStr = uniStr + tok + ' '
    uniStr = uniStr[:-1]
    for st in dublist:
        dubStr = dubStr + st + ' '
    dubStr = dubStr[:-1]
    for st in triplist:
        triStr = triStr + st + ' '
    triStr = triStr[:-1]
    unigrams = uniStr
    trigrams = triStr
    bigrams = dubStr
    parsed_text = finalStr
#print(parsed_text)
#print(unigrams)
#print(bigrams)
#print(trigrams)
    return [parsed_text, unigrams, bigrams, trigrams]


if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # YOUR CODE GOES BELOW.

    # We are "requiring" your write a main function so you can
    # debug your code. It will not be graded.
