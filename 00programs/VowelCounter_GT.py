import random

verbose = False
wordsFile = open('words3000.txt', 'r')
wordList = wordsFile.readlines()
wordsFile.close()

phrase = random.choice(wordList).strip() + " " + random.choice(wordList).strip() + " " + random.choice(wordList).strip() + " " + random.choice(wordList).strip() + " " + random.choice(wordList).strip()
inputTC = phrase
outputTC = phrase.count('a') + phrase.count('e') + phrase.count('i') + phrase.count('o') + phrase.count('u')
