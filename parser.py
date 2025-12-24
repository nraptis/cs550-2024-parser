import nltk
import sys
import re
from typing import List
from collections import Counter

# nltk.download('punkt_tab')

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""



#==>['we', 'arrived', 'the', 'day', 'before', 'thursday']
#-->[['N'], ['V'], ['Det'], ['N'], ['P'], ['N']]

#==>['holmes', 'sat', 'in', 'the', 'red', 'armchair', 'and', 'he', 'chuckled']
#-->[['N'], ['V'], ['P'], ['Det'], ['Adj'], ['N'], ['Conj'], ['N'], ['V']]

#==>['my', 'companion', 'smiled', 'an', 'enigmatical', 'smile']
#-->[['Det'], ['N'], ['V'], ['Det'], ['Adj'], ['N']]

#==>['holmes', 'chuckled', 'to', 'himself']
#-->[['N'], ['V'], ['P'], ['N']]

#==>['she', 'never', 'said', 'a', 'word', 'until', 'we', 'were', 'at', 'the', 'door', 'here']
#-->[['N'], ['Adv'], ['V'], ['Det'], ['N'], ['Conj'], ['N'], ['V'], ['P'], ['Det'], ['N'], ['Adv']]

#==>['holmes', 'sat', 'down', 'and', 'lit', 'his', 'pipe']
#-->[['N'], ['V'], ['Adv'], ['Conj'], ['V'], ['Det'], ['N']]

#==>['i', 'had', 'a', 'country', 'walk', 'on', 'thursday', 'and', 'came', 'home', 'in', 'a', 'dreadful', 'mess']
#-->[['N'], ['V'], ['Det'], ['Adj'], ['N'], ['P'], ['N'], ['Conj'], ['V'], ['N'], ['P'], ['Det'], ['Adj'], ['N']]

#==>['i', 'had', 'a', 'little', 'moist', 'red', 'paint', 'in', 'the', 'palm', 'of', 'my', 'hand']
#-->[['N'], ['V'], ['Det'], ['Adj'], ['Adj'], ['Adj'], ['N'], ['P'], ['Det'], ['N'], ['P'], ['Det'], ['N']]

NONTERMINALS = """
S -> NP VP|NP VP Conj NP VP|NP VP Conj VP
NP -> N|Det NP|P NP|NP P NP|Det AP NP
VP -> V|Adv VP|VP Adv|VP NP Adv|VP NP
AP -> Adj|AP Adj
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def tag_words(words, token_to_tag, unknown="UNK"):
    tags = []
    for w in words:
        w2 = re.sub(r'[^a-zA-Z]', '', w).lower()
        if not w2:
            continue
        tags.append(token_to_tag.get(w2, unknown))
    return tags

def ngrams(seq, n):
    for i in range(len(seq) - n + 1):
        yield tuple(seq[i:i+n])

def debug_chart(parser, tokens):
    chart = parser.chart_parse(tokens)

    # Find furthest end position reached by any edge
    furthest = 0
    for e in chart.edges():
        furthest = max(furthest, e.end())

    print(f"Furthest token index reached: {furthest} / {len(tokens)}")
    if furthest < len(tokens):
        print("Stuck around token:", furthest, "=>", repr(tokens[furthest]))

    # Show completed edges that end at the furthest position
    print("\nCompleted edges ending at furthest position:")
    for e in chart.edges():
        if e.is_complete() and e.end() == furthest:
            print(f"  {e.start()}..{e.end()}  {e.lhs()} -> {' '.join(map(str, e.rhs()))}")

    # Show what categories can start at furthest position (predicted/incomplete)
    print("\nEdges that start at furthest position (what it was expecting next):")
    for e in chart.edges():
        if e.start() == furthest and not e.is_complete():
            print(f"  {e.start()}..{e.end()}  {e.lhs()} -> {' '.join(map(str, e.rhs()))}  • {e.nextsym()}")


def main():

    loot = {}
    t_lines = TERMINALS.split("\n")
    for line in t_lines:
        s_hakves = line.split(">")
        if len(s_hakves) == 2:
            kind = s_hakves[0]
            kind = re.sub(r'[^a-zA-Z]', '', kind)
            words = s_hakves[1].split("|")
            for word in words:
                word = re.sub(r'[^a-z]', '', word)
                if kind not in loot:
                    loot[kind] = []
                loot[kind].append(word)

    for key in loot.keys():
        print(f"dict[{key}] => {loot[key]}")

    # Build reverse map: token -> tag
    token_to_tag = {}
    for tag, words in loot.items():
        for w in words:
            w = re.sub(r'[^a-zA-Z]', '', w).lower()
            if not w:
                continue
            if w in token_to_tag and token_to_tag[w] != tag:
                print("WARNING: token maps to multiple tags:", w, token_to_tag[w], tag)
            token_to_tag[w] = tag
        
    
    ngram_counts = Counter()
    for number in range(1, 11):
        path = f"sentences/{number}.txt"
        with open(path) as file:
            sentence = file.read()
        words = preprocess(sentence)
        tags = tag_words(words, token_to_tag)

        for n in range(2, 6):  # 2..5 length chunks
            ngram_counts.update(ngrams(tags, n))

    for pat, c in ngram_counts.most_common(40):
        print(c, " ".join(pat))
        


    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    print(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)

        debug_chart(parser, s)

        return
    if not trees:
        print("Could not parse sentence.")
        
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence: str):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    tokens = nltk.word_tokenize(sentence.lower(), language="english")
    result = []
    for token in tokens:
        token = re.sub(r'[^a-z]', '', token)
        if token:
            result.append(token)
    return result

def is_np_chunk(tree: nltk.Tree) -> bool:
    if tree.label() != "NP":
        return False
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            return False
    return True

def np_chunk(tree) -> List[nltk.Tree]:
    return list(tree.subtrees(filter=is_np_chunk))

if __name__ == "__main__":
    main()
