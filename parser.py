import nltk
import sys
import re

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

NONTERMINALS = """
S -> NP VP|NP VP Conj NP VP|NP VP Conj VP
NP -> N|Det NP|P NP|NP P NP|Det AP NP
VP -> V|Adv VP|VP Adv|VP NP Adv|VP NP
AP -> Adj|AP Adj
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)

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

def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    result = []
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            count_np = 0
            for subsubtree in subtree.subtrees():
                if subsubtree.label() == "NP":
                    count_np += 1
            if count_np == 1:
                result.append(subtree)
    return result

if __name__ == "__main__":
    main()
