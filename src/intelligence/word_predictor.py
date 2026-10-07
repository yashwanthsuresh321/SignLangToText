"""
Word Predictor
--------------
Provides word completion suggestions based on
the currently typed/inferred word.
"""

class WordPredictor:

    def __init__(self):

        # Common English vocabulary.
        # We can expand this later.
        self.words = [
            "ABOUT",
            "AFTER",
            "AGAIN",
            "ALL",
            "ALWAYS",
            "AM",
            "AND",
            "ANSWER",
            "ARE",
            "AROUND",
            "BE",
            "BECAUSE",
            "BEFORE",
            "BEST",
            "BUT",
            "CAN",
            "COME",
            "COULD",
            "DAY",
            "DO",
            "DOING",
            "DONE",
            "EACH",
            "EVEN",
            "EVERY",
            "FIND",
            "FIRST",
            "FOR",
            "FROM",
            "GET",
            "GIVE",
            "GO",
            "GOOD",
            "GREAT",
            "HAVE",
            "HELLO",
            "HELP",
            "HERE",
            "HOW",
            "I",
            "IF",
            "IN",
            "IS",
            "IT",
            "KNOW",
            "LEARN",
            "LIKE",
            "LOOK",
            "MAKE",
            "ME",
            "MORE",
            "MY",
            "NEED",
            "NEVER",
            "NEW",
            "NEXT",
            "NO",
            "NOT",
            "NOW",
            "OF",
            "ON",
            "ONE",
            "ONLY",
            "OR",
            "OTHER",
            "OUR",
            "OUT",
            "PLEASE",
            "PROBLEM",
            "READY",
            "RIGHT",
            "SAY",
            "SEE",
            "SHE",
            "SHOULD",
            "SIGN",
            "SO",
            "SOME",
            "START",
            "STOP",
            "TAKE",
            "TELL",
            "THANK",
            "THANKS",
            "THAT",
            "THE",
            "THEIR",
            "THEM",
            "THEN",
            "THERE",
            "THESE",
            "THEY",
            "THING",
            "THIS",
            "TIME",
            "TO",
            "TOGETHER",
            "TRY",
            "UNDERSTAND",
            "UP",
            "USE",
            "VERY",
            "WANT",
            "WAS",
            "WAY",
            "WE",
            "WELL",
            "WHAT",
            "WHEN",
            "WHERE",
            "WHICH",
            "WHO",
            "WHY",
            "WILL",
            "WITH",
            "WORD",
            "WORK",
            "WORLD",
            "WOULD",
            "YES",
            "YOU",
            "YOUR",
        ]

        self.words = sorted(
            set(word.upper() for word in self.words)
        )

    # ==========================================================
    # Predict
    # ==========================================================

    def predict(self, current_word, limit=3):

        if not current_word:
            return []

        current_word = current_word.strip().upper()

        if not current_word:
            return []

        matches = [
            word
            for word in self.words
            if word.startswith(current_word)
            and word != current_word
        ]

        return matches[:limit]