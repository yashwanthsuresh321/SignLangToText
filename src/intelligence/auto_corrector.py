"""
SignLanguageAI
Smart Auto-Correction Engine

Version 1.0.0
Offline word-level correction for recognized sign-language text.
"""

import re
from difflib import SequenceMatcher


class AutoCorrector:
    """
    Offline word-level auto-correction engine.

    The recognizer produces individual characters.
    This class waits until a word is completed and then
    checks whether the word is likely to be a recognition
    error.
    """

    def __init__(self):

        # ------------------------------------------------------
        # Correction Settings
        # ------------------------------------------------------

        self.minimum_word_length = 3

        # Minimum similarity required before correcting.
        #
        # Example:
        # WATR -> WATER
        # has a high similarity score.
        #
        # Completely unrelated words will not be changed.
        self.minimum_similarity = 0.78

        # ------------------------------------------------------
        # Common SignLanguageAI Vocabulary
        # ------------------------------------------------------
        #
        # This is intentionally kept local.
        # No internet connection or API is required.
        #
        # The vocabulary can be expanded later.
        #

        vocabulary = """
        a
        about
        above
        after
        again
        all
        am
        an
        and
        are
        around
        as
        at
        away
        back
        bad
        be
        because
        become
        before
        help
        below
        between
        big
        can
        come
        could
        day
        did
        do
        does
        doing
        done
        dont
        down
        eat
        enough
        every
        feel
        find
        food
        for
        from
        get
        give
        go
        good
        goodbye
        had
        has
        have
        he
        hello
        her
        here
        him
        his
        home
        how
        i
        if
        in
        is
        it
        just
        know
        like
        little
        live
        look
        make
        me
        more
        morning
        my
        need
        new
        no
        not
        now
        of
        off
        oh
        okay
        on
        one
        only
        or
        other
        our
        out
        over
        please
        put
        read
        right
        see
        she
        should
        show
        small
        so
        some
        something
        sorry
        speak
        stop
        take
        tell
        thank
        thanks
        that
        the
        their
        them
        then
        there
        these
        they
        thing
        think
        this
        time
        to
        today
        together
        too
        try
        understand
        up
        us
        use
        very
        wait
        want
        water
        way
        we
        well
        what
        when
        where
        which
        who
        why
        will
        with
        work
        would
        yes
        you
        your
        """

        self.vocabulary = set(
            word.strip().lower()
            for word in vocabulary.split()
            if word.strip()
        )

        # ------------------------------------------------------
        # Common Recognition Corrections
        # ------------------------------------------------------
        #
        # These handle frequent one-character recognition
        # mistakes before general similarity matching.
        #

        self.common_corrections = {

            "helo": "hello",
            "hellp": "hello",
            "heloo": "hello",

            "wat": "what",
            "wht": "what",

            "watr": "water",
            "wate": "water",

            "hel": "help",
            "hepl": "help",

            "plese": "please",
            "pleas": "please",

            "thak": "thank",
            "thnks": "thanks",
            "tahnk": "thank",

            "frmo": "from",
            "form": "from",

            "hte": "the",
            "teh": "the",

            "adn": "and",
            "nad": "and",

            "yu": "you",
            "yuo": "you",

            "ur": "your",
            "yor": "your",

            "nead": "need",
            "ned": "need",

            "wnat": "want",
            "waht": "what",

            "goob": "good",
            "godo": "good",

            "hom": "home",

            "foood": "food",

            "pleas": "please",

            "sory": "sorry",
            "sorr": "sorry",

            "stpo": "stop",

            "waitt": "wait",

            "comee": "come",

            "helpp": "help",

        }

    # ==========================================================
    # Similarity
    # ==========================================================

    def _similarity(self, word_a, word_b):

        return SequenceMatcher(
            None,
            word_a,
            word_b
        ).ratio()

    # ==========================================================
    # Find Best Candidate
    # ==========================================================

    def _find_best_candidate(self, word):

        word = word.lower()

        best_word = word
        best_score = 0.0

        for candidate in self.vocabulary:

            # Avoid expensive comparisons against obviously
            # unrelated words.
            if abs(len(candidate) - len(word)) > 2:
                continue

            score = self._similarity(
                word,
                candidate
            )

            if score > best_score:

                best_score = score
                best_word = candidate

        if best_score >= self.minimum_similarity:

            return best_word, best_score

        return word, best_score

    # ==========================================================
    # Correct Word
    # ==========================================================

    def correct_word(self, word):

        if not word:
            return word, False

        original = word
        normalized = word.lower()

        # ------------------------------------------------------
        # Do not modify very short words.
        # ------------------------------------------------------

        if len(normalized) < self.minimum_word_length:

            return original, False

        # ------------------------------------------------------
        # Already valid.
        # ------------------------------------------------------

        if normalized in self.vocabulary:

            return original, False

        # ------------------------------------------------------
        # Known recognition mistake.
        # ------------------------------------------------------

        if normalized in self.common_corrections:

            corrected = self.common_corrections[
                normalized
            ]

            if original.isupper():

                corrected = corrected.upper()

            elif original.istitle():

                corrected = corrected.capitalize()

            return corrected, True

        # ------------------------------------------------------
        # Similarity-based correction.
        # ------------------------------------------------------

        corrected, score = self._find_best_candidate(
            normalized
        )

        if corrected == normalized:

            return original, False

        if original.isupper():

            corrected = corrected.upper()

        elif original.istitle():

            corrected = corrected.capitalize()

        return corrected, True

    # ==========================================================
    # Correct Completed Sentence
    # ==========================================================

    def correct_sentence(self, sentence):

        if not sentence:

            return sentence, False, None

        # Preserve trailing spaces.
        trailing_space = sentence.endswith(" ")

        words = sentence.strip().split()

        if not words:

            return sentence, False, None

        corrected_words = []

        changed = False
        correction_info = None

        for word in words:

            corrected, was_changed = (
                self.correct_word(word)
            )

            corrected_words.append(corrected)

            if was_changed and correction_info is None:

                correction_info = (
                    word,
                    corrected
                )

            if was_changed:

                changed = True

        corrected_sentence = " ".join(
            corrected_words
        )

        if trailing_space:

            corrected_sentence += " "

        return (
            corrected_sentence,
            changed,
            correction_info
        )

    # ==========================================================
    # Correct Last Completed Word
    # ==========================================================

    def correct_last_word(self, sentence):

        if not sentence:

            return sentence, False, None

        trailing_space = sentence.endswith(" ")

        words = sentence.strip().split()

        if not words:

            return sentence, False, None

        last_word = words[-1]

        corrected, changed = self.correct_word(
            last_word
        )

        if not changed:

            return sentence, False, None

        words[-1] = corrected

        corrected_sentence = " ".join(words)

        if trailing_space:

            corrected_sentence += " "

        return (
            corrected_sentence,
            True,
            (last_word, corrected)
        )