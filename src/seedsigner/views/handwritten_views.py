"""
Loads a seed from a photo of its handwritten words: a SeedSigner 12-word or 24-word
seed card, or a free list of words.

Flow: photo (HandwrittenScanView), reading (HandwrittenReadView), result
(HandwrittenStatusView: verified by the fingerprint written with the words, or else a
trust level), review of every word, with each uncertain word confirmed (HandwrittenReviewView,
HandwrittenWordView, HandwrittenWordEntryView), the fingerprint to write next to the words
when the seed was not verified (HandwrittenFingerprintView), then the standard SeedFinalizeView.

The photo is only kept in memory (as JPEG data), and only until the reading is done.
"""
import threading

from gettext import gettext as _

from seedsigner.gui.components import FontAwesomeIconConstants, GUIConstants, SeedSignerIconConstants
from seedsigner.gui.screens import (RET_CODE__BACK_BUTTON, ButtonListScreen, DireWarningScreen,
    LargeIconStatusScreen, WarningScreen)
from seedsigner.gui.screens.handwritten_screens import (HandwrittenNoticeScreen, HandwrittenReadingScreen,
    HandwrittenScanScreen, HandwrittenWordScreen)
from seedsigner.gui.screens.screen import ButtonOption, ButtonOptionWithoutTranslation
from seedsigner.gui.screens.seed_screens import SeedMnemonicEntryScreen
from seedsigner.hardware.camera import Camera
from seedsigner.helpers.l10n import mark_for_translation as _mft
from seedsigner.models.seed import InvalidSeedException, Seed
from seedsigner.views.view import BackStackView, Destination, MainMenuView, View



class HandwrittenReview:
    """
    The reading of the photo and the words confirmed by the user so far, shared by the
    review Views through their `view_args`. The default `repr()` keeps the words out of
    the Controller's back_stack logs.
    """
    def __init__(self, reading):
        self.reading = reading
        self.confirmed: dict[int, str] = {}
        self._analyses = {}


    def analyze(self, excluded_index: int = None):
        """
        Checksum analysis of the reading with the confirmed words fixed, except the
        word at `excluded_index`. Each analysis is computed once (up to a few seconds
        on a Pi Zero for 24 uncertain words).
        """
        confirmed = {index: word for index, word in self.confirmed.items() if index != excluded_index}
        key = frozenset(confirmed.items())
        if key not in self._analyses:
            self._analyses[key] = self.reading.analyze(confirmed)
        return self._analyses[key]



def load_engine():
    """
    Loads the word models (about a second on a Pi Zero). Errors are left to the
    reading, which loads them again.
    """
    try:
        from seedsigner.helpers.handwriting import load_engine
        load_engine()
    except Exception:
        pass



class HandwrittenScanView(View):
    def run(self):
        # The word models are loaded while the photo is framed: the preview is limited by
        # the camera, not by the processor
        threading.Thread(target=load_engine, daemon=True).start()

        photo = self.run_screen(HandwrittenScanScreen)

        if photo == RET_CODE__BACK_BUTTON:
            return Destination(BackStackView)

        return Destination(HandwrittenReadView, view_args={"photo": photo})



class HandwrittenReadView(View):
    """
    Reads the words in a background thread while the screen displays the progress.
    Removed from the back_stack once done, which releases the photo.

    `photo`: JPEG data from the camera, decoded in the reading thread.
    """
    def __init__(self, photo: bytes):
        super().__init__()
        self.photo = photo


    def run(self):
        # The word models are loaded by HandwrittenScanView, or else by read_seed in the
        # reading thread, while the reading screen is displayed.
        from seedsigner.helpers.handwriting import WordsNotFound, read_seed

        def read_function(screen):
            photo = Camera.get_instance().decode_frame(self.photo, expand=True, grayscale=True)
            screen.show_photo(photo)
            return read_seed(photo, progress=screen.report_progress, on_word=screen.show_word)

        try:
            reading = self.run_screen(HandwrittenReadingScreen, read_function=read_function)
        except WordsNotFound as e:
            return Destination(HandwrittenReadFailedView, view_args={"at_edge": e.at_edge, "line_count": e.line_count, "card": e.card}, skip_current_view=True)

        return Destination(HandwrittenStatusView, view_args={"review": HandwrittenReview(reading)}, skip_current_view=True)



class HandwrittenReadFailedView(View):
    """
    `at_edge`: the words reach the edge of the photo, so some are probably cut off.
    `line_count`: number of words found, when it is not a valid seed length.
    `card`: the words are those of a seed card, part of which is out of the photo.
    """
    RETAKE = ButtonOption("Retake photo", FontAwesomeIconConstants.CAMERA)
    CANCEL = ButtonOption("Cancel")

    def __init__(self, at_edge: bool = False, line_count: int = 0, card: bool = False):
        super().__init__()
        self.at_edge = at_edge
        self.line_count = line_count
        self.card = card


    def run(self):
        if self.at_edge and self.card:
            status_headline = _("Card cut off")
            text = _("Move the camera back so the whole card fits in the photo.")
        elif self.at_edge:
            status_headline = _("Words cut off")
            text = _("Move the camera back so all the words fit in the photo.")
        elif self.line_count:
            status_headline = _("{} words found").format(self.line_count)
            text = _("A seed has 12, 15, 18, 21 or 24 words. Check that none is missing.")
        else:
            status_headline = _("Words not found")
            text = _("Fit all the words in the photo, with good light and no blur.")

        button_data = [self.RETAKE, self.CANCEL]
        selected_menu_num = self.run_screen(
            WarningScreen,
            title=_("Scan Failed"),
            status_headline=status_headline,
            text=text,
            show_back_button=False,
            button_data=button_data,
        )

        if button_data[selected_menu_num] == self.RETAKE:
            return Destination(BackStackView)

        elif button_data[selected_menu_num] == self.CANCEL:
            return Destination(MainMenuView)



class HandwrittenStatusView(View):
    """
    A seed whose fingerprint, written with the words, matches them is announced as
    verified. Any other reading is never called valid: a full screen notice shows its
    trust level and explains what to do, and must be scrolled through before going on
    to the review.
    """
    REVIEW = ButtonOption("Review words")
    UNDERSTAND = ButtonOption("I understand")

    # Label of each trust level below "verified" (see handwriting.TRUST_RATES)
    TRUST_LABELS = {
        1: _mft("Confidence: low"),
        2: _mft("Confidence: medium"),
        3: _mft("Confidence: high"),
    }

    def __init__(self, review: HandwrittenReview):
        super().__init__()
        self.review = review


    def run(self):
        analysis = self.review.analyze()
        trust = self.review.reading.trust()
        # TRANSLATOR_NOTE: Inserts the number of words read (e.g. "12 words read")
        title = _("{} words read").format(len(analysis.mnemonic))

        fingerprint = getattr(self.review.reading, "fingerprint", None)
        if trust.verified:
            self.run_screen(
                LargeIconStatusScreen,
                title=title,
                status_headline=_("Verified"),
                # TRANSLATOR_NOTE: Inserts the fingerprint written with the words (e.g. "73c5da0a")
                text=_("The fingerprint {} written with your words matches them.").format(fingerprint.matched),
                show_back_button=False,
                button_data=[self.REVIEW],
            )
            return Destination(HandwrittenReviewView, view_args={"review": self.review}, skip_current_view=True)

        if analysis.status == "invalid":
            status_color = GUIConstants.DIRE_WARNING_COLOR
            what = _("Some words were misread: this is not a valid seed.")
        else:
            status_color = GUIConstants.WARNING_COLOR
            if fingerprint is not None and fingerprint.status == "mismatch":
                # TRANSLATOR_NOTE: Inserts the fingerprint as read (e.g. "73c5da0a")
                what = _("The fingerprint read, {}, does not match these words: a word or a digit may be "
                         "misread, or it is that of a passphrase.").format(fingerprint.text)
            elif self.review.reading.ambiguous_order:
                what = _("Your words are in two columns that can be read in two orders. "
                         "Check the order of every word.")
            else:
                what = _("No fingerprint was found with your words to verify them.")

        paragraphs = [
            (None, what),
            (None, _("Compare every word with your paper.")),
        ]
        if analysis.uncertain:
            if len(analysis.uncertain) == len(analysis.mnemonic):
                marks = _("Every word is marked like this: open each one, then confirm or correct it.")
            else:
                marks = _("The least sure words are marked like this: open each one, then confirm or correct it.")
            paragraphs.append((SeedSignerIconConstants.WARNING, marks))
        paragraphs.append((None, _("Unmarked words can be wrong too. Never use this seed without checking it.")))

        self.run_screen(
            HandwrittenNoticeScreen,
            title=title,
            status_headline=_("Not verified"),
            status_color=status_color,
            gauge=(trust.level, _(self.TRUST_LABELS[trust.level])),
            paragraphs=paragraphs,
            button_data=[self.UNDERSTAND],
        )

        return Destination(HandwrittenReviewView, view_args={"review": self.review}, skip_current_view=True)



class HandwrittenReviewView(View):
    """
    Every word read, with a warning icon on the uncertain words and a check icon on the
    words confirmed by the user. Selecting a word opens its HandwrittenWordView.
    """
    DONE = ButtonOption("Done")

    def __init__(self, review: HandwrittenReview):
        super().__init__()
        self.review = review


    def run(self):
        analysis = self.review.analyze()

        button_data = []
        for index, word in enumerate(analysis.mnemonic):
            if index in self.review.confirmed:
                icon_name, icon_color = SeedSignerIconConstants.CHECK, GUIConstants.SUCCESS_COLOR
            elif index in analysis.uncertain:
                icon_name, icon_color = SeedSignerIconConstants.WARNING, GUIConstants.WARNING_COLOR
            else:
                icon_name, icon_color = None, None
            button_data.append(ButtonOptionWithoutTranslation(f"{index + 1}. {word}", icon_name=icon_name, icon_color=icon_color))

        # "Done" is only offered once every uncertain word has been confirmed
        unchecked = [index for index in sorted(analysis.uncertain) if index not in self.review.confirmed]
        if not unchecked:
            button_data.append(self.DONE)

        selected_menu_num = self.run_screen(
            ButtonListScreen,
            title=_("Review words"),
            is_button_text_centered=False,
            button_data=button_data,
            selected_button=unchecked[0] if unchecked else 0,
        )

        if selected_menu_num == RET_CODE__BACK_BUTTON:
            return Destination(BackStackView)

        if button_data[selected_menu_num] == self.DONE:
            try:
                seed = Seed(analysis.mnemonic)
            except InvalidSeedException:
                return Destination(HandwrittenInvalidView)

            self.controller.storage.set_pending_seed(seed)
            if not self.review.reading.trust(self.review.confirmed).verified:
                return Destination(HandwrittenFingerprintView, view_args={"review": self.review})
            from seedsigner.views.seed_views import SeedFinalizeView
            return Destination(SeedFinalizeView)

        return Destination(HandwrittenWordView, view_args={"review": self.review, "word_index": selected_menu_num})



class HandwrittenWordView(View):
    """
    The photographed word with its most likely readings; the selected word is confirmed
    and the checksum analysis is run again with it.
    """
    TYPE_WORD = ButtonOption("Type the word", FontAwesomeIconConstants.KEYBOARD)

    def __init__(self, review: HandwrittenReview, word_index: int):
        super().__init__()
        self.review = review
        self.word_index = word_index


    def run(self):
        analysis = self.review.analyze(excluded_index=self.word_index)

        button_data = [
            ButtonOptionWithoutTranslation(f"{word}  {probability:.0%}", return_data=word)
            for word, probability in analysis.options(self.word_index)
        ]
        button_data.append(self.TYPE_WORD)

        selected_menu_num = self.run_screen(
            HandwrittenWordScreen,
            # TRANSLATOR_NOTE: Inserts the word number (e.g. "Seed Word #6")
            title=_("Seed Word #{}").format(self.word_index + 1),  # Human-readable 1-indexing!
            button_data=button_data,
            word_image=self.review.reading.words[self.word_index].image,
        )

        if selected_menu_num == RET_CODE__BACK_BUTTON:
            return Destination(BackStackView)

        if button_data[selected_menu_num] == self.TYPE_WORD:
            return Destination(
                HandwrittenWordEntryView,
                view_args={"review": self.review, "word_index": self.word_index},
                skip_current_view=True
            )

        self.review.confirmed[self.word_index] = button_data[selected_menu_num].return_data
        return Destination(BackStackView)



class HandwrittenWordEntryView(View):
    def __init__(self, review: HandwrittenReview, word_index: int):
        super().__init__()
        self.review = review
        self.word_index = word_index


    def run(self):
        current_word = self.review.analyze().mnemonic[self.word_index]

        word = self.run_screen(
            SeedMnemonicEntryScreen,
            # TRANSLATOR_NOTE: Inserts the word number (e.g. "Seed Word #6")
            title=_("Seed Word #{}").format(self.word_index + 1),  # Human-readable 1-indexing!
            initial_letters=list(current_word),
            wordlist=Seed.get_wordlist(),
        )

        if word == RET_CODE__BACK_BUTTON:
            return Destination(
                HandwrittenWordView,
                view_args={"review": self.review, "word_index": self.word_index},
                skip_current_view=True
            )

        self.review.confirmed[self.word_index] = word
        return Destination(BackStackView)



class HandwrittenFingerprintView(View):
    """
    Shown when the seed loaded was not verified by a fingerprint written with its words:
    asks to write its fingerprint, without passphrase, next to the words, so that the
    next reading can be verified.
    """
    NEXT = ButtonOption("Next")

    def __init__(self, review: HandwrittenReview):
        super().__init__()
        self.review = review


    def run(self):
        # Fingerprint of the words alone, whatever passphrase is added afterwards
        pending_seed = self.controller.storage.get_pending_seed()
        seed_fingerprint = Seed(pending_seed.mnemonic_list, wordlist_language_code=pending_seed.wordlist_language_code).get_fingerprint()
        fingerprint = getattr(self.review.reading, "fingerprint", None)
        if fingerprint is not None and fingerprint.status == "mismatch":
            # TRANSLATOR_NOTE: Inserts the fingerprint as read on the paper (e.g. "73c5da0a")
            text = _("Yours, {}, did not match: it may include a passphrase. Write this one, without "
                     "passphrase.").format(fingerprint.text)
        else:
            text = _("Write this fingerprint next to your words, without passphrase: next time, it will verify them.")

        selected_menu_num = self.run_screen(
            LargeIconStatusScreen,
            title=_("Fingerprint"),
            status_icon_name=SeedSignerIconConstants.FINGERPRINT,
            status_color=GUIConstants.ACCENT_COLOR,
            status_headline=seed_fingerprint,
            text=text,
            show_back_button=True,
            button_data=[self.NEXT],
        )

        if selected_menu_num == RET_CODE__BACK_BUTTON:
            return Destination(BackStackView)

        from seedsigner.views.seed_views import SeedFinalizeView
        return Destination(SeedFinalizeView)



class HandwrittenInvalidView(View):
    EDIT = ButtonOption("Review & edit")
    DISCARD = ButtonOption("Discard", button_label_color="red")

    def run(self):
        button_data = [self.EDIT, self.DISCARD]
        selected_menu_num = self.run_screen(
            DireWarningScreen,
            title=_("Invalid Mnemonic!"),
            status_icon_name=SeedSignerIconConstants.ERROR,
            status_headline=None,
            text=_("Checksum failure; not a valid seed phrase."),
            show_back_button=False,
            button_data=button_data,
        )

        if button_data[selected_menu_num] == self.EDIT:
            return Destination(BackStackView)

        elif button_data[selected_menu_num] == self.DISCARD:
            return Destination(MainMenuView)
