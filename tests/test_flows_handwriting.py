import io
import sys
import time
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageColor, ImageDraw

# Must import test base before the Controller
from base import BaseTest, FlowTest, FlowStep, FlowTestInvalidButtonDataSelectionException

from seedsigner.helpers.handwriting import TRUST_HIGH, TRUST_LOW, TRUST_MEDIUM, TRUST_VERIFIED, Trust, WordsNotFound
from seedsigner.gui.screens.screen import RET_CODE__BACK_BUTTON
from seedsigner.hardware.camera import Camera
from seedsigner.models.settings_definition import SettingsConstants
from seedsigner.views import handwritten_views, seed_views
from seedsigner.views.view import MainMenuView, View


VALID_MNEMONIC = ["abandon"] * 11 + ["about"]
INVALID_MNEMONIC = ["abandon"] * 12



class StubAnalysis:
    def __init__(self, mnemonic: list[str], status: str, uncertain: dict[int, list[str]]):
        self.mnemonic = mnemonic
        self.status = status
        self.uncertain = uncertain


    def options(self, k: int, n: int = 3):
        alternatives = self.uncertain.get(k, [])
        return [(self.mnemonic[k], 0.6)] + [(word, 0.2) for word in alternatives][:n - 1]



class StubReading:
    """
    Stands in for the reading returned by `seedsigner.helpers.handwriting.read_seed`.
    Each uncertain word stays uncertain until it is confirmed.
    """
    def __init__(self, mnemonic: list[str], uncertain: dict[int, list[str]] = None, is_valid: bool = True):
        self.kind = "card12"
        self.words = [MagicMock(image=Image.new("L", (120, 30), 255)) for word in mnemonic]
        self.mnemonic = mnemonic
        self.uncertain = uncertain or {}
        self.is_valid = is_valid
        self.ambiguous_order = False
        self.verified = False
        self.fingerprint = None
        self.analyze_calls = []


    def analyze(self, confirmed: dict[int, str] = None):
        confirmed = confirmed or {}
        self.analyze_calls.append(dict(confirmed))
        return self._analysis(confirmed)


    def _analysis(self, confirmed: dict[int, str]):
        mnemonic = [confirmed.get(k, word) for k, word in enumerate(self.mnemonic)]
        uncertain = {k: words for k, words in self.uncertain.items() if k not in confirmed}
        if not self.is_valid:
            status = "invalid"
        elif uncertain:
            status = "uncertain"
        else:
            status = "valid"
        return StubAnalysis(mnemonic, status, uncertain)


    def trust(self, confirmed: dict[int, str] = None):
        confirmed = confirmed or {}
        if self.verified and all(self.mnemonic[k] == word for k, word in confirmed.items()):
            return Trust(TRUST_VERIFIED)
        status = self._analysis(confirmed).status
        return Trust({"valid": TRUST_HIGH, "uncertain": TRUST_MEDIUM, "invalid": TRUST_LOW}[status])



def make_photo() -> bytes:
    """ JPEG data, as returned by Camera.capture_jpeg() """
    stream = io.BytesIO()
    Image.new("RGB", (2592, 1944), "white").save(stream, format="jpeg")
    return stream.getvalue()



class TestHandwritingFlows(FlowTest):

    def setup_method(self):
        super().setup_method()
        self.settings.set_value(SettingsConstants.SETTING__HANDWRITTEN_SEEDS, SettingsConstants.OPTION__ENABLED)


    def scan_steps(self, reading) -> list[FlowStep]:
        return [
            FlowStep(MainMenuView, button_data_selection=MainMenuView.SEEDS),
            FlowStep(seed_views.SeedsMenuView, is_redirect=True),  # No seeds loaded: redirects to LoadSeedView
            FlowStep(seed_views.LoadSeedView, button_data_selection=seed_views.LoadSeedView.SCAN_HANDWRITTEN),
            FlowStep(handwritten_views.HandwrittenScanView, screen_return_value=make_photo()),
            FlowStep(handwritten_views.HandwrittenReadView, screen_return_value=reading),
        ]


    def test_option_disabled_by_default(self):
        """
        The handwritten seed option is only offered once enabled in the advanced settings.
        """
        from seedsigner.controller import Controller
        self.settings.set_value(SettingsConstants.SETTING__HANDWRITTEN_SEEDS, SettingsConstants.OPTION__DISABLED)

        with pytest.raises(FlowTestInvalidButtonDataSelectionException):
            self.run_sequence([
                FlowStep(seed_views.LoadSeedView, button_data_selection=seed_views.LoadSeedView.SCAN_HANDWRITTEN),
            ])

        with pytest.raises(FlowTestInvalidButtonDataSelectionException):
            self.run_sequence([
                FlowStep(seed_views.SeedSelectSeedView, button_data_selection=seed_views.SeedSelectSeedView.SCAN_HANDWRITTEN),
            ], initial_destination_view_args=dict(flow=Controller.FLOW__VERIFY_SINGLESIG_ADDR))


    def test_valid_reading_flow(self):
        """
        A reading with a valid checksum and no fingerprint is not called valid: the
        status notice shows its trust level, then the review, the fingerprint to write
        next to the words, and SeedFinalizeView.
        """
        reading = StubReading(VALID_MNEMONIC)
        screens = []

        def record_screen(view: handwritten_views.HandwrittenStatusView):
            run_screen = view.run_screen
            view.run_screen = lambda *args, **kwargs: screens.append((args, kwargs)) or run_screen(*args, **kwargs)

        self.run_sequence(self.scan_steps(reading) + [
            FlowStep(handwritten_views.HandwrittenStatusView, before_run=record_screen, button_data_selection=handwritten_views.HandwrittenStatusView.UNDERSTAND),
            FlowStep(handwritten_views.HandwrittenReviewView, button_data_selection=handwritten_views.HandwrittenReviewView.DONE),
            FlowStep(handwritten_views.HandwrittenFingerprintView, button_data_selection=handwritten_views.HandwrittenFingerprintView.NEXT),
            FlowStep(seed_views.SeedFinalizeView, button_data_selection=seed_views.SeedFinalizeView.FINALIZE),
            FlowStep(seed_views.SeedOptionsView),
        ])

        assert self.controller.storage.seeds[0].mnemonic_list == VALID_MNEMONIC
        kwargs = screens[0][1]
        assert kwargs["status_headline"] == "Not verified"
        assert kwargs["gauge"] == (TRUST_HIGH, "Confidence: high")
        assert all("valid" not in text.lower().replace("invalid", "") for _, text in kwargs["paragraphs"])


    def test_verified_reading_flow(self):
        """
        A seed verified by its fingerprint is announced as verified and loaded without
        asking to write its fingerprint.
        """
        reading = StubReading(VALID_MNEMONIC)
        reading.verified = True
        reading.fingerprint = MagicMock(status="verified", matched="73c5da0a")
        self.run_sequence(self.scan_steps(reading) + [
            FlowStep(handwritten_views.HandwrittenStatusView, button_data_selection=handwritten_views.HandwrittenStatusView.REVIEW),
            FlowStep(handwritten_views.HandwrittenReviewView, button_data_selection=handwritten_views.HandwrittenReviewView.DONE),
            FlowStep(seed_views.SeedFinalizeView),
        ])


    def test_verified_reading_word_changed(self):
        """
        A word of a verified seed changed by the user takes the verification away: the
        fingerprint of the seed loaded is then shown to be written.
        """
        reading = StubReading(VALID_MNEMONIC)
        reading.verified = True
        reading.fingerprint = MagicMock(status="verified", matched="73c5da0a")
        self.run_sequence(self.scan_steps(reading) + [
            FlowStep(handwritten_views.HandwrittenStatusView, button_data_selection=handwritten_views.HandwrittenStatusView.REVIEW),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=11),
            FlowStep(handwritten_views.HandwrittenWordView, button_data_selection=handwritten_views.HandwrittenWordView.TYPE_WORD),
            FlowStep(handwritten_views.HandwrittenWordEntryView, screen_return_value="actual"),
            FlowStep(handwritten_views.HandwrittenReviewView, button_data_selection=handwritten_views.HandwrittenReviewView.DONE),
            FlowStep(handwritten_views.HandwrittenFingerprintView, button_data_selection=handwritten_views.HandwrittenFingerprintView.NEXT),
            FlowStep(seed_views.SeedFinalizeView),
        ])

        assert self.controller.storage.get_pending_seed().mnemonic_list == VALID_MNEMONIC[:11] + ["actual"]


    def test_fingerprint_mismatch_flow(self):
        """
        A fingerprint that matches none of the seeds read: the notice and the fingerprint
        screen both quote it, the fingerprint screen shows that of the words alone, and
        neither gives a cause.
        """
        reading = StubReading(VALID_MNEMONIC, uncertain={k: ["ability"] for k in range(12)})
        reading.fingerprint = MagicMock(status="mismatch", text="0badf00d")
        screens = []

        def record_screen(view: View):
            run_screen = view.run_screen
            view.run_screen = lambda *args, **kwargs: screens.append(kwargs) or run_screen(*args, **kwargs)

        self.run_sequence(self.scan_steps(reading) + [
            FlowStep(handwritten_views.HandwrittenStatusView, before_run=record_screen, button_data_selection=handwritten_views.HandwrittenStatusView.UNDERSTAND),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=1),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=2),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=3),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=4),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=5),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=6),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=7),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=8),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=9),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=10),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=11),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),
            FlowStep(handwritten_views.HandwrittenReviewView, button_data_selection=handwritten_views.HandwrittenReviewView.DONE),
            FlowStep(handwritten_views.HandwrittenFingerprintView, before_run=record_screen, button_data_selection=handwritten_views.HandwrittenFingerprintView.NEXT),
            FlowStep(seed_views.SeedFinalizeView),
        ])

        assert "0badf00d" in screens[0]["paragraphs"][0][1] and "words alone" in screens[0]["paragraphs"][0][1]
        assert "Compare every word" in screens[0]["paragraphs"][1][1]
        assert screens[0]["gauge"][0] == TRUST_MEDIUM
        fingerprint = self.controller.storage.get_pending_seed().get_fingerprint()
        assert screens[1]["status_headline"] == fingerprint
        assert "0badf00d" in screens[1]["text"] and "words alone" in screens[1]["text"]
        assert all("passphrase" not in str(s) for s in screens)


    def test_seed_select_flow_entry(self):
        """
        The handwritten seed option is also offered when a flow asks for a seed.
        """
        from seedsigner.controller import Controller
        self.run_sequence([
            FlowStep(seed_views.SeedSelectSeedView, button_data_selection=seed_views.SeedSelectSeedView.SCAN_HANDWRITTEN),
            FlowStep(handwritten_views.HandwrittenScanView),
        ], initial_destination_view_args=dict(flow=Controller.FLOW__VERIFY_SINGLESIG_ADDR))


    def test_uncertain_word_option_flow(self):
        """
        Selecting a candidate word confirms it; the review then runs the checksum
        analysis again with the confirmed word fixed.
        """
        reading = StubReading(VALID_MNEMONIC, uncertain={2: ["ability"]})

        def check_review(view: handwritten_views.HandwrittenReviewView):
            assert view.review.confirmed == {2: "ability"}

        self.run_sequence(self.scan_steps(reading) + [
            FlowStep(handwritten_views.HandwrittenStatusView, button_data_selection=handwritten_views.HandwrittenStatusView.UNDERSTAND),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=2),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=1),  # "ability"
            FlowStep(handwritten_views.HandwrittenReviewView, before_run=check_review, screen_return_value=2),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),  # back to "abandon"
            FlowStep(handwritten_views.HandwrittenReviewView, button_data_selection=handwritten_views.HandwrittenReviewView.DONE),
            FlowStep(handwritten_views.HandwrittenFingerprintView, button_data_selection=handwritten_views.HandwrittenFingerprintView.NEXT),
            FlowStep(seed_views.SeedFinalizeView),
        ])

        # The word view analyzes without the word's own confirmation, the review with it;
        # each distinct analysis is computed once
        assert reading.analyze_calls == [{}, {2: "ability"}, {2: "abandon"}]
        assert self.controller.storage.get_pending_seed().mnemonic_list == VALID_MNEMONIC


    def test_ambiguous_order_flow(self):
        """
        A list whose two column orders both give a plausible seed is announced as such,
        every word being uncertain.
        """
        reading = StubReading(VALID_MNEMONIC, uncertain={k: ["ability"] for k in range(12)})
        reading.ambiguous_order = True
        screens = []

        def record_screen(view: handwritten_views.HandwrittenStatusView):
            run_screen = view.run_screen
            view.run_screen = lambda *args, **kwargs: screens.append(kwargs) or run_screen(*args, **kwargs)

        self.run_sequence(self.scan_steps(reading) + [
            FlowStep(handwritten_views.HandwrittenStatusView, before_run=record_screen, button_data_selection=handwritten_views.HandwrittenStatusView.UNDERSTAND),
            FlowStep(handwritten_views.HandwrittenReviewView),
        ])

        assert "two orders" in screens[0]["paragraphs"][0][1]


    def test_done_requires_uncertain_words_confirmed(self):
        """
        "Done" is not offered while an uncertain word has not been confirmed.
        """
        reading = StubReading(VALID_MNEMONIC, uncertain={2: ["ability"]})
        with pytest.raises(FlowTestInvalidButtonDataSelectionException):
            self.run_sequence(self.scan_steps(reading) + [
                FlowStep(handwritten_views.HandwrittenStatusView, button_data_selection=handwritten_views.HandwrittenStatusView.UNDERSTAND),
                FlowStep(handwritten_views.HandwrittenReviewView, button_data_selection=handwritten_views.HandwrittenReviewView.DONE),
            ])


    def test_type_word_flow(self):
        """
        "Type the word" opens the keyboard; backing out returns to the word view and the
        typed word is confirmed.
        """
        reading = StubReading(VALID_MNEMONIC, uncertain={11: ["above"]})

        def check_review(view: handwritten_views.HandwrittenReviewView):
            assert view.review.confirmed == {11: "about"}

        self.run_sequence(self.scan_steps(reading) + [
            FlowStep(handwritten_views.HandwrittenStatusView, button_data_selection=handwritten_views.HandwrittenStatusView.UNDERSTAND),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=11),
            FlowStep(handwritten_views.HandwrittenWordView, button_data_selection=handwritten_views.HandwrittenWordView.TYPE_WORD),
            FlowStep(handwritten_views.HandwrittenWordEntryView, screen_return_value=RET_CODE__BACK_BUTTON),
            FlowStep(handwritten_views.HandwrittenWordView, button_data_selection=handwritten_views.HandwrittenWordView.TYPE_WORD),
            FlowStep(handwritten_views.HandwrittenWordEntryView, screen_return_value="about"),
            FlowStep(handwritten_views.HandwrittenReviewView, before_run=check_review, button_data_selection=handwritten_views.HandwrittenReviewView.DONE),
            FlowStep(handwritten_views.HandwrittenFingerprintView, button_data_selection=handwritten_views.HandwrittenFingerprintView.NEXT),
            FlowStep(seed_views.SeedFinalizeView),
        ])


    def test_invalid_checksum_flow(self):
        """
        "Done" with an invalid checksum shows the invalid mnemonic screen, which returns
        to the review or discards the reading.
        """
        reading = StubReading(INVALID_MNEMONIC, is_valid=False)
        self.run_sequence(self.scan_steps(reading) + [
            FlowStep(handwritten_views.HandwrittenStatusView, button_data_selection=handwritten_views.HandwrittenStatusView.UNDERSTAND),
            FlowStep(handwritten_views.HandwrittenReviewView, button_data_selection=handwritten_views.HandwrittenReviewView.DONE),
            FlowStep(handwritten_views.HandwrittenInvalidView, button_data_selection=handwritten_views.HandwrittenInvalidView.EDIT),
            FlowStep(handwritten_views.HandwrittenReviewView, button_data_selection=handwritten_views.HandwrittenReviewView.DONE),
            FlowStep(handwritten_views.HandwrittenInvalidView, button_data_selection=handwritten_views.HandwrittenInvalidView.DISCARD),
            FlowStep(MainMenuView),
        ])

        assert self.controller.storage.get_pending_seed() is None


    def test_reading_failure_flow(self):
        """
        A reading failure offers to retake the photo or to cancel.
        """
        failure = WordsNotFound("11 lines found")
        self.run_sequence(self.scan_steps(failure) + [
            FlowStep(handwritten_views.HandwrittenReadFailedView, button_data_selection=handwritten_views.HandwrittenReadFailedView.RETAKE),
            FlowStep(handwritten_views.HandwrittenScanView, screen_return_value=make_photo()),
            FlowStep(handwritten_views.HandwrittenReadView, screen_return_value=failure),
            FlowStep(handwritten_views.HandwrittenReadFailedView, button_data_selection=handwritten_views.HandwrittenReadFailedView.CANCEL),
            FlowStep(MainMenuView),
        ])


    def test_reading_failure_at_edge(self):
        """
        A reading failure on words reaching the edge of the photo is passed on to the
        failure screen, which then asks to move the camera back.
        """
        def check_at_edge(view: handwritten_views.HandwrittenReadFailedView):
            assert view.at_edge

        self.run_sequence(self.scan_steps(WordsNotFound("11 lines found", at_edge=True)) + [
            FlowStep(handwritten_views.HandwrittenReadFailedView, before_run=check_at_edge, button_data_selection=handwritten_views.HandwrittenReadFailedView.CANCEL),
            FlowStep(MainMenuView),
        ])


    def test_reading_failure_card_cut_off(self):
        """
        A seed card partly out of the photo is passed on to the failure screen, which
        then asks to move the camera back so that the whole card fits.
        """
        def check_card(view: handwritten_views.HandwrittenReadFailedView):
            assert view.at_edge and view.card

        self.run_sequence(self.scan_steps(WordsNotFound("card cut off by the photo", at_edge=True, card=True)) + [
            FlowStep(handwritten_views.HandwrittenReadFailedView, before_run=check_card, button_data_selection=handwritten_views.HandwrittenReadFailedView.CANCEL),
            FlowStep(MainMenuView),
        ])


    def test_reading_failure_line_count(self):
        """
        A list with an invalid number of words is passed on to the failure screen, which
        then gives the number of words found.
        """
        def check_line_count(view: handwritten_views.HandwrittenReadFailedView):
            assert view.line_count == 11 and not view.at_edge

        self.run_sequence(self.scan_steps(WordsNotFound("11 lines found", line_count=11)) + [
            FlowStep(handwritten_views.HandwrittenReadFailedView, before_run=check_line_count, button_data_selection=handwritten_views.HandwrittenReadFailedView.CANCEL),
            FlowStep(MainMenuView),
        ])


    def test_scan_back_button(self):
        self.run_sequence([
            FlowStep(MainMenuView, button_data_selection=MainMenuView.SEEDS),
            FlowStep(seed_views.SeedsMenuView, is_redirect=True),
            FlowStep(seed_views.LoadSeedView, button_data_selection=seed_views.LoadSeedView.SCAN_HANDWRITTEN),
            FlowStep(handwritten_views.HandwrittenScanView, screen_return_value=RET_CODE__BACK_BUTTON),
            FlowStep(seed_views.LoadSeedView),
        ])


    def test_read_function_calls_engine(self):
        """
        The reading screen's function decodes the photo upright and in grayscale, shows it,
        and passes it to `read_seed` with the screen's progress, word and stage callbacks.
        """
        reading = StubReading(VALID_MNEMONIC)
        engine = MagicMock()
        engine.read_seed.return_value = reading
        screen = MagicMock()

        view = handwritten_views.HandwrittenReadView(photo=make_photo())
        with patch.dict(sys.modules, {"seedsigner.helpers.handwriting": engine}):
            with patch.object(View, "run_screen", side_effect=lambda Screen_cls, read_function: read_function(screen)):
                destination = view.run()

        photo = engine.read_seed.call_args.args[0]
        assert photo.mode == "L" and photo.size == (1944, 2592)
        screen.show_photo.assert_called_once_with(photo)
        engine.read_seed.assert_called_once_with(photo, progress=screen.report_progress, on_word=screen.show_word,
                                                  on_stage=screen.show_stage)
        assert destination.View_cls == handwritten_views.HandwrittenStatusView
        assert destination.view_args["review"].reading is reading
        assert destination.skip_current_view



class HandwrittenScreenTestBase(BaseTest):

    def setup_method(self):
        super().setup_method()

        from seedsigner.gui.renderer import Renderer
        self.mock_renderer = MagicMock()
        self.mock_renderer.canvas_width = 240
        self.mock_renderer.canvas_height = 240
        self.renderer_patch = patch.object(Renderer, "get_instance", return_value=self.mock_renderer)



class TestHandwrittenScanScreen(HandwrittenScreenTestBase):

    def build_screen(self, mock_camera: MagicMock, left_script: list, anyclick_script: list):
        from seedsigner.gui.screens.handwritten_screens import HandwrittenScanScreen

        with self.renderer_patch:
            with patch.object(Camera, "get_instance", return_value=mock_camera):
                screen = HandwrittenScanScreen()

        left_feed = list(left_script)
        anyclick_feed = list(anyclick_script)

        def check_for_low(key=None, keys=None):
            if keys is None:
                return left_feed.pop(0) if left_feed else False
            return anyclick_feed.pop(0) if anyclick_feed else False

        screen.hw_inputs = MagicMock()
        screen.hw_inputs.check_for_low.side_effect = check_for_low
        return screen


    def test_click_captures_full_resolution_photo(self):
        """
        A click made after the opening click was released takes the photo at the full
        camera resolution and returns it.
        """
        from seedsigner.gui.screens.handwritten_screens import HandwrittenScanScreen
        photo = make_photo()
        mock_camera = MagicMock()
        mock_camera.read_single_frame_preview.return_value = Image.new("RGB", (240, 320))
        mock_camera.capture_jpeg.return_value = photo

        # Opening click still held, then released, then a new click
        screen = self.build_screen(mock_camera, left_script=[], anyclick_script=[True, True, False, True])
        result = screen._run()

        assert result is photo
        # The preview and the photo come from the same camera mode: no switch at the click
        mock_camera.start_video_stream_mode.assert_not_called()
        mock_camera.start_single_frame_mode.assert_called_once_with(
            resolution=HandwrittenScanScreen.PHOTO_RESOLUTION,
            framerate=HandwrittenScanScreen.PHOTO_FRAMERATE,
        )
        mock_camera.capture_jpeg.assert_called_once_with()
        mock_camera.stop_single_frame_mode.assert_called_once()


    def test_preview_shows_whole_frame(self):
        """
        The preview shows the whole camera frame, as large as the display allows, and
        the instructions are drawn beside it without covering it.
        """
        # Portrait frame (as returned after rotation) with lime bands at the top and
        # bottom, which must stay visible
        frame = Image.new("RGB", (240, 320), "red")
        frame.paste("lime", (0, 0, 240, 40))
        frame.paste("lime", (0, 280, 240, 320))
        mock_camera = MagicMock()
        mock_camera.read_single_frame_preview.return_value = frame
        mock_camera.capture_jpeg.return_value = make_photo()

        canvas = Image.new("RGB", (240, 240))
        self.mock_renderer.canvas = canvas
        self.mock_renderer.draw = ImageDraw.Draw(canvas)
        shown = []
        self.mock_renderer.show_image.side_effect = lambda: shown.append(canvas.copy())

        screen = self.build_screen(mock_camera, left_script=[], anyclick_script=[False, True])
        screen._run()

        preview, capturing = shown
        # The image is 180 x 240, centered: its top and bottom bands are visible and
        # nothing is drawn over it
        assert preview.getpixel((120, 2)) == (0, 255, 0) and preview.getpixel((120, 237)) == (0, 255, 0)
        assert all(preview.getpixel((x, y)) == (255, 0, 0) for x in range(31, 209, 4) for y in range(32, 208, 4))
        assert preview.getpixel((29, 120)) == (0, 0, 0) and preview.getpixel((210, 120)) == (0, 0, 0)

        # One icon in each side band, near mid-height, and no text: the lit pixels of
        # each band fit in a box about one icon high
        def lit_box(image, box):
            points = [(x, y) for x in range(box[0], box[2]) for y in range(box[1], box[3]) if image.getpixel((x, y)) != (0, 0, 0)]
            assert points
            xs, ys = [x for x, _ in points], [y for _, y in points]
            return min(xs), min(ys), max(xs), max(ys)
        for box in ((0, 0, 30, 240), (210, 0, 240, 240)):
            left, top, right, bottom = lit_box(preview, box)
            assert bottom - top < 30 and top > 100 and bottom < 140

        # At the click the camera icon turns to the accent color; the image and the
        # back icon do not change
        from seedsigner.gui.components import GUIConstants
        accent = ImageColor.getrgb(GUIConstants.ACCENT_COLOR)
        assert accent not in [preview.getpixel((x, y)) for x in range(210, 240) for y in range(240)]
        assert accent in [capturing.getpixel((x, y)) for x in range(210, 240) for y in range(240)]
        assert capturing.crop((0, 0, 210, 240)).tobytes() == preview.crop((0, 0, 210, 240)).tobytes()

        # A landscape frame (other camera rotation) takes the whole width, the
        # icons go below it
        image, xy = screen.fit_display(Image.new("RGB", (320, 240)))
        assert image.size == (240, 180) and xy == (0, 30)


    def test_back_button(self):
        mock_camera = MagicMock()
        screen = self.build_screen(mock_camera, left_script=[True], anyclick_script=[])

        assert screen._run() == RET_CODE__BACK_BUTTON
        mock_camera.stop_single_frame_mode.assert_called_once()
        mock_camera.capture_jpeg.assert_not_called()



class TestHandwrittenReadingScreen(HandwrittenScreenTestBase):

    def run_screen(self, read_function):
        from seedsigner.gui.screens.handwritten_screens import HandwrittenReadingScreen

        with self.renderer_patch:
            screen = HandwrittenReadingScreen(read_function=read_function)
            screen.reading_thread.start()
            return screen, screen._run()


    def test_returns_reading_result(self):
        fractions = []

        def read_function(screen):
            screen.report_progress(0.5)
            fractions.append(screen.fraction)
            return "result"

        screen, result = self.run_screen(read_function)

        assert result == "result"
        # The first attempt fills ATTEMPT_SHARE of the bar
        assert fractions == [pytest.approx(0.5 * screen.ATTEMPT_SHARE)]
        # The last image before the next screen shows the full bar
        assert screen.fraction == 1.0


    def test_bar_never_goes_back_across_attempts(self):
        """
        A new attempt is said in words; the bar goes on from where it is, in the part
        of what is left that the attempt can fill, and never goes back.
        """
        shown = []

        def read_function(screen):
            screen.show_stage("attempt:first")
            for f in (0.2, 0.6, 0.9, 0.97):
                screen.report_progress(f)
                shown.append(screen.fraction)
            screen.show_stage("attempt:extra")
            assert screen.context == "Try 2: another layout"
            for f in (0.975, 0.98, 0.99, 0.999):
                screen.report_progress(f)
                shown.append(screen.fraction)
            screen.show_stage("attempt:first")
            assert screen.context == ""
            screen.show_stage("attempt:crossed")
            assert screen.context == "Try 3: no crossed-out words"
            # an attempt taken up again keeps its number
            screen.show_stage("attempt:extra")
            assert screen.context == "Try 2: another layout"
            screen.show_stage("attempt:crossed")
            screen.report_progress(0.9995)
            shown.append(screen.fraction)
            return None

        screen, _ = self.run_screen(read_function)

        assert shown == sorted(shown) and shown[-1] < 1.0
        # the second attempt moves the bar visibly (several pixels of 224), unlike the progress reported
        assert shown[7] - shown[3] > 0.05


    def test_stage_lines(self):
        """
        The second line says the step under way and its counter; the first one the
        document found, during the first attempt. The word outlined is shown with its
        first reading.
        """
        layout = MagicMock(count=12, kind="list")
        layout.outline.side_effect = lambda k: [(100, 200 + 100 * k), (900, 200 + 100 * k), (900, 280 + 100 * k), (100, 280 + 100 * k)]
        seen = []

        def read_function(screen):
            screen.show_photo(Image.new("L", (1944, 2592), 200))
            seen.append(screen.action)
            screen.show_stage("attempt:first")
            screen.show_stage("numbers", 3, 12)
            seen.append((screen.action, screen.counter))
            screen.show_stage("spare", 4)
            seen.append((screen.action, screen.counter))
            screen.show_stage("words", 5, 12, layout, 4)
            seen.append((screen.context, screen.action, screen.counter, screen.word))
            screen.show_word(layout, 4, MagicMock(candidates=[("humor", 0.0)]))
            screen.show_stage("words2", 1, 12, layout, 0)
            seen.append((screen.action, screen.word))
            screen.show_stage("words2", 5, 12, layout, 4)
            seen.append((screen.action, screen.word))
            screen.show_stage("fp_check", 2, 3)
            seen.append((screen.action, screen.counter))
            screen._render_reading()
            return None

        self.run_screen(read_function)

        assert seen == [
            "Finding the words",
            ("Reading line numbers", "3/12"),
            ("Looking for a fingerprint", "4"),
            ("12 lines found", "Reading the words", "5/12", None),
            ("Second reading", (0, 12, "…")),
            ("Second reading", (4, 12, "humor")),
            ("Checking the fingerprint", "2/3"),
        ]


    def test_new_attempt_starts_from_the_photo(self):
        """
        A new hypothesis shows the whole photo again, then its own area of the words;
        the first one taken up again shows what it showed.
        """
        first = MagicMock(count=12, kind="list")
        first.outline.side_effect = lambda k: [(100, 200 + 100 * k), (900, 200 + 100 * k), (900, 280 + 100 * k), (100, 280 + 100 * k)]
        other = MagicMock(count=12, kind="list")
        other.outline.side_effect = lambda k: [(150, 300 + 100 * k), (950, 300 + 100 * k), (950, 380 + 100 * k), (150, 380 + 100 * k)]
        seen = []

        def read_function(screen):
            screen.show_photo(Image.new("L", (1944, 2592), 200))
            whole = screen.whole
            screen.show_stage("attempt:first")
            screen.show_word(first, 11, MagicMock(candidates=[("zoo", 0.0)]))
            outlined, word = screen.outlined, screen.word
            screen.report_progress(0.9)
            screen.show_stage("attempt:extra")
            seen.append(screen.current_image() is whole and screen.word is None)
            screen.show_word(other, 0, MagicMock(candidates=[("abandon", 0.0)]))
            seen.append(screen.outlined == (other, 0) and screen.current_image() is not whole)
            screen.show_stage("attempt:first")
            seen.append((screen.outlined, screen.word, screen.context) == (outlined, word, "12 lines found"))
            return None

        self.run_screen(read_function)

        assert seen == [True, True, True]


    def test_redraws_while_nothing_is_reported(self):
        """
        The screen is redrawn at least every IDLE_INTERVAL seconds, the ribs of the bar
        moved, even when the reading reports nothing.
        """
        from seedsigner.gui.screens.handwritten_screens import HandwrittenReadingScreen

        def read_function(screen):
            time.sleep(2.5 * HandwrittenReadingScreen.IDLE_INTERVAL)
            return None

        screen, _ = self.run_screen(read_function)

        # two idle redraws at least, plus the last image
        assert self.mock_renderer.show_image.call_count >= 3


    def test_photo_and_words_shown(self):
        """
        The photo, then the area of the words with each word read outlined, are prepared
        by the reading thread and drawn within the screen.
        """
        layout = MagicMock(count=12)
        layout.outline.side_effect = lambda k: [(100, 200 + 100 * k), (900, 200 + 100 * k), (900, 280 + 100 * k), (100, 280 + 100 * k)]
        shown = []

        def read_function(screen):
            screen.show_photo(Image.new("L", (1944, 2592), 200))
            screen._render_reading()
            shown.append(screen.image.size)
            for k in range(12):
                screen.show_word(layout, k, MagicMock(candidates=[("abandon", 0.0)]))
                screen._render_reading()
            return screen

        screen, _ = self.run_screen(read_function)

        left, top, right, bottom = screen.area
        assert shown[0][0] <= right - left and shown[0][1] <= bottom - top
        assert screen.word == (11, 12, "abandon")
        assert screen.image.width <= right - left - screen.TEXT_WIDTH and screen.image.height <= bottom - top
        # The word being read is outlined in the accent color
        from PIL import ImageColor
        from seedsigner.gui.components import GUIConstants
        assert ImageColor.getrgb(GUIConstants.ACCENT_COLOR) in screen.image.getdata()


    def test_trail_of_the_words_read(self):
        """
        The words already read in the pass stay outlined in TRAIL_COLOR, the word read
        in the accent color; a new pass starts a new trail.
        """
        from PIL import ImageColor
        from seedsigner.gui.components import GUIConstants
        layout = MagicMock(count=12, kind="list", numbers_linked=12)
        layout.outline.side_effect = lambda k: [(100, 200 + 200 * k), (900, 200 + 200 * k), (900, 360 + 200 * k), (100, 360 + 200 * k)]
        seen = []

        def read_function(screen):
            screen.show_photo(Image.new("L", (1944, 2592), 200))
            trail, accent = ImageColor.getrgb(screen.TRAIL_COLOR), ImageColor.getrgb(GUIConstants.ACCENT_COLOR)

            def rows(color):
                # rows of the thumbnail holding the color
                image = screen.current_image()
                return {y for y in range(image.height) for x in range(image.width) if image.getpixel((x, y)) == color}

            screen.show_word(layout, 0, MagicMock(candidates=[("abandon", 0.0)]))
            seen.append((len(rows(trail)), len(rows(accent)) > 0))
            for k in range(1, 4):
                screen.show_word(layout, k, MagicMock(candidates=[("abandon", 0.0)]))
            # three words read before the one outlined, drawn on the thumbnail of the pass
            seen.append(screen.trail[1] == [0, 1, 2, 3] and len(rows(trail)) > 0)
            first_pass = len(rows(trail))
            screen.show_stage("words2", 1, 12, layout, 0)
            seen.append(screen.trail[1] == [0] and len(rows(trail)) < first_pass)
            return None

        self.run_screen(read_function)

        assert seen == [(0, True), True, True]


    def test_redraw_on_each_word(self):
        """
        A change of the word outlined is drawn WORD_INTERVAL seconds after the last
        redraw; other changes RENDER_INTERVAL seconds after it.
        """
        layout = MagicMock(count=12, kind="list")
        layout.outline.side_effect = lambda k: [(100, 200), (900, 200), (900, 280), (100, 280)]
        seen = []

        def read_function(screen):
            screen.WORD_INTERVAL = 0.2
            screen.show_word(layout, 0, MagicMock(candidates=[("abandon", 0.0)]))
            shown, version = screen.outlined, screen.version
            screen.show_word(layout, 1, MagicMock(candidates=[("ability", 0.0)]))
            seen.append(screen.render_due(10.25, 10.0, version, shown))
            seen.append(screen.render_due(10.15, 10.0, version, shown))
            # a change of the counter only
            screen.show_stage("spare", 2)
            seen.append(screen.render_due(10.25, 10.0, version, screen.outlined))
            seen.append(screen.render_due(10.75, 10.0, version, screen.outlined))
            seen.append(screen.render_due(10.5, 10.0, screen.version, screen.outlined))
            return None

        self.run_screen(read_function)

        assert seen == [True, False, False, True, False]


    def test_word_text_on_a_layout_not_confirmed(self):
        """
        A list with numbers on some of its lines only is a hypothesis: the text of its
        words is shown as UNCONFIRMED_WORD says. Cards, lists numbered on every line and
        complete lists without numbers are confirmed.
        """
        from seedsigner.gui.screens.handwritten_screens import HandwrittenReadingScreen
        layouts = [
            MagicMock(kind="list", count=21, numbers_linked=14, numbers_incoherent=False, complete=False),
            MagicMock(kind="list", count=12, numbers_linked=12, numbers_incoherent=False, complete=False),
            MagicMock(kind="list", count=24, numbers_linked=0, numbers_incoherent=False, complete=True),
            MagicMock(kind="list", count=24, numbers_linked=24, numbers_incoherent=True, complete=True),
            MagicMock(kind="card12", count=12, numbers_linked=0, numbers_incoherent=False, complete=False),
        ]
        for layout in layouts:
            layout.outline.side_effect = lambda k: [(100, 200), (900, 200), (900, 280), (100, 280)]
        seen = []
        pixels = {}

        def read_function(screen):
            from PIL import ImageDraw
            screen.renderer.canvas = Image.new("RGB", (240, 240))
            screen.renderer.draw = ImageDraw.Draw(screen.renderer.canvas)
            seen.extend(screen.confirmed(layout) for layout in layouts)
            screen.show_photo(Image.new("L", (1944, 2592), 200))
            for shown in ("show", "dim", "hide"):
                screen.UNCONFIRMED_WORD = shown
                screen.drawn = (None, None)
                screen.show_word(layouts[0], 3, MagicMock(candidates=[("lion", 0.0)]))
                screen._render_reading()
                left, top, right, bottom = screen.area
                region = screen.renderer.canvas.crop((left + screen.image.width, top, right, bottom))
                pixels[shown] = (max(sum(p) for p in region.getdata()), sum(1 for p in region.getdata() if sum(p) > 60))
            return None

        with patch.object(HandwrittenReadingScreen, "UNCONFIRMED_WORD", "show"):
            self.run_screen(read_function)

        assert seen == [False, True, True, False, True]
        # white text, gray text, number only (gray)
        assert pixels["show"][0] > pixels["dim"][0] == pixels["hide"][0]
        assert pixels["dim"][1] > pixels["hide"][1] > 0


    def test_reraises_reading_error(self):
        def read_function(screen):
            raise ValueError("No seed words found")

        with pytest.raises(ValueError, match="No seed words found"):
            self.run_screen(read_function)



class TestHandwrittenNoticeScreen(HandwrittenScreenTestBase):

    def build_screen(self, inputs: list):
        from seedsigner.gui.screens.handwritten_screens import HandwrittenNoticeScreen
        self.mock_renderer.canvas = Image.new("RGB", (240, 240))

        with self.renderer_patch:
            screen = HandwrittenNoticeScreen(
                title="12 words read",
                status_headline="Words to check: 2",
                paragraphs=[(None, "Some words could not be read with certainty. " * 4)] * 3,
            )
        screen.hw_inputs = MagicMock()
        screen.hw_inputs.wait_for.side_effect = inputs
        return screen


    def run(self, screen):
        # The hardware buttons module is mocked: give it a real list of click keys
        from seedsigner.hardware.buttons import HardwareButtonsConstants as Keys
        with patch.object(Keys, "KEYS__ANYCLICK", [Keys.KEY_PRESS]):
            return screen._run()


    def test_click_ignored_until_end_of_text(self):
        """
        Clicks do nothing until the last page of text is shown and the button selected.
        """
        from seedsigner.hardware.buttons import HardwareButtonsConstants as Keys
        pages = len(self.build_screen([]).pages)
        assert pages > 1

        screen = self.build_screen([Keys.KEY_PRESS] + [Keys.KEY_DOWN] * (pages - 1) + [Keys.KEY_PRESS, Keys.KEY_DOWN, Keys.KEY_PRESS])
        assert self.run(screen) == 0
        assert screen.page == pages - 1
        assert screen.hw_inputs.wait_for.call_count == pages + 3


    def test_up_deselects_button(self):
        from seedsigner.hardware.buttons import HardwareButtonsConstants as Keys
        screen = self.build_screen([Keys.KEY_DOWN] * 20 + [Keys.KEY_UP, Keys.KEY_PRESS, Keys.KEY_DOWN, Keys.KEY_PRESS])
        assert self.run(screen) == 0
        assert screen.hw_inputs.wait_for.call_count == 24
