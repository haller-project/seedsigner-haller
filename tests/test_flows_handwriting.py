import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Must import test base before the Controller
from base import BaseTest, FlowTest, FlowStep, FlowTestInvalidButtonDataSelectionException

from seedsigner.helpers.handwriting import WordsNotFound
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
        self.analyze_calls = []


    def analyze(self, confirmed: dict[int, str] = None):
        confirmed = confirmed or {}
        self.analyze_calls.append(dict(confirmed))
        mnemonic = [confirmed.get(k, word) for k, word in enumerate(self.mnemonic)]
        uncertain = {k: words for k, words in self.uncertain.items() if k not in confirmed}
        if not self.is_valid:
            status = "invalid"
        elif uncertain:
            status = "uncertain"
        else:
            status = "valid"
        return StubAnalysis(mnemonic, status, uncertain)



def make_photo() -> Image.Image:
    return Image.new("RGB", (1944, 2592), "white")



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
        A reading with a valid checksum goes through the status and review screens and
        loads the seed through SeedFinalizeView.
        """
        reading = StubReading(VALID_MNEMONIC)
        self.run_sequence(self.scan_steps(reading) + [
            FlowStep(handwritten_views.HandwrittenStatusView, button_data_selection=handwritten_views.HandwrittenStatusView.REVIEW),
            FlowStep(handwritten_views.HandwrittenReviewView, button_data_selection=handwritten_views.HandwrittenReviewView.DONE),
            FlowStep(seed_views.SeedFinalizeView, button_data_selection=seed_views.SeedFinalizeView.FINALIZE),
            FlowStep(seed_views.SeedOptionsView),
        ])

        assert self.controller.storage.seeds[0].mnemonic_list == VALID_MNEMONIC


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
            FlowStep(handwritten_views.HandwrittenStatusView, button_data_selection=handwritten_views.HandwrittenStatusView.REVIEW),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=2),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=1),  # "ability"
            FlowStep(handwritten_views.HandwrittenReviewView, before_run=check_review, screen_return_value=2),
            FlowStep(handwritten_views.HandwrittenWordView, screen_return_value=0),  # back to "abandon"
            FlowStep(handwritten_views.HandwrittenReviewView, button_data_selection=handwritten_views.HandwrittenReviewView.DONE),
            FlowStep(seed_views.SeedFinalizeView),
        ])

        # The word view analyzes without the word's own confirmation, the review with it;
        # each distinct analysis is computed once
        assert reading.analyze_calls == [{}, {2: "ability"}, {2: "abandon"}]
        assert self.controller.storage.get_pending_seed().mnemonic_list == VALID_MNEMONIC


    def test_done_requires_uncertain_words_confirmed(self):
        """
        "Done" is not offered while an uncertain word has not been confirmed.
        """
        reading = StubReading(VALID_MNEMONIC, uncertain={2: ["ability"]})
        with pytest.raises(FlowTestInvalidButtonDataSelectionException):
            self.run_sequence(self.scan_steps(reading) + [
                FlowStep(handwritten_views.HandwrittenStatusView, button_data_selection=handwritten_views.HandwrittenStatusView.REVIEW),
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
            FlowStep(handwritten_views.HandwrittenStatusView, button_data_selection=handwritten_views.HandwrittenStatusView.REVIEW),
            FlowStep(handwritten_views.HandwrittenReviewView, screen_return_value=11),
            FlowStep(handwritten_views.HandwrittenWordView, button_data_selection=handwritten_views.HandwrittenWordView.TYPE_WORD),
            FlowStep(handwritten_views.HandwrittenWordEntryView, screen_return_value=RET_CODE__BACK_BUTTON),
            FlowStep(handwritten_views.HandwrittenWordView, button_data_selection=handwritten_views.HandwrittenWordView.TYPE_WORD),
            FlowStep(handwritten_views.HandwrittenWordEntryView, screen_return_value="about"),
            FlowStep(handwritten_views.HandwrittenReviewView, before_run=check_review, button_data_selection=handwritten_views.HandwrittenReviewView.DONE),
            FlowStep(seed_views.SeedFinalizeView),
        ])


    def test_invalid_checksum_flow(self):
        """
        "Done" with an invalid checksum shows the invalid mnemonic screen, which returns
        to the review or discards the reading.
        """
        reading = StubReading(INVALID_MNEMONIC, is_valid=False)
        self.run_sequence(self.scan_steps(reading) + [
            FlowStep(handwritten_views.HandwrittenStatusView, button_data_selection=handwritten_views.HandwrittenStatusView.REVIEW),
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
        The reading screen's function passes the photo and the progress callback to
        `read_seed`.
        """
        photo = make_photo()
        reading = StubReading(VALID_MNEMONIC)
        engine = MagicMock()
        engine.read_seed.return_value = reading
        progress = MagicMock()

        view = handwritten_views.HandwrittenReadView(photo=photo)
        with patch.dict(sys.modules, {"seedsigner.helpers.handwriting": engine}):
            with patch.object(View, "run_screen", side_effect=lambda Screen_cls, read_function: read_function(progress)):
                destination = view.run()

        engine.read_seed.assert_called_once_with(photo, progress=progress)
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
        mock_camera.capture_frame.return_value = photo

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
        mock_camera.capture_frame.assert_called_once_with(expand=True)
        mock_camera.stop_single_frame_mode.assert_called_once()


    def test_back_button(self):
        mock_camera = MagicMock()
        screen = self.build_screen(mock_camera, left_script=[True], anyclick_script=[])

        assert screen._run() == RET_CODE__BACK_BUTTON
        mock_camera.stop_single_frame_mode.assert_called_once()
        mock_camera.capture_frame.assert_not_called()



class TestHandwrittenReadingScreen(HandwrittenScreenTestBase):

    def run_screen(self, read_function):
        from seedsigner.gui.screens.handwritten_screens import HandwrittenReadingScreen

        with self.renderer_patch:
            screen = HandwrittenReadingScreen(read_function=read_function)
            screen.reading_thread.start()
            return screen, screen._run()


    def test_returns_reading_result(self):
        def read_function(progress):
            progress(0.5)
            return "result"

        screen, result = self.run_screen(read_function)

        assert result == "result"
        assert screen.fraction == 0.5


    def test_reraises_reading_error(self):
        def read_function(progress):
            raise ValueError("No seed words found")

        with pytest.raises(ValueError, match="No seed words found"):
            self.run_screen(read_function)
