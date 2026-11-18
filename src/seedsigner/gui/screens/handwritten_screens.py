import time

from dataclasses import dataclass
from gettext import gettext as _
from typing import Callable
from PIL import Image

from seedsigner.gui.components import Fonts, GUIConstants
from seedsigner.hardware.buttons import HardwareButtonsConstants
from seedsigner.hardware.camera import Camera
from seedsigner.helpers.l10n import mark_for_translation as _mft
from seedsigner.models.threads import BaseThread

from .screen import RET_CODE__BACK_BUTTON, BaseScreen, BaseTopNavScreen, ButtonListScreen



@dataclass
class HandwrittenScanScreen(BaseScreen):
    """
    Live camera preview to frame the handwritten words. A click takes a still photo at
    the full resolution of the camera and returns it as a PIL Image; KEY_LEFT returns
    RET_CODE__BACK_BUTTON.

    The preview comes from the camera that takes the photo and shows its whole field of
    view, so the photo is the framed image, taken without delay after the click.
    """
    # Full resolution of the OV5647 sensor; the sensor only provides it at up to 15fps,
    # and the framerate also caps the exposure time (1/15s instead of 1/24s in dim light).
    PHOTO_RESOLUTION = (2592, 1944)
    PHOTO_FRAMERATE = 15

    # Size of the preview frames, before rotation (4:3 like the photo)
    PREVIEW_RESOLUTION = (320, 240)


    def __post_init__(self):
        super().__post_init__()

        self.camera = Camera.get_instance()

        self.camera.start_single_frame_mode(resolution=self.PHOTO_RESOLUTION, framerate=self.PHOTO_FRAMERATE)

        self.instructions_font = Fonts.get_font(GUIConstants.get_body_font_name(), GUIConstants.get_button_font_size())


    def _render_instructions(self, text: str, color: str):
        self.renderer.draw.text(
            xy=(
                int(self.canvas_width/2),
                self.canvas_height - GUIConstants.EDGE_PADDING
            ),
            text=text,
            fill=color,
            font=self.instructions_font,
            stroke_width=4,
            stroke_fill=GUIConstants.BACKGROUND_COLOR,
            anchor="ms"
        )


    def _capture_photo(self) -> Image.Image:
        try:
            # expand: the photo keeps the whole field of view once rotated to portrait
            return self.camera.capture_frame(expand=True)
        finally:
            self.camera.stop_single_frame_mode()


    def _run(self):
        # If the user continues holding the button that brought them into this screen,
        # it must not take the photo: only a click after all buttons were released does.
        is_maybe_still_holding = True

        while True:
            if self.hw_inputs.check_for_low(HardwareButtonsConstants.KEY_LEFT):
                # Have to manually update last input time since we're not in a wait_for loop
                self.hw_inputs.update_last_input_time()
                self.camera.stop_single_frame_mode()
                return RET_CODE__BACK_BUTTON

            if not self.hw_inputs.check_for_low(keys=HardwareButtonsConstants.KEYS__ANYCLICK):
                is_maybe_still_holding = False

            elif not is_maybe_still_holding:
                self.hw_inputs.update_last_input_time()

                with self.renderer.lock:
                    self._render_instructions(_("Capturing image..."), GUIConstants.ACCENT_COLOR)
                    self.renderer.show_image()

                return self._capture_photo()

            frame = self.camera.read_single_frame_preview(self.PREVIEW_RESOLUTION)

            # The whole frame, fitted to the display height, so that nothing outside of
            # the preview is in the photo
            width = round(frame.width * self.canvas_height / frame.height)
            with self.renderer.lock:
                self.renderer.draw.rectangle((0, 0, self.canvas_width, self.canvas_height), fill=GUIConstants.BACKGROUND_COLOR)
                self.renderer.canvas.paste(frame.resize((width, self.canvas_height)), ((self.canvas_width - width) // 2, 0))
                self._render_instructions("< " + _("back") + "  |  " + _("click a button"), GUIConstants.BODY_FONT_COLOR)
                self.renderer.show_image()



@dataclass
class HandwrittenReadingScreen(BaseTopNavScreen):
    """
    Runs `read_function(progress)` in a background thread and displays the progress it
    reports through `progress(fraction)`.

    Returns the value returned by `read_function`; re-raises the exception it raised.
    """
    title: str = _mft("Reading words")
    show_back_button: bool = False
    read_function: Callable = None


    def __post_init__(self):
        super().__post_init__()
        self.fraction = 0.0
        self.font = Fonts.get_font(GUIConstants.get_body_font_name(), GUIConstants.get_body_font_size())

        self.reading_thread = HandwrittenReadingScreen.ReadingThread(
            read_function=self.read_function,
            progress=self.report_progress,
        )
        self.threads.append(self.reading_thread)


    def report_progress(self, fraction: float):
        """ Called from the reading thread; the screen redraws on its next loop """
        self.fraction = min(max(fraction, 0.0), 1.0)


    def _render(self):
        super()._render()
        self._render_progress()


    def _render_progress(self):
        draw = self.renderer.draw
        draw.rectangle((0, self.top_nav.height, self.canvas_width, self.canvas_height), fill=GUIConstants.BACKGROUND_COLOR)

        center_x = int(self.canvas_width/2)
        bar_y = int((self.top_nav.height + self.canvas_height)/2)
        # Progress bar
        bar_left = GUIConstants.EDGE_PADDING
        bar_right = self.canvas_width - GUIConstants.EDGE_PADDING
        draw.rectangle((bar_left, bar_y, bar_right, bar_y + GUIConstants.COMPONENT_PADDING), outline=GUIConstants.INACTIVE_COLOR)
        draw.rectangle(
            (bar_left, bar_y, bar_left + int((bar_right - bar_left) * self.fraction), bar_y + GUIConstants.COMPONENT_PADDING),
            fill=GUIConstants.ACCENT_COLOR
        )

        draw.text(
            xy=(center_x, bar_y + 2*GUIConstants.COMPONENT_PADDING),
            text=f"{self.fraction:.0%}",
            fill=GUIConstants.LABEL_FONT_COLOR,
            font=self.font,
            anchor="mt",
        )


    def _run(self):
        displayed = None
        while self.reading_thread.is_alive():
            if displayed != self.fraction:
                displayed = self.fraction
                with self.renderer.lock:
                    self._render_progress()
                    self.renderer.show_image()
            time.sleep(0.1)

        if self.reading_thread.error is not None:
            raise self.reading_thread.error
        return self.reading_thread.result


    class ReadingThread(BaseThread):
        def __init__(self, read_function: Callable, progress: Callable):
            super().__init__()
            self.read_function = read_function
            self.progress = progress
            self.result = None
            self.error = None


        def run(self):
            try:
                self.result = self.read_function(self.progress)
            except Exception as e:
                # Re-raised by the screen in the main thread
                self.error = e



@dataclass
class HandwrittenWordScreen(ButtonListScreen):
    """
    Candidate words as a bottom button list, with the photographed handwritten word
    displayed above it.
    """
    word_image: Image.Image = None


    def __post_init__(self):
        self.is_bottom_list = True
        super().__post_init__()

        # Scale the word image to the space between the top nav and the buttons
        top = self.top_nav.height
        max_width = self.canvas_width - 2*GUIConstants.EDGE_PADDING
        max_height = self.buttons[0].screen_y - top - 2*GUIConstants.LIST_ITEM_PADDING
        scale = min(max_width / self.word_image.width, max_height / self.word_image.height)
        image = self.word_image.convert("RGB").resize(
            (max(1, int(self.word_image.width * scale)), max(1, int(self.word_image.height * scale))),
            Image.Resampling.LANCZOS
        )
        self.paste_images.append((
            image,
            (int((self.canvas_width - image.width)/2), top + int((self.buttons[0].screen_y - top - image.height)/2))
        ))
