import time

from dataclasses import dataclass, field
from gettext import gettext as _
from typing import Callable
from PIL import Image, ImageDraw

from seedsigner.gui.components import Button, Fonts, GUIConstants
from seedsigner.hardware.buttons import HardwareButtonsConstants
from seedsigner.hardware.camera import Camera
from seedsigner.helpers.l10n import mark_for_translation as _mft
from seedsigner.models.threads import BaseThread

from .screen import RET_CODE__BACK_BUTTON, BaseScreen, BaseTopNavScreen, ButtonListScreen, ButtonOption



@dataclass
class HandwrittenScanScreen(BaseScreen):
    """
    Live camera preview to frame the handwritten words. A click takes a still photo at
    the full resolution of the camera and returns it as JPEG data (decoded by
    `Camera.decode_frame()`); KEY_LEFT returns RET_CODE__BACK_BUTTON.

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


    def _capture_photo(self) -> bytes:
        try:
            return self.camera.capture_jpeg()
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
    Runs `read_function(screen)` in a background thread, which reports to the screen
    the photo (`show_photo`), each word read (`show_word`) and the progress
    (`report_progress`). The photo is shown while the words are searched, then the
    area of the words with the word being read outlined, next to that word.

    The images are prepared by the reading thread, and the screen is redrawn at most
    every RENDER_INTERVAL seconds: a redraw costs the reading about 11ms of processor
    time on a Pi Zero.

    Returns the value returned by `read_function`; re-raises the exception it raised.
    """
    title: str = _mft("Reading words")
    show_back_button: bool = False
    read_function: Callable = None

    RENDER_INTERVAL = 0.7
    TEXT_WIDTH = 100            # width left for the word read, right of the words area
    BAR_HEIGHT = 8


    def __post_init__(self):
        super().__post_init__()
        self.fraction = 0.0
        self.photo = None       # the photo, while the words are being read
        self.sheet = None       # (thumbnail of the words area, its origin and scale on the photo)
        self.image = None       # image shown above the bar
        self.word = None        # (index, count, text) of the last word read
        self.version = 0        # incremented by the reading thread on each change

        self.bar_top = self.canvas_height - GUIConstants.EDGE_PADDING - self.BAR_HEIGHT
        self.area = (GUIConstants.EDGE_PADDING, self.top_nav.height, self.canvas_width - GUIConstants.EDGE_PADDING,
                     self.bar_top - GUIConstants.COMPONENT_PADDING)
        self.label_font = Fonts.get_font(GUIConstants.get_body_font_name(), GUIConstants.LABEL_FONT_SIZE)
        self.word_font = Fonts.get_font(GUIConstants.FIXED_WIDTH_EMPHASIS_FONT_NAME, GUIConstants.get_top_nav_title_font_size())

        self.reading_thread = HandwrittenReadingScreen.ReadingThread(read_function=self.read_function, screen=self)
        self.threads.append(self.reading_thread)


    @staticmethod
    def thumbnail(image: Image.Image, width: int, height: int) -> Image.Image:
        """ `image` reduced to fit in width x height, in RGB """
        factor = max(1, -(-image.height // height), -(-image.width // width))
        return image.reduce(factor).convert("RGB")


    def report_progress(self, fraction: float):
        """ Called from the reading thread; the screen redraws on its next loop """
        self.fraction = min(max(fraction, 0.0), 1.0)
        self.version += 1


    def show_photo(self, photo: Image.Image):
        """ Called from the reading thread with the upright photo, before the reading """
        left, top, right, bottom = self.area
        self.photo = photo
        self.image = self.thumbnail(photo, right - left, bottom - top)
        self.version += 1


    def show_word(self, layout, k: int, reading):
        """ Called from the reading thread after the first reading of each word """
        if k == 0:
            # Thumbnail of the area of the words, with a margin
            corners = [p for i in range(layout.count) for p in layout.outline(i)]
            x0, y0 = min(x for x, _ in corners), min(y for _, y in corners)
            x1, y1 = max(x for x, _ in corners), max(y for _, y in corners)
            margin = 0.05 * max(x1 - x0, y1 - y0)
            box = tuple(int(v) for v in (max(0, x0 - margin), max(0, y0 - margin),
                                         min(self.photo.width, x1 + margin), min(self.photo.height, y1 + margin)))
            left, top, right, bottom = self.area
            thumbnail = self.thumbnail(self.photo.crop(box), right - left - self.TEXT_WIDTH, bottom - top)
            self.sheet = (thumbnail, box[:2], thumbnail.width / (box[2] - box[0]))

        thumbnail, (x0, y0), scale = self.sheet
        image = thumbnail.copy()
        ImageDraw.Draw(image).polygon([((x - x0) * scale, (y - y0) * scale) for x, y in layout.outline(k)],
                                      outline=GUIConstants.ACCENT_COLOR, width=2)
        self.image = image
        self.word = (k, layout.count, reading.candidates[0][0])
        self.version += 1


    def _render(self):
        super()._render()
        self._render_reading()


    def _render_reading(self):
        draw = self.renderer.draw
        draw.rectangle((0, self.top_nav.height, self.canvas_width, self.canvas_height), fill=GUIConstants.BACKGROUND_COLOR)
        left, top, right, bottom = self.area

        if self.image is not None:
            x = left if self.word else int((left + right - self.image.width) / 2)
            self.renderer.canvas.paste(self.image, (x, top + int((bottom - top - self.image.height) / 2)))

        if self.word is not None:
            # Number and text of the word, right of the words area
            k, count, text = self.word
            x, y = int((left + self.image.width + right) / 2), int((top + bottom) / 2)
            draw.text((x, y - GUIConstants.LIST_ITEM_PADDING), f"{k + 1}/{count}", fill=GUIConstants.LABEL_FONT_COLOR,
                      font=self.label_font, anchor="mb")
            draw.text((x, y + GUIConstants.LIST_ITEM_PADDING), text, fill=GUIConstants.BODY_FONT_COLOR,
                      font=self.word_font, anchor="mt")

        # Progress bar
        draw.rectangle((left, self.bar_top, right, self.bar_top + self.BAR_HEIGHT), outline=GUIConstants.INACTIVE_COLOR)
        draw.rectangle((left, self.bar_top, left + int((right - left) * self.fraction), self.bar_top + self.BAR_HEIGHT),
                       fill=GUIConstants.ACCENT_COLOR)


    def _run(self):
        displayed, last_render = None, 0.0
        while self.reading_thread.is_alive():
            if displayed != self.version and time.time() - last_render >= self.RENDER_INTERVAL:
                displayed, last_render = self.version, time.time()
                with self.renderer.lock:
                    self._render_reading()
                    self.renderer.show_image()
            time.sleep(0.05)

        if self.reading_thread.error is not None:
            raise self.reading_thread.error

        # The full bar, on the last image before the next screen
        self.fraction = 1.0
        with self.renderer.lock:
            self._render_reading()
            self.renderer.show_image()
        return self.reading_thread.result


    class ReadingThread(BaseThread):
        def __init__(self, read_function: Callable, screen):
            super().__init__()
            self.read_function = read_function
            self.screen = screen
            self.result = None
            self.error = None


        def run(self):
            try:
                self.result = self.read_function(self.screen)
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



@dataclass
class HandwrittenNoticeScreen(BaseTopNavScreen):
    """
    Full screen notice to read before going on: a colored headline, then `paragraphs`,
    (icon name or None, text) pairs, the icon being drawn before its paragraph. KEY_DOWN
    and KEY_UP turn the pages of the text. The button below it can only be selected
    from the last page, so that a click made by mistake does not skip the notice.

    Returns 0 when the button is clicked.
    """
    show_back_button: bool = False
    status_headline: str = None
    status_color: str = GUIConstants.WARNING_COLOR
    paragraphs: list = None
    icon_color: str = GUIConstants.WARNING_COLOR
    button_data: list = field(default_factory=lambda: [ButtonOption("I understand")])

    ARROW_HALF_WIDTH = 10


    def __post_init__(self):
        super().__post_init__()
        button_y = self.canvas_height - GUIConstants.EDGE_PADDING - GUIConstants.BUTTON_HEIGHT
        self.button = Button(
            text=_(self.button_data[0].button_label),
            screen_x=GUIConstants.EDGE_PADDING,
            screen_y=button_y,
            width=self.canvas_width - 2 * GUIConstants.EDGE_PADDING,
            height=GUIConstants.BUTTON_HEIGHT,
            font_name=GUIConstants.get_button_font_name(),
            font_size=GUIConstants.get_button_font_size(),
            is_scrollable_text=False,
        )
        self.components.append(self.button)

        self.text_top = self.top_nav.height
        # Room left below the text for the down arrow
        self.text_height = button_y - 2 * GUIConstants.COMPONENT_PADDING - self.text_top
        self.text_image, self.lines = self._render_text_image()

        # First line of each page: a page shows the whole lines that fit, and starts
        # with a paragraph unless that paragraph is longer than a page
        self.pages = [0]
        while True:
            top = self.lines[self.pages[-1]][0]
            following = [k for k, (_, bottom, _) in enumerate(self.lines) if bottom > top + self.text_height]
            if not following:
                break
            k = following[0]
            start = min(j for j, line in enumerate(self.lines) if line[2] == self.lines[k][2])
            self.pages.append(start if start > self.pages[-1] else k)
        self.page = 0


    def _render_text_image(self) -> tuple[Image.Image, list[tuple[int, int, int]]]:
        """
        The headline and paragraphs, rendered once into an image as wide as the screen,
        and the (top, bottom, paragraph) of each line of text in it.
        """
        padding = 2 * GUIConstants.EDGE_PADDING
        body_font = Fonts.get_font(GUIConstants.get_body_font_name(), GUIConstants.get_body_font_size())
        headline_font = Fonts.get_font(GUIConstants.get_button_font_name(), GUIConstants.get_body_font_size())
        icon_font = Fonts.get_font(GUIConstants.ICON_FONT_NAME__SEEDSIGNER, GUIConstants.ICON_FONT_SIZE, file_extension="otf")
        ascent, descent = body_font.getmetrics()
        line_height = ascent + descent

        def wrap(text: str, font, width: int) -> list[str]:
            lines = []
            for word in text.split():
                if lines and font.getlength(lines[-1] + " " + word) <= width:
                    lines[-1] += " " + word
                else:
                    lines.append(word)
            return lines

        # (x, y, text, font, color, anchor) of everything to draw
        items = []
        lines = []
        y = int(GUIConstants.COMPONENT_PADDING / 2)
        paragraphs = ([(None, self.status_headline)] if self.status_headline else []) + list(self.paragraphs or [])
        for k, (icon_name, text) in enumerate(paragraphs):
            is_headline = k == 0 and self.status_headline
            x = padding
            if icon_name:
                items.append((x, y, icon_name, icon_font, self.icon_color, "la"))
                x += GUIConstants.ICON_FONT_SIZE + GUIConstants.COMPONENT_PADDING
            font = headline_font if is_headline else body_font
            for line in wrap(text, font, self.canvas_width - padding - x):
                if is_headline:
                    items.append((int(self.canvas_width / 2), y, line, font, self.status_color, "ma"))
                else:
                    items.append((x, y, line, font, GUIConstants.BODY_FONT_COLOR, "la"))
                # The headline is kept on the page of the first paragraph
                lines.append((y, y + line_height, max(k, 1) if self.status_headline else k))
                y += line_height
            y += GUIConstants.COMPONENT_PADDING

        image = Image.new("RGB", (self.canvas_width, y), GUIConstants.BACKGROUND_COLOR)
        draw = ImageDraw.Draw(image)
        for x, y, text, font, color, anchor in items:
            draw.text((x, y), text, font=font, fill=color, anchor=anchor)
        return image, lines


    def _render(self):
        super()._render()
        self._render_text()


    def _render_text(self):
        """ The current page of text, with arrows when there are other pages or the button below """
        first = self.pages[self.page]
        last = self.pages[self.page + 1] if self.page + 1 < len(self.pages) else len(self.lines)
        top, bottom = self.lines[first][0], self.lines[last - 1][1]
        self.image_draw.rectangle((0, self.text_top, self.canvas_width, self.button.screen_y - 1), fill=GUIConstants.BACKGROUND_COLOR)
        self.canvas.paste(self.text_image.crop((0, top, self.canvas_width, bottom)), (0, self.text_top))

        x, w = int(self.canvas_width / 2), self.ARROW_HALF_WIDTH
        if self.page > 0:
            y = self.top_nav.height - 10
            self.image_draw.line((x - w, y + 6, x, y, x + w, y + 6), fill=GUIConstants.BUTTON_FONT_COLOR)
        if not self.button.is_selected:
            y = self.button.screen_y - 2
            self.image_draw.line((x - w, y - 6, x, y, x + w, y - 6), fill=GUIConstants.BUTTON_FONT_COLOR)


    def _run(self):
        while True:
            user_input = self.hw_inputs.wait_for(
                [HardwareButtonsConstants.KEY_UP, HardwareButtonsConstants.KEY_DOWN] + HardwareButtonsConstants.KEYS__ANYCLICK
            )

            with self.renderer.lock:
                if user_input == HardwareButtonsConstants.KEY_DOWN:
                    if self.page + 1 < len(self.pages):
                        self.page += 1
                    else:
                        self.button.is_selected = True

                elif user_input == HardwareButtonsConstants.KEY_UP:
                    if self.button.is_selected:
                        self.button.is_selected = False
                    else:
                        self.page = max(self.page - 1, 0)

                elif user_input in HardwareButtonsConstants.KEYS__ANYCLICK and self.button.is_selected:
                    return 0

                else:
                    continue

                self._render()
                self.renderer.show_image()
