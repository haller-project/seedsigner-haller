import time

from dataclasses import dataclass, field
from gettext import gettext as _
from typing import Callable
from PIL import Image, ImageDraw

from seedsigner.gui.components import Button, FontAwesomeIconConstants, Fonts, GUIConstants, SeedSignerIconConstants
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
    view, as large as the display allows: what is photographed is exactly what is shown.
    The controls are shown by two icons, without text, in the bands left free beside the
    image: a back arrow on the side of the joystick, a camera on the side of the buttons,
    which turns to the accent color when the photo is being taken. When the image is
    wider than tall they go in the band below it. Nothing covers the image. The photo is
    taken without delay after the click.
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

        self.back_icon_font = Fonts.get_font(GUIConstants.ICON_FONT_NAME__SEEDSIGNER, GUIConstants.ICON_FONT_SIZE, file_extension="otf")
        self.camera_icon_font = Fonts.get_font(GUIConstants.ICON_FONT_NAME__FONT_AWESOME, GUIConstants.ICON_FONT_SIZE - 2)


    def fit_display(self, frame: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
        """
        The whole `frame`, scaled to the largest size that fits in the display, and the
        position that centers it.
        """
        scale = min(self.canvas_width / frame.width, self.canvas_height / frame.height)
        width, height = round(frame.width * scale), round(frame.height * scale)
        return frame.resize((width, height)), ((self.canvas_width - width) // 2, (self.canvas_height - height) // 2)


    def _render_frame(self, frame: Image.Image, camera_color: str):
        """
        Draws the whole `frame`, the back icon in the band on its left and the camera
        icon, in `camera_color`, in the band on its right; both side by side in the band
        below a frame wider than tall.
        """
        image, (x, y) = self.fit_display(frame)
        self.renderer.draw.rectangle((0, 0, self.canvas_width, self.canvas_height), fill=GUIConstants.BACKGROUND_COLOR)
        self.renderer.canvas.paste(image, (x, y))

        if x > 0:
            back_center = (x // 2, self.canvas_height // 2)
            camera_center = (x + image.width + (self.canvas_width - x - image.width) // 2, self.canvas_height // 2)
        else:
            band_center = (y + image.height + self.canvas_height) // 2
            back_center = (self.canvas_width // 4, band_center)
            camera_center = (self.canvas_width * 3 // 4, band_center)

        self.renderer.draw.text(back_center, SeedSignerIconConstants.BACK, fill=GUIConstants.BODY_FONT_COLOR,
                                font=self.back_icon_font, anchor="mm")
        self.renderer.draw.text(camera_center, FontAwesomeIconConstants.CAMERA, fill=camera_color,
                                font=self.camera_icon_font, anchor="mm")


    def _capture_photo(self) -> bytes:
        try:
            return self.camera.capture_jpeg()
        finally:
            self.camera.stop_single_frame_mode()


    def _run(self):
        # If the user continues holding the button that brought them into this screen,
        # it must not take the photo: only a click after all buttons were released does.
        is_maybe_still_holding = True
        frame = None

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

                if frame is not None:
                    with self.renderer.lock:
                        self._render_frame(frame, GUIConstants.ACCENT_COLOR)
                        self.renderer.show_image()

                return self._capture_photo()

            frame = self.camera.read_single_frame_preview(self.PREVIEW_RESOLUTION)
            with self.renderer.lock:
                self._render_frame(frame, GUIConstants.BODY_FONT_COLOR)
                self.renderer.show_image()



@dataclass
class HandwrittenReadingScreen(BaseTopNavScreen):
    """
    Runs `read_function(screen)` in a background thread, which reports to the screen
    the photo (`show_photo`), each word read (`show_word`), what the reading does
    (`show_stage`, see handwriting.STAGES) and the progress (`report_progress`).

    Above the bar, two lines say what is being done: the hypothesis on the page (the
    document found, or "Try 2: another layout" when the reading goes back to try
    another one) and the step under way, with its counter ("Reading the words 5/12").
    The photo is shown while the words are searched, then the area of the words with
    the word being read outlined, next to its number and its first reading; a new
    layout gets its own area.

    The bar never goes back: a new hypothesis is said in words, not by the bar. Each
    attempt fills at most ATTEMPT_SHARE of what is left of the bar (the progress
    reported is mapped onto that part), so that a later attempt still has room to
    move; the bar is filled at the end. Its ribs move to the left on each redraw, at
    least every IDLE_INTERVAL seconds, so that the screen changes even during a long
    step that reports nothing (detection, ink of the lines).

    While the words are read, the words already read in the current pass stay
    outlined in a dimmer color (TRAIL), so that every word read shows even when several
    are read between two redraws, and the screen is redrawn when the word outlined
    changes if WORD_INTERVAL seconds have passed (a word is read every 0.15-0.4s on a
    Pi Zero). Otherwise it is redrawn at most every RENDER_INTERVAL seconds: a redraw
    costs the reading about 30ms on a Pi Zero (6ms of drawing, 24ms of conversion and
    transfer to the display, during which the reading thread does not run). The text of a word read on a layout that is not confirmed (see
    confirmed) is shown as UNCONFIRMED_WORD says: words read in stains or lines of
    another text while a wrong layout is tried are not shown as words.

    Returns the value returned by `read_function`; re-raises the exception it raised.
    """
    title: str = _mft("Reading words")
    show_back_button: bool = False
    read_function: Callable = None

    RENDER_INTERVAL = 0.7
    IDLE_INTERVAL = 1.0         # longest time without a redraw (moving ribs)
    WORD_INTERVAL = 0.35        # shortest time between two redraws when the word outlined changes
    TRAIL = True                # words already read in the current pass stay outlined, in TRAIL_COLOR
    TRAIL_COLOR = "#6E4A14"
    UNCONFIRMED_WORD = "show"   # text of a word on a layout not confirmed: "show", "dim" (gray) or "hide"
    TEXT_WIDTH = 100            # width left for the word read, right of the words area
    BAR_HEIGHT = 8
    RIB_PERIOD = 8              # ribs of the bar: one every RIB_PERIOD pixels, moved by RIB_STEP per redraw
    RIB_STEP = 3
    MIN_FONT_SIZE = 12          # a text too long for the screen (translations) is drawn smaller, down to this size
    ATTEMPT_SHARE = 0.85        # share of the rest of the bar an attempt can fill
    MAX_TEXTS = 100             # text images kept

    # Step under way (handwriting.STAGES)
    STAGE_TEXTS = {
        "load": _mft("Loading the reader"),
        "find": _mft("Finding the words"),
        "closer": _mft("Looking closer"),
        "lines": _mft("Sorting the lines"),
        "numbers": _mft("Reading line numbers"),
        "order": _mft("Checking the order"),
        "guard": _mft("Checking the words"),
        "spare": _mft("Looking for a fingerprint"),
        "rows": _mft("Checking extra rows"),
        "words": _mft("Reading the words"),
        "words2": _mft("Second reading"),
        "words3": _mft("Third reading"),
        "reread": _mft("Re-reading unclear words"),
        "fp_read": _mft("Reading the fingerprint"),
        "fp_check": _mft("Checking the fingerprint"),
    }
    # Hypothesis tried again on the page (handwriting.ATTEMPTS), with its number
    ATTEMPT_TEXTS = {
        "list": _mft("Try {}: as a list"),
        "extra": _mft("Try {}: another layout"),
        "crossed": _mft("Try {}: no crossed-out words"),
    }
    LIST_TEXT = _mft("{} lines found")
    CARD_TEXT = _mft("Seed card, {} words")


    def __post_init__(self):
        super().__post_init__()
        self.fraction = 0.0     # fraction of the bar shown
        self.reported = 0.0     # progress reported by the reading
        self.segment = (0.0, 0.0, self.ATTEMPT_SHARE)  # (progress, bar) at the start of the attempt, end of its part
        self.photo = None       # the photo, while the words are being read
        self.whole = None       # thumbnail of the photo
        self.sheets = {}        # id of a layout: (layout, thumbnail of its words area, its origin and scale on the photo)
        self.outlined = None    # (layout, index) of the word outlined, None for the whole photo
        self.image = None       # image shown above the bar, made from self.outlined when drawn
        self.image_of = None    # the self.outlined it was made from
        self.word = None        # (index, count, text) of the word outlined
        self.texts = {}         # (text, font, color): image of the text (see _text)
        self.fonts = {}         # (text, width): font of the text (see _fit)
        self.first_readings = {}  # (id of the layout, index): word of the first reading
        self.context = ""       # first line: the document found, or the hypothesis tried again
        self.step = ("find", None, None)  # second line: the step under way (STAGE_TEXTS) and its counter
        self.translated = {}    # text: its translation (gettext looks for files on each call)
        self.described = None   # (kind, count) of the document described on the first line
        self.attempt = None     # (name, number) of the hypothesis tried
        self.tries = 1
        self.try_numbers = {}   # name of an attempt: its number
        self.views = {}         # name of an attempt left: (outlined, word, context) shown then
        self.ribs = 0           # offset of the ribs of the bar
        self.version = 0        # incremented by the reading thread on each change
        self.trail = (None, [])  # ((id of the layout, pass), indices of the words outlined in the pass so far)
        self.trail_image = (None, None, 0)  # (key of the trail, thumbnail with its words outlined, how many)
        self.confirmations = {}  # id of a layout: confirmed(layout)

        self.bar_top = self.canvas_height - GUIConstants.EDGE_PADDING - self.BAR_HEIGHT
        line_height = GUIConstants.LABEL_FONT_SIZE + GUIConstants.LIST_ITEM_PADDING
        self.action_y = self.bar_top - GUIConstants.LIST_ITEM_PADDING - line_height
        self.context_y = self.action_y - line_height
        self.area = (GUIConstants.EDGE_PADDING, self.top_nav.height, self.canvas_width - GUIConstants.EDGE_PADDING,
                     self.context_y - GUIConstants.LIST_ITEM_PADDING)
        self.label_font = Fonts.get_font(GUIConstants.get_body_font_name(), GUIConstants.LABEL_FONT_SIZE)
        self.word_font = Fonts.get_font(GUIConstants.FIXED_WIDTH_EMPHASIS_FONT_NAME, GUIConstants.get_top_nav_title_font_size())

        self.strips = self._ribbed_strips()
        self.drawn = (None, None)   # (image, word) and lines last drawn: _render_reading draws only what changed

        self.reading_thread = HandwrittenReadingScreen.ReadingThread(read_function=self.read_function, screen=self)
        self.threads.append(self.reading_thread)


    @staticmethod
    def thumbnail(image: Image.Image, width: int, height: int) -> Image.Image:
        """ `image` reduced to fit in width x height, in RGB """
        factor = max(1, -(-image.height // height), -(-image.width // width))
        return image.reduce(factor).convert("RGB")


    def report_progress(self, fraction: float):
        """
        Called from the reading thread; the screen redraws on its next loop. The
        progress is mapped onto the part of the bar of the current attempt.
        """
        self.reported = min(max(fraction, 0.0), 1.0)
        start, bar, end = self.segment
        share = (self.reported - start) / (1 - start) if start < 1 else 1.0
        self.fraction = max(self.fraction, bar + (end - bar) * share)
        self.version += 1


    def _new_segment(self):
        """ A new attempt: it fills ATTEMPT_SHARE of what is left of the bar from here """
        self.segment = (self.reported, self.fraction, self.fraction + (1 - self.fraction) * self.ATTEMPT_SHARE)


    def show_photo(self, photo: Image.Image):
        """ Called from the reading thread with the upright photo, before the reading """
        left, top, right, bottom = self.area
        self.photo = photo
        self.whole = self.thumbnail(photo, right - left, bottom - top)
        self.version += 1


    def _sheet(self, layout) -> tuple:
        """ (layout, thumbnail of its words area, origin and scale on the photo), made once per layout """
        sheet = self.sheets.get(id(layout))
        if sheet is None or sheet[0] is not layout:
            # Thumbnail of the area of the words, with a margin
            corners = [p for i in range(layout.count) for p in layout.outline(i)]
            x0, y0 = min(x for x, _ in corners), min(y for _, y in corners)
            x1, y1 = max(x for x, _ in corners), max(y for _, y in corners)
            margin = 0.05 * max(x1 - x0, y1 - y0)
            box = tuple(int(v) for v in (max(0, x0 - margin), max(0, y0 - margin),
                                         min(self.photo.width, x1 + margin), min(self.photo.height, y1 + margin)))
            left, top, right, bottom = self.area
            thumbnail = self.thumbnail(self.photo.crop(box), right - left - self.TEXT_WIDTH, bottom - top)
            sheet = self.sheets[id(layout)] = (layout, thumbnail, box[:2], thumbnail.width / (box[2] - box[0]))
        return sheet


    def _outline(self, layout, k: int, trail_key=None, passed: list = ()) -> Image.Image:
        """
        The area of the words of `layout`, word `k` outlined, and with TRAIL the words
        `passed` in the pass `trail_key` outlined in TRAIL_COLOR (drawn once each on a
        thumbnail kept for the pass).
        """
        _, thumbnail, (x0, y0), scale = self._sheet(layout)

        def corners(i):
            return [((x - x0) * scale, (y - y0) * scale) for x, y in layout.outline(i)]

        if self.TRAIL and passed and trail_key is not None and trail_key[0] == id(layout):
            key, base, count = self.trail_image
            if key != trail_key or base is None:
                base, count = thumbnail.copy(), 0
            draw = ImageDraw.Draw(base)
            for i in passed[count:]:
                draw.polygon(corners(i), outline=self.TRAIL_COLOR, width=1)
            self.trail_image = (trail_key, base, len(passed))
            image = base.copy()
        else:
            image = thumbnail.copy()
        ImageDraw.Draw(image).polygon(corners(k), outline=GUIConstants.ACCENT_COLOR, width=2)
        return image


    def current_image(self) -> Image.Image:
        """
        The image to show, made when drawn (in the screen's thread) and only when the
        word outlined changed: the reading thread only says which one it is.
        """
        outlined = self.outlined
        trail_key, trail = self.trail
        state = (outlined, trail_key, len(trail))
        of = self.image_of
        if self.image is None or of is None or of[0] is not outlined or of[1:] != state[1:]:
            self.image = self.whole if outlined is None or self.photo is None \
                else self._outline(*outlined, trail_key, trail[:state[2]])
            self.image_of = state
        return self.image


    def _pass(self, layout, k: int, name: str):
        """ Notes word `k` of `layout` as read in pass `name` (reading thread): the trail of the pass """
        key = (id(layout), name)
        trail_key, trail = self.trail
        if key != trail_key:
            self.trail = (key, [k])
        else:
            trail.append(k)


    def confirmed(self, layout) -> bool:
        """
        True if the layout is not a mere hypothesis, for the text of its words: a card,
        or a list with a number linked to every line, or a list without any number
        whose text is complete and whose numbers read do not contradict its order.
        A list with numbers on some of its lines only (lines taken from stains or from
        another text) is not. For display only.
        """
        key = id(layout)
        if key not in self.confirmations:
            linked = getattr(layout, "numbers_linked", None)
            if getattr(layout, "kind", "list") != "list" or not isinstance(linked, int):
                found = True
            elif getattr(layout, "numbers_incoherent", False) is True:
                found = False
            else:
                found = linked >= layout.count or (linked == 0 and getattr(layout, "complete", True) is True)
            self.confirmations[key] = found
        return self.confirmations[key]


    def show_word(self, layout, k: int, reading):
        """ Called from the reading thread after the first reading of each word """
        self.outlined = (layout, k)
        text = reading.candidates[0][0]
        self.first_readings[(id(layout), k)] = text
        self.word = (k, layout.count, text)
        self._pass(layout, k, "words")
        self.version += 1


    def show_stage(self, name: str, done: int = None, total: int = None, layout=None, k: int = None):
        """
        Called from the reading thread with what the reading does (handwriting.STAGES):
        an attempt ("attempt:<name>") changes the first line, a step the second one; the
        word `k` of `layout` is outlined.
        """
        if name.startswith("attempt:"):
            attempt = name[len("attempt:"):]
            if self.reported > 0:
                self._new_segment()
            # what the attempt left showed is kept; one taken up again shows it again, a new one starts from the photo
            self.views[self.attempt[0] if self.attempt else "first"] = (self.outlined, self.word, self.context)
            if attempt in self.views:
                self.outlined, self.word, self.context = self.views[attempt]
            else:
                self.outlined, self.word = None, None
            self.trail = (None, [])
            if attempt in self.ATTEMPT_TEXTS:
                # numbered once, at its first try (a card read as a list is tried again with each network)
                if attempt not in self.try_numbers:
                    self.tries += 1
                    self.try_numbers[attempt] = self.tries
                self.attempt = (attempt, self.try_numbers[attempt])
                self.context = _(self.ATTEMPT_TEXTS[attempt]).format(self.attempt[1])
                self.described = None
            else:
                self.attempt = None
                if self.outlined is not None:
                    self._describe(self.outlined[0])
            self.version += 1
            return

        # only noted here: called often by the reading thread; the texts are made when drawn
        if name in self.STAGE_TEXTS:
            self.step = (name, done, total)
        if layout is not None:
            if self.attempt is None:
                self._describe(layout)
            if k is not None and name != "words":
                # second readings and re-readings: the word outlined, with its first reading (show_word outlines
                # each word of the first reading once read)
                self.outlined = (layout, k)
                self.word = (k, layout.count, self.first_readings.get((id(layout), k), "…"))
                self._pass(layout, k, name)
        self.version += 1


    def _describe(self, layout):
        """ The first line for the first attempt: the document found """
        key = (layout.kind, layout.count)
        if key != self.described:
            self.described = key
            self.context = _(self.LIST_TEXT if layout.kind == "list" else self.CARD_TEXT).format(layout.count)


    @property
    def action(self) -> str:
        """ The step under way, translated """
        text = self.STAGE_TEXTS[self.step[0]]
        if text not in self.translated:
            self.translated[text] = _(text)
        return self.translated[text]


    @property
    def counter(self) -> str:
        """ Its counter: "done/total", "done" or "" """
        _name, done, total = self.step
        return "" if done is None else f"{done}/{total}" if total else str(done)


    def _text(self, text: str, font, color: str) -> Image.Image:
        """
        Image of `text` on the background, its origin at the left of its first line
        (anchor "la"), kept for the next redraws (counters and steps come back often).
        """
        key = (text, id(font), color)
        image = self.texts.get(key)
        if image is None:
            if len(self.texts) >= self.MAX_TEXTS:
                self.texts.clear()
                self.fonts.clear()
            width = max(1, int(font.getlength(text)) + 2)
            height = max(1, font.getbbox(text, anchor="la")[3] + 1)
            image = Image.new("RGB", (width, height), GUIConstants.BACKGROUND_COLOR)
            ImageDraw.Draw(image).text((0, 0), text, fill=color, font=font, anchor="la")
            self.texts[key] = image
        return image


    def _fit(self, text: str, width: int):
        """ The label font, smaller if needed for `text` to fit in `width` (kept with the text images) """
        key = (text, width)
        if key in self.fonts:
            return self.fonts[key]
        size = GUIConstants.LABEL_FONT_SIZE
        font = self.label_font
        while size > self.MIN_FONT_SIZE and font.getlength(text) > width:
            size -= 1
            font = Fonts.get_font(GUIConstants.get_body_font_name(), size)
        self.fonts[key] = font
        return font


    def _render(self):
        super()._render()
        self.drawn = (None, None)
        self._render_reading()


    def _render_reading(self):
        """
        Draws what changed since the last call (the canvas keeps the rest): the image and
        the word, the two lines, and always the bar. Text costs most of a redraw on a Pi
        Zero (about 4ms per line), so that a redraw that only moves the ribs is cheap.
        """
        draw = self.renderer.draw
        left, top, right, bottom = self.area
        word = self.word
        image = self.current_image()
        lines = (self.context, self.action, self.counter)
        drawn_image, drawn_lines = self.drawn

        if drawn_image is None or drawn_image[0] is not image or drawn_image[1] != word:
            draw.rectangle((0, self.top_nav.height, self.canvas_width, self.context_y - 1), fill=GUIConstants.BACKGROUND_COLOR)
            if image is not None:
                x = left if word else int((left + right - image.width) / 2)
                self.renderer.canvas.paste(image, (x, top + int((bottom - top - image.height) / 2)))

            if word is not None and image is not None:
                # Number and text of the word, right of the words area (text images kept: words come back in each pass)
                k, count, text = word
                x, y = int((left + image.width + right) / 2), int((top + bottom) / 2)
                number = self._text(f"{k + 1}/{count}", self.label_font, GUIConstants.LABEL_FONT_COLOR)
                self.renderer.canvas.paste(number, (x - number.width // 2, y - GUIConstants.LIST_ITEM_PADDING - number.height))
                shown = self.UNCONFIRMED_WORD
                if self.outlined is not None and self.confirmed(self.outlined[0]):
                    shown = "show"
                if shown != "hide":
                    color = GUIConstants.BODY_FONT_COLOR if shown == "show" else GUIConstants.LABEL_FONT_COLOR
                    word_image = self._text(text, self.word_font, color)
                    self.renderer.canvas.paste(word_image, (x - word_image.width // 2, y + GUIConstants.LIST_ITEM_PADDING))

        context, action, counter = lines
        canvas = self.renderer.canvas
        if drawn_lines is None or context != drawn_lines[0]:
            # What is being done: the hypothesis...
            draw.rectangle((0, self.context_y, self.canvas_width, self.action_y - 1), fill=GUIConstants.BACKGROUND_COLOR)
            if context:
                text = self._text(context, self._fit(context, right - left), GUIConstants.LABEL_FONT_COLOR)
                canvas.paste(text, (int((self.canvas_width - text.width) / 2), self.context_y))
        if drawn_lines is None or (action, counter) != drawn_lines[1:]:
            # ... then the step and its counter (text images kept: a counter only changes a small image)
            draw.rectangle((0, self.action_y, self.canvas_width, self.bar_top - 1), fill=GUIConstants.BACKGROUND_COLOR)
            if action:
                number = self._text(counter, self.label_font, GUIConstants.ACCENT_COLOR) if counter else None
                gap = number.width + GUIConstants.COMPONENT_PADDING if number else 0
                text = self._text(action, self._fit(action, right - left - gap), GUIConstants.BODY_FONT_COLOR)
                x = int((self.canvas_width - text.width - gap) / 2)
                canvas.paste(text, (x, self.action_y))
                if number:
                    canvas.paste(number, (x + text.width + GUIConstants.COMPONENT_PADDING, self.action_y))
        self.drawn = ((image, word), lines)

        # Progress bar: the parts of the two ribbed strips, moved by self.ribs, then its outline
        filled = int((right - left) * self.fraction)
        canvas = self.renderer.canvas
        if filled > 0:
            canvas.paste(self.strips[0].crop((self.ribs, 0, self.ribs + filled, self.BAR_HEIGHT + 1)), (left, self.bar_top))
        if filled < right - left:
            canvas.paste(self.strips[1].crop((self.ribs + filled, 0, self.ribs + right - left, self.BAR_HEIGHT + 1)),
                         (left + filled, self.bar_top))
        draw.rectangle((left, self.bar_top, right, self.bar_top + self.BAR_HEIGHT), outline=GUIConstants.INACTIVE_COLOR)


    RIB_FILLED_COLOR = "#C77A00"


    def _ribbed_strips(self) -> tuple[Image.Image, Image.Image]:
        """
        The filled and the empty bar, RIB_PERIOD pixels wider than the bar, with slanted
        ribs every RIB_PERIOD pixels: a part starting at an offset moves them.
        """
        width = self.area[2] - self.area[0] + 2 * self.RIB_PERIOD + self.BAR_HEIGHT
        strips = []
        for background, rib in ((GUIConstants.ACCENT_COLOR, self.RIB_FILLED_COLOR),
                                (GUIConstants.BACKGROUND_COLOR, GUIConstants.INACTIVE_COLOR)):
            strip = Image.new("RGB", (width, self.BAR_HEIGHT + 1), background)
            draw = ImageDraw.Draw(strip)
            for x in range(-self.BAR_HEIGHT, width, self.RIB_PERIOD):
                draw.line((x, self.BAR_HEIGHT, x + self.BAR_HEIGHT, 0), fill=rib, width=2)
            strips.append(strip)
        return tuple(strips)


    def render_due(self, now: float, last_render: float, displayed: int, shown_word) -> bool:
        """
        True if the screen is to be redrawn at `now`: IDLE_INTERVAL seconds after the
        last redraw, or on a change RENDER_INTERVAL seconds after it, or WORD_INTERVAL
        seconds after it if the word outlined is not `shown_word`, the one it showed.
        """
        elapsed = now - last_render
        if elapsed >= self.IDLE_INTERVAL:
            return True
        if displayed == self.version:
            return False
        return elapsed >= self.RENDER_INTERVAL or elapsed >= self.WORD_INTERVAL and self.outlined is not shown_word


    def _run(self):
        displayed, last_render, shown_word = None, 0.0, None
        while self.reading_thread.is_alive():
            now = time.time()
            if self.render_due(now, last_render, displayed, shown_word):
                displayed, last_render, shown_word = self.version, now, self.outlined
                self.ribs = (self.ribs + self.RIB_STEP) % self.RIB_PERIOD
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
    Full screen notice to read before going on: a colored headline, then `gauge`, a
    (level, label) pair drawn as GAUGE_LEVELS segments of which `level` are filled, with
    its label below, then `paragraphs`, (icon name or None, text) pairs, the icon being
    drawn before its paragraph. KEY_DOWN
    and KEY_UP turn the pages of the text. The button below it can only be selected
    from the last page, so that a click made by mistake does not skip the notice.

    Returns 0 when the button is clicked.
    """
    show_back_button: bool = False
    status_headline: str = None
    status_color: str = GUIConstants.WARNING_COLOR
    gauge: tuple = None
    paragraphs: list = None
    icon_color: str = GUIConstants.WARNING_COLOR
    button_data: list = field(default_factory=lambda: [ButtonOption("I understand")])

    ARROW_HALF_WIDTH = 10

    # The gauge: its last segment stands for a seed verified by its fingerprint
    GAUGE_LEVELS = 4
    GAUGE_SEGMENT_WIDTH = 34
    GAUGE_SEGMENT_HEIGHT = 10
    GAUGE_SEGMENT_GAP = 4


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

        # (x, y, text, font, color, anchor) of everything to draw, and the gauge segments (box, filled)
        items = []
        segments = []
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
            if is_headline and self.gauge:
                level, label = self.gauge
                width = self.GAUGE_LEVELS * self.GAUGE_SEGMENT_WIDTH + (self.GAUGE_LEVELS - 1) * self.GAUGE_SEGMENT_GAP
                x0 = int((self.canvas_width - width) / 2)
                for i in range(self.GAUGE_LEVELS):
                    x1 = x0 + i * (self.GAUGE_SEGMENT_WIDTH + self.GAUGE_SEGMENT_GAP)
                    segments.append(((x1, y, x1 + self.GAUGE_SEGMENT_WIDTH - 1, y + self.GAUGE_SEGMENT_HEIGHT - 1), i < level))
                top = y
                y += self.GAUGE_SEGMENT_HEIGHT + int(GUIConstants.COMPONENT_PADDING / 2)
                items.append((int(self.canvas_width / 2), y, label, body_font, GUIConstants.BODY_FONT_COLOR, "ma"))
                y += line_height
                lines.append((top, y, 1))
                y += GUIConstants.COMPONENT_PADDING

        image = Image.new("RGB", (self.canvas_width, y), GUIConstants.BACKGROUND_COLOR)
        draw = ImageDraw.Draw(image)
        for x, y, text, font, color, anchor in items:
            draw.text((x, y), text, font=font, fill=color, anchor=anchor)
        for box, filled in segments:
            if filled:
                draw.rectangle(box, fill=self.status_color)
            else:
                draw.rectangle(box, outline=GUIConstants.BUTTON_BACKGROUND_COLOR, width=2)
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
