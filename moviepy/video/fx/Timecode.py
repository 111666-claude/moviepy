from dataclasses import dataclass, field
from functools import lru_cache
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.tools import compute_position
from moviepy.video.tools.timecode import frame_to_timecode, timecode_to_frame


@dataclass
class Timecode(Effect):
    """Burns a frame-accurate ``HH:MM:SS:FF`` timecode onto every frame.

    The timecode is computed from the clip's frame index and is therefore
    locked to frame boundaries: it does not drift with sub-frame rendering
    times. Applying the effect before :meth:`subclipped` keeps displaying the
    timecode of the source material, because subclipping shifts the frame
    lookup back to the original timeline.

    Parameters
    ----------

    start_timecode : str or int, optional
      Timecode of the first frame of the clip. Either a ``HH:MM:SS:FF``
      string (a ``;`` separator before the frames denotes a drop-frame
      timecode) or an absolute frame number. Defaults to ``"00:00:00:00"``.

    fps : int or float, optional
      Frame rate used to build the timecode. When ``None`` (default), the
      frame rate of the clip is used. Integer rates produce non drop-frame
      timecodes; ``29.97`` and ``59.94`` can be combined with
      ``drop_frame=True``.

    drop_frame : bool, optional
      When ``True``, render a SMPTE drop-frame timecode. Only valid for
      29.97 and 59.94 fps. When ``None`` (default), the mode is inferred
      from ``start_timecode`` when it is a string, and non drop-frame
      otherwise.

    position : tuple or str, optional
      Position of the timecode on the frame. Accepts the same values as
      :meth:`VideoClip.with_position`, for instance ``("left", "top")``
      (default), ``("right", "bottom")`` or an exact ``(x, y)`` pixel
      position.

    margin : int, optional
      Padding in pixels added around the timecode when it is anchored to an
      edge or a corner. Ignored for exact pixel positions. Defaults to 8.

    font : str, optional
      Path to a TrueType/OpenType font file. When ``None`` (default), or when
      the font cannot be loaded, the Pillow default font is used and a
      warning is emitted.

    font_size : int, optional
      Font size in pixels. Defaults to 24.

    color : tuple or str, optional
      Color of the text, as an RGB tuple or a Pillow color name. Defaults to
      white ``(255, 255, 255)``.

    bg_color : tuple or str, optional
      Color of the opaque box drawn behind the text. When ``None`` (default),
      no box is drawn and the timecode is composited directly over the frame.

    stroke_color : tuple or str, optional
      Color of the text outline. Defaults to black.

    stroke_width : int, optional
      Width in pixels of the text outline. Set to 0 to disable the outline.
      Defaults to 1.
    """

    start_timecode: object = "00:00:00:00"
    fps: object = None
    drop_frame: object = None
    position: object = ("left", "top")
    margin: int = 8
    font: str = None
    font_size: int = 24
    color: object = (255, 255, 255)
    bg_color: object = None
    stroke_color: object = (0, 0, 0)
    stroke_width: int = 1
    # Resolved at apply() time, not part of the public constructor surface.
    _start_frame: int = field(default=None, init=False, repr=False)
    _resolved_fps: object = field(default=None, init=False, repr=False)
    _drop_frame: bool = field(default=False, init=False, repr=False)
    _pil_font: object = field(default=None, init=False, repr=False)

    def _resolve_parameters(self, clip):
        """Validate the parameters against the target clip."""
        if self.fps is None:
            if getattr(clip, "fps", None) is None:
                raise ValueError(
                    "The clip has no fps attribute and no fps was provided to "
                    "the Timecode effect."
                )
            self._resolved_fps = clip.fps
        else:
            self._resolved_fps = self.fps

        if isinstance(self.start_timecode, str):
            self._start_frame = timecode_to_frame(
                self.start_timecode,
                self._resolved_fps,
                drop_frame=self.drop_frame,
            )
            if self.drop_frame is None:
                self._drop_frame = ";" in self.start_timecode
            else:
                self._drop_frame = self.drop_frame
        elif isinstance(self.start_timecode, int) and not isinstance(
            self.start_timecode, bool
        ):
            if self.start_timecode < 0:
                raise ValueError(
                    "start_timecode frame number must be non-negative, got "
                    "{}".format(self.start_timecode)
                )
            self._start_frame = self.start_timecode
            self._drop_frame = bool(self.drop_frame)
        else:
            raise TypeError(
                "start_timecode must be a HH:MM:SS:FF string or an integer "
                "frame number, got {} ({})".format(
                    self.start_timecode, type(self.start_timecode).__name__
                )
            )

        # Validates the fps / drop-frame combination up front.
        frame_to_timecode(self._start_frame, self._resolved_fps, self._drop_frame)

        self._pil_font = self._load_font()

    def _load_font(self):
        """Load the requested font, falling back on the Pillow default."""
        if self.font is None:
            return self._default_font()
        try:
            return ImageFont.truetype(self.font, self.font_size)
        except (OSError, IOError) as exc:
            warnings.warn(
                "Could not load font {!r} ({}), falling back on the Pillow "
                "default font.".format(self.font, exc),
                RuntimeWarning,
                stacklevel=3,
            )
            return self._default_font()

    def _default_font(self):
        try:
            # Pillow >= 10.1 accepts a size argument.
            return ImageFont.load_default(self.font_size)
        except TypeError:
            return ImageFont.load_default()

    def _render_text(self, text):
        """Render one timecode label as an RGBA Pillow image."""
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        x0, y0, x1, y1 = draw.textbbox(
            (0, 0),
            text,
            font=self._pil_font,
            stroke_width=self.stroke_width,
        )
        text_w, text_h = x1 - x0, y1 - y0

        img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text(
            (-x0, -y0),
            text,
            font=self._pil_font,
            fill=self.color,
            stroke_fill=self.stroke_color,
            stroke_width=self.stroke_width,
        )
        return img

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        self._resolve_parameters(clip)

        # Only a handful of distinct labels exist (one per frame of the
        # rate's second), so caching rendered images keeps per-frame work low.
        @lru_cache(maxsize=128)
        def text_image(text):
            return self._render_text(text)

        drop_frame = self._drop_frame
        resolved_fps = self._resolved_fps
        start_frame = self._start_frame
        position = self.position
        margin = self.margin

        def filter(get_frame, t):
            frame = get_frame(t)
            if frame.ndim != 3 or frame.shape[2] not in (3, 4):
                raise TypeError(
                    "The Timecode effect can only be applied to RGB(A) clips, "
                    "got frame shape {}".format(frame.shape)
                )

            # Round to the nearest frame so sub-frame render times cannot
            # drift the label away from the actual frame boundary.
            frame_index = start_frame + int(round(float(t) * resolved_fps))
            label = frame_to_timecode(frame_index, resolved_fps, drop_frame)

            overlay = text_image(label)
            overlay_w, overlay_h = overlay.size

            frame_h, frame_w = frame.shape[:2]
            if isinstance(position, str) or any(
                isinstance(component, str) for component in tuple(position)
            ):
                x, y = compute_position(overlay.size, (frame_w, frame_h), position)
                if isinstance(position, str):
                    anchors = {
                        "center": ("center", "center"),
                        "left": ("left", "center"),
                        "right": ("right", "center"),
                        "top": ("center", "top"),
                        "bottom": ("center", "bottom"),
                    }[position]
                else:
                    anchors = tuple(position)
                if anchors[0] == "left":
                    x += margin
                elif anchors[0] == "right":
                    x -= margin
                if anchors[1] == "top":
                    y += margin
                elif anchors[1] == "bottom":
                    y -= margin
            else:
                x, y = int(position[0]), int(position[1])

            x0 = max(x, 0)
            y0 = max(y, 0)
            x1 = min(x + overlay_w, frame_w)
            y1 = min(y + overlay_h, frame_h)
            if x1 <= x0 or y1 <= y0:
                return frame

            base = Image.fromarray(frame.astype(np.uint8)).convert("RGBA")
            if self.bg_color is not None:
                bg = self.bg_color
                if len(bg) == 3:
                    bg = tuple(bg) + (255,)
                box = Image.new("RGBA", (x1 - x0, y1 - y0), bg)
                base.alpha_composite(box, (x0, y0))

            # Paste the (possibly partially off-screen) overlay on a
            # full-frame transparent layer so clipping keeps it aligned.
            layer = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
            layer.paste(overlay, (x, y), overlay)
            base.alpha_composite(layer)

            result = np.array(base)
            return result if frame.shape[2] == 4 else result[:, :, :3]

        return clip.transform(filter)
