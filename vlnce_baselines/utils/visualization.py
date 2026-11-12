import cv2
import numpy as np
import os
import re
from habitat import logger
from collections.abc import Sequence
from vlnce_baselines.utils.constant import legend_color_palette
from habitat.core.simulator import Observations
import time
from PIL import Image
import torch

def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


def init_vis_image():
    vis_image = np.ones((755, 1165, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Goal: "
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Predicted Semantic Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    color = [100, 100, 100]
    vis_image[49, 15:655] = color
    vis_image[49, 670:1150] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 655] = color
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    vis_image[530, 15:655] = color
    vis_image[530, 670:1150] = color
    
    vis_image = add_class(vis_image, 0, "out of map", legend_color_palette)
    vis_image = add_class(vis_image, 1, "obstacle", legend_color_palette)
    vis_image = add_class(vis_image, 2, "free space", legend_color_palette)
    vis_image = add_class(vis_image, 3, "agent trajecy", legend_color_palette)
    vis_image = add_class(vis_image, 4, "waypoint", legend_color_palette)

    return vis_image


def add_class(vis_image, id, name, color_palette):
    text = f"{id}:{name}"
    font_color = (0, 0, 0)
    class_color = list(color_palette[id])
    class_color.reverse() # BGR -> RGB
    fontScale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = 20, 50
    start_x, start_y = 25, 540 # x is horizontal and point to right, y is vertical and point to down
    gap_x, gap_y = 280, 30
    
    x1 = start_x + (id % 4) * gap_x
    y1 = start_y + (id // 4) * gap_y
    x2 = x1 + w
    y2 = y1 + h
    text_x = x2 + 10
    text_y = (y1 + y2) // 2 + 5
    
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), class_color, thickness=cv2.FILLED)
    vis_image = cv2.putText(vis_image, text, (text_x, text_y),
                            font, fontScale, font_color, thickness, cv2.LINE_AA)
    
    return vis_image


def add_text(image: np.ndarray, text: str, position: Sequence):
    textsize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
    x, y = position
    x += (480 - textsize[0]) // 2
    image = cv2.putText(image, text, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, color=0, thickness=1, lineType=cv2.LINE_AA)
    
    return image

ACTION_TO_TEXT = ['stop', 'forward', 'turn left', 'turn right']
ACTION_TO_DISPLAY = {
    0: 'STOP',
    1: 'FORWARD',
    2: 'TURN LEFT',
    3: 'TURN RIGHT',
    'stop': 'STOP',
    'forward': 'FORWARD',
    'turn left': 'TURN LEFT',
    'turn right': 'TURN RIGHT',
}

# ---------------------------------------------------------------------------
# Combined-frame layout (fixed 1920x1080 so ffmpeg gets a constant frame size)
#
#   title bar   y [0, 90)      run id / episode / step + step-budget bar
#   main row    y [90, 780)    RGB 920 | topdown map 690 | harness side panel 310
#   bottom      y [780, 1080)  instruction + TODO / obligation list, full width
# ---------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 1920, 1080
TITLE_H = 90
MAIN_Y0, MAIN_Y1 = 90, 780
STRIP_Y0 = MAIN_Y1
RGB_W, MAP_W, SIDE_W = 920, 690, 310
PANEL_HDR_H = 26
MAP_LEGEND_H = 26
# mapping.py's marker sizes were tuned for a full-frame 1024px map; the panel
# autocrops to the explored region and rescales, which blows them up.
MAP_MARKER_SCALE = 0.5

# Every colour below is **RGB**, because the finished canvas is handed straight
# to PIL's Image.fromarray. Do not paste BGR tuples in here: the older version of
# this file mixed both conventions in one function and rendered red as blue.
RGB = {
    'bg': (255, 255, 255),
    'panel': (247, 248, 250),
    'title_bg': (26, 30, 38),
    'title_fg': (255, 255, 255),
    'title_dim': (150, 158, 172),
    'hdr_bg': (236, 239, 244),
    'text': (24, 24, 28),
    'muted': (112, 118, 130),
    'sep': (203, 208, 216),
    'accent': (0, 92, 200),
    'ok': (0, 150, 62),
    'warn': (214, 132, 0),
    'bad': (200, 40, 40),
    'hdr_rgb': (0, 108, 52),
    'hdr_map': (150, 32, 32),
    'hdr_side': (92, 44, 150),
    'todo_done': (132, 138, 150),
    'bar_bg': (58, 64, 76),
    'bar_fg': (86, 190, 255),
    # markers mirrored from vlnce_baselines/map/mapping.py (BGR there, RGB here)
    'm_agent': (255, 0, 0),
    'm_goal': (0, 200, 0),
    'm_waypoint': (0, 0, 255),
    'm_path': (255, 0, 0),
    'm_plan': (0, 200, 0),
    'm_obstacle': (0, 0, 0),
    'm_explored': (220, 220, 220),
    'bbox_nav': (0, 220, 70),
    'bbox_stop': (235, 55, 55),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


# cv2's Hershey fonts are ASCII-only -- anything else renders as "??". R2R/RxR
# instructions routinely carry typographic quotes and dashes, so map the common
# ones and drop the rest rather than printing noise.
_ASCII_MAP = {0x2018: "'", 0x2019: "'", 0x201a: "'", 0x201c: '"', 0x201d: '"',
              0x2013: '-', 0x2014: '-', 0x2212: '-', 0x2026: '...', 0x00b7: '|',
              0x2022: '*', 0x00a0: ' ', 0x00b0: ' deg'}


def _ascii(text):
    s = str(text).translate(_ASCII_MAP)
    return s if s.isascii() else ''.join(c if c.isascii() else '?' for c in s)


def wrap_text(text, max_width, font=FONT, font_scale=0.5, thickness=1):
    """Greedy word wrap against a pixel width. Falls back to per-character
    splitting for single words that are wider than the box."""
    if not text:
        return [text]

    text = _ascii(text)
    lines = []
    current_line = ""
    for word in str(text).split(' '):
        test_line = current_line + (" " if current_line else "") + word
        if cv2.getTextSize(test_line, font, font_scale, thickness)[0][0] <= max_width:
            current_line = test_line
            continue
        if current_line:
            lines.append(current_line)
            current_line = ""
        # the word alone may still overflow -> hard split it
        while cv2.getTextSize(word, font, font_scale, thickness)[0][0] > max_width and len(word) > 1:
            cut = len(word)
            while cut > 1 and cv2.getTextSize(word[:cut], font, font_scale, thickness)[0][0] > max_width:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        current_line = word

    if current_line:
        lines.append(current_line)
    return lines if lines else [text]


def _put(img, text, org, scale=0.5, color=RGB['text'], thickness=1):
    cv2.putText(img, _ascii(text), (int(org[0]), int(org[1])), FONT, scale, color,
                thickness, lineType=cv2.LINE_AA)


def _text_w(text, scale=0.5, thickness=1):
    return cv2.getTextSize(_ascii(text), FONT, scale, thickness)[0][0]


def _panel_header(canvas, x0, y0, w, title, color, right=None):
    """Grey caption bar at the top of a panel."""
    canvas[y0:y0 + PANEL_HDR_H, x0:x0 + w] = RGB['hdr_bg']
    _put(canvas, title, (x0 + 10, y0 + 18), 0.52, color, 2)
    if right:
        _put(canvas, right, (x0 + w - 8 - _text_w(right, 0.45), y0 + 18), 0.45, RGB['muted'], 1)
    canvas[y0 + PANEL_HDR_H - 1, x0:x0 + w] = RGB['sep']


def _progress_bar(canvas, x, y, w, h, frac, fg=RGB['bar_fg'], bg=RGB['bar_bg']):
    frac = 0.0 if frac is None else float(np.clip(frac, 0.0, 1.0))
    cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), bg, -1)
    filled = int(round(w * frac))
    if filled > 0:
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + filled), int(y + h)), fg, -1)


def _fit_into(img, box_w, box_h):
    """Aspect-preserving resize of img into a box. Returns (resized, off_x, off_y, scale)."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img, 0, 0, 1.0
    scale = min(box_w / float(w), box_h / float(h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return resized, (box_w - new_w) // 2, (box_h - new_h) // 2, scale


def _autocrop_map(vlm_map, margin=0.08):
    """Crop the topdown map to its non-background content, then square it up.

    create_vlm_map_from_state renders the whole local_map, so early in an episode
    the explored region is a speck floating on white. Cropping to the content
    bbox (kept square so the resize below cannot stretch the grid) makes step 0
    legible without touching the map's own coordinate transforms.
    """
    try:
        h, w = vlm_map.shape[:2]
        mask = np.any(vlm_map < 250, axis=2) if vlm_map.ndim == 3 else (vlm_map < 250)
        ys, xs = np.nonzero(mask)
        if ys.size < 16:
            return vlm_map
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        side = max(y1 - y0, x1 - x0)
        side = int(round(side * (1.0 + 2 * margin)))
        side = max(48, min(side, max(h, w)))
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        y0 = int(np.clip(cy - side // 2, 0, max(0, h - 1)))
        x0 = int(np.clip(cx - side // 2, 0, max(0, w - 1)))
        y1 = min(h, y0 + side)
        x1 = min(w, x0 + side)
        y0 = max(0, y1 - side)
        x0 = max(0, x1 - side)
        crop = vlm_map[y0:y1, x0:x1]
        return crop if crop.size else vlm_map
    except Exception as e:
        logger.warning(f'map autocrop failed, using full map: {e}')
        return vlm_map


# The labels track what mapping.py actually draws, not what its arguments are
# called: create_vlm_map_from_state's `goal` parameter receives
# nav.target_map_x/y -- the waypoint the planner is currently steering to, NOT the
# episode's ground-truth goal (which the agent is never shown). Calling it
# "goal" in a figure would claim the agent can see the answer.
_LEGEND = [
    ('agent', 'arrow', 'm_agent'),
    ('nav target', 'cross', 'm_goal'),
    ('past targets', 'dot', 'm_waypoint'),
    ('path', 'line', 'm_path'),
    ('NavDP plan', 'line', 'm_plan'),
    ('obstacle', 'box', 'm_obstacle'),
    ('explored', 'box', 'm_explored'),
]


def _draw_map_legend(canvas, x0, y0, w):
    """One-row key for the topdown map markers. The map itself is drawn by
    create_vlm_map_from_state, which never labels anything."""
    canvas[y0:y0 + MAP_LEGEND_H, x0:x0 + w] = RGB['hdr_bg']
    scale = 0.4
    items = []
    for label, kind, key in _LEGEND:
        items.append((label, kind, RGB[key], 18 + 5 + _text_w(label, scale)))
    total = sum(it[3] for it in items)
    gap = max(6, int((w - 16 - total) / max(1, len(items) - 1)))
    x = x0 + 8
    cy = y0 + MAP_LEGEND_H // 2
    for label, kind, color, width in items:
        if kind == 'arrow':
            pts = np.array([(x, cy - 6), (x + 14, cy), (x, cy + 6), (x + 4, cy)], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], color)
        elif kind == 'cross':
            cv2.drawMarker(canvas, (x + 7, cy), color, cv2.MARKER_TILTED_CROSS, 13, 2)
        elif kind == 'dot':
            cv2.circle(canvas, (x + 7, cy), 5, color, -1)
        elif kind == 'line':
            cv2.line(canvas, (x, cy), (x + 16, cy), color, 2)
        else:
            cv2.rectangle(canvas, (x, cy - 6), (x + 16, cy + 6), color, -1)
            cv2.rectangle(canvas, (x, cy - 6), (x + 16, cy + 6), RGB['sep'], 1)
        _put(canvas, label, (x + 18 + 5, cy + 4), scale, RGB['text'], 1)
        x += width + gap


# TodoMemory.render() emits "[todo-3] (COMPLETED)  find the pillow  => ...".
# Legacy runs (todo.legacy_text) instead carry a raw Markdown checklist, so both
# shapes have to be recognised -- keying only off "- [x]" reported every harness
# TODO as unfinished.
_TODO_RE = re.compile(r'^\[(?P<id>[^\]]+)\]\s*\((?P<status>[A-Z]+)\)\s*(?P<body>.*)$')
_TODO_MD = [('- [x] ', 'COMPLETED'), ('- [X] ', 'COMPLETED'),
            ('- [ ] ', 'PENDING'), ('* [x] ', 'COMPLETED'), ('* [ ] ', 'PENDING')]
# status -> (palette key, checkbox glyph)
_TODO_STYLE = {
    'COMPLETED': ('todo_done', 'check'),
    'ACTIVE': ('accent', 'dot'),
    'PENDING': ('text', 'empty'),
    'BLOCKED': ('bad', 'cross'),
}


def _parse_todo_line(line):
    """(todo_id, status, body) for one rendered TODO line.

    Unrecognised lines come back as PENDING with the text untouched -- the strip
    should still show whatever the planner was given, just without a status.
    """
    m = _TODO_RE.match(line)
    if m:
        status = m.group('status')
        return (m.group('id'), status if status in _TODO_STYLE else 'PENDING',
                m.group('body').strip())
    for marker, status in _TODO_MD:
        if line.startswith(marker):
            return None, status, line[len(marker):].strip()
    return None, 'PENDING', line


class LaViRAVisualizer:
    def __init__(self, mapping_module, visualize=False, save_dir="./visualizations", width=640, height=480,
                 debug_log_dir=None):
        self.visualize = visualize
        self.save_dir = save_dir
        self.width = width
        self.height = height
        self.current_episode_id = None
        self.current_step = 0
        if self.visualize:
            os.makedirs(self.save_dir, exist_ok=True)
        self.rgb_history = []
        self.mapping_module = mapping_module
        # Cache the latest grounding box for overlays on per-step frames.
        self._last_bbox = None
        # Last _save_rgb_frame call, replayed by redraw_last_frame().
        self._last_frame = None
        self._last_frame_sig = None
        self.model_name = os.environ.get('MODEL_NAME', '') or ''
        self.dataset_label = ''
        # When set, every per-step combined image also dumps the topdown VLM map
        # plus the raw RGB into <debug_log_dir>/<episode>/step_<N>/ so that the
        # debug_logs structure has paired prompt/response/images + map view.
        self.debug_log_dir = debug_log_dir

    def sync(self, step, ep_id):
        if ep_id != self.current_episode_id:
            # New episode: nothing from the previous one may be redrawn or
            # overlaid onto the first frames of this one.
            self._last_frame = None
            self._last_frame_sig = None
            self._last_bbox = None
        self.current_step = step
        self.current_episode_id = ep_id

    def update_map(self, mapping_module):
        self.mapping_module = mapping_module

    def _save_rgb_with_bbox(self, rgb_image, bbox, target_coords=None):
        """Save RGB image with bounding box annotation as a separate file
        
        Args:
            rgb_image: RGB image to save
            bbox: Bounding box dictionary
            target_coords: Optional (x, y) tuple of target point to visualize
        """
        if not self.visualize:
            return

        # Cache for the combined frame's RGB panel overlay (see _last_bbox above).
        try:
            self._last_bbox = (self.current_step, dict(bbox))
        except Exception:
            self._last_bbox = None

        try:
            # Convert RGB image to proper format
            if isinstance(rgb_image, np.ndarray):
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                # Make a copy to avoid modifying the original
                annotated_image = rgb_image.copy()
            else:
                annotated_image = np.array(rgb_image)

            # Draw bounding box
            x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            target_description = bbox.get('target', 'target')
            action_decision = bbox.get('action', 'NAVIGATE')

            # Choose color based on action decision
            box_color = (0, 255, 0) if action_decision == 'NAVIGATE' else (0, 0,
                                                                           255)  # Green for navigate, Red for stop

            # Draw rectangle with thicker border for better visibility
            cv2.rectangle(annotated_image, (x, y), (x + width, y + height), box_color, 4)

            # Draw target point if provided
            if target_coords is not None:
                tx, ty = target_coords
                # Draw red dot
                cv2.circle(annotated_image, (int(tx), int(ty)), 8, (0, 0, 255), -1)
                # Draw white border for better visibility
                cv2.circle(annotated_image, (int(tx), int(ty)), 8, (255, 255, 255), 2)

            # Create footer with "Navigate: XXX"
            action_text = "Navigate" if action_decision == 'NAVIGATE' else action_decision.capitalize()
            footer_text = f"{action_text}: {target_description}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # Dynamic font scaling
            h, w = annotated_image.shape[:2]
            max_text_width = w - 40 # Padding of 20px on each side
            
            (text_width, text_height), baseline = cv2.getTextSize(footer_text, font, font_scale, thickness)
            
            if text_width > max_text_width:
                font_scale = font_scale * (max_text_width / text_width)
                (text_width, text_height), baseline = cv2.getTextSize(footer_text, font, font_scale, thickness)
            
            footer_height = int(text_height * 2.5) # Dynamic footer height
            
            # Extend image
            new_h = h + footer_height
            
            new_image = np.zeros((new_h, w, 3), dtype=np.uint8) # Black background
            new_image[:h, :, :] = annotated_image
            
            # Center text in footer
            text_x = max(20, (w - text_width) // 2)
            text_y = h + (footer_height + text_height) // 2 - int(text_height * 0.2)
            
            cv2.putText(new_image, footer_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
            
            annotated_image = new_image

            # Save to episode-specific folder with chronological naming
            episode_id = str(self.current_episode_id) if self.current_episode_id else "unknown"
            img_folder = os.path.join(self.save_dir, episode_id)

            # Create episode folder if it doesn't exist
            os.makedirs(img_folder, exist_ok=True)

            # Use current step for chronological ordering
            filename = f"bbox_step{self.current_step:04d}.png"
            filepath = os.path.join(img_folder, filename)

            # Convert RGB to BGR for cv2 saving (cv2 expects BGR)
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, annotated_image_bgr)


        except Exception as e:
            logger.info(f"Error saving RGB with bbox: {e}")

    def _save_depth(
            self,
            depth_image: np.ndarray,
            step: int,
            invert: bool = False,
            save_16bit: bool = False,
            use_colormap: bool = True,
            colormap: int = cv2.COLORMAP_JET
    ):
        """
        Save depth:
          - If save_16bit=True: 16-bit single channel PNG (no colormap).
          - Else if use_colormap=True: colored PNG using OpenCV colormap (default Jet).
          - Else: 8-bit grayscale PNG.
        Args:
            depth_image: float32 depth normalized or raw in [0,1] (will be clipped).
            invert: if True, near becomes bright.
            save_16bit: save a 16-bit raw depth (no colormap).
            use_colormap: apply colormap (ignored if save_16bit=True).
            colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET).
        Returns:
            np.ndarray saved array (8-bit color/grayscale or 16-bit).
        """
        if not self.visualize:
            return None

        try:
            depth = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.clip(depth, 0.0, 1.0)
            if invert:
                depth = 1.0 - depth

            episode_id = str(self.current_episode_id) if getattr(self, "current_episode_id", None) else "unknown"
            out_dir = os.path.join(self.save_dir, episode_id)
            os.makedirs(out_dir, exist_ok=True)

            if save_16bit:
                depth_u16 = (depth * 65535.0 + 0.5).astype(np.uint16)
                return depth_u16

            depth_u8 = (depth * 255.0 + 0.5).astype(np.uint8)

            if use_colormap:
                colored = cv2.applyColorMap(depth_u8, colormap)
                return colored
            else:
                return depth_u8
        except Exception as e:
            logger.info(f"Error saving depth: {e}")
            return None

    # ------------------------------------------------------------------
    # Combined per-step frame
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_action_display(action):
        """Normalise whatever the evaluator stashed in metadata['action'] into a
        display string. Falls back to '-' rather than 'UNKNOWN': at step 0 no
        low-level action has been issued yet, and 'UNKNOWN' reads like a fault.
        """
        try:
            if isinstance(action, dict):
                action = action.get('action', '')
            if isinstance(action, str) and action.isdigit():
                action = int(action)
            if isinstance(action, int) and 0 <= action < len(ACTION_TO_TEXT):
                return ACTION_TO_DISPLAY.get(action, ACTION_TO_TEXT[action].upper())
            if isinstance(action, str) and action and action.lower() != 'unknown':
                return ACTION_TO_DISPLAY.get(action, action.upper())
        except Exception as e:
            logger.info(f"Error parsing action: {e}")
        return '-'

    @staticmethod
    def _describe_plan(reasoning_info):
        """Return (short action phrase, dict of side-panel fields) for the planner output."""
        if not reasoning_info:
            return '', {}
        action = reasoning_info.get('action', reasoning_info.get('decision', '')) or ''
        direction = reasoning_info.get('direction', '') or ''
        waypoint = reasoning_info.get('waypoint', '') or ''
        stop_signal = bool(reasoning_info.get('stop_signal', False))

        if action == 'NAVIGATE' and direction:
            phrase = f"navigate to {direction}"
        elif action == 'BACKTRACK':
            phrase = f"backtrack to {waypoint}"
        else:
            phrase = str(action)
        if stop_signal:
            phrase += " (STOP)"

        fields = {
            'action': str(action) or '-',
            'target': direction or waypoint or '-',
            'stop_signal': stop_signal,
        }
        return phrase, fields

    def _draw_title_bar(self, canvas, metadata, step, action_display, telemetry):
        canvas[0:TITLE_H, :] = RGB['title_bg']
        cv2.line(canvas, (0, TITLE_H - 1), (CANVAS_W, TITLE_H - 1), RGB['bar_fg'], 2)

        _put(canvas, 'HarnessVLN', (24, 42), 0.95, RGB['title_fg'], 2)
        x = 24 + _text_w('HarnessVLN', 0.95, 2) + 18
        for chip in [c for c in (telemetry.get('dataset') or self.dataset_label,
                                 telemetry.get('model') or self.model_name) if c]:
            cw = _text_w(chip, 0.46) + 18
            cv2.rectangle(canvas, (x, 20), (x + cw, 46), (46, 52, 64), -1)
            _put(canvas, chip, (x + 9, 39), 0.46, RGB['title_dim'], 1)
            x += cw + 8

        # Only the closing frame carries a terminal reason; it says how the
        # episode actually ended, which no counter on its own does.
        terminal = telemetry.get('terminal')
        if terminal:
            chip = f"TERMINAL: {terminal}"
            cw = _text_w(chip, 0.46, 2) + 18
            cv2.rectangle(canvas, (x, 20), (x + cw, 46), RGB['bad'], -1)
            _put(canvas, chip, (x + 9, 39), 0.46, (255, 255, 255), 2)
            x += cw + 8

        try:
            max_step = int(telemetry.get('max_step') or 500)
        except Exception:
            max_step = 500
        max_step = max(1, max_step)

        parts = [('episode', str(metadata.get('episode_id', 'N/A'))),
                 ('step', f"{step} / {max_step}"),
                 ('action', action_display),
                 ('goal', str(metadata.get('destination', 'unknown')))]
        d2g = telemetry.get('distance_to_goal')
        if isinstance(d2g, (int, float)) and np.isfinite(d2g):
            parts.append(('d(goal)', f"{float(d2g):.2f} m"))

        x = 24
        for key, val in parts:
            _put(canvas, key, (x, 74), 0.45, RGB['title_dim'], 1)
            x += _text_w(key, 0.45) + 8
            _put(canvas, val, (x, 74), 0.5, RGB['title_fg'], 1)
            x += _text_w(val, 0.5) + 24

        bar_w = 240
        bx = CANVAS_W - 24 - bar_w
        _put(canvas, 'step budget', (bx, 32), 0.42, RGB['title_dim'], 1)
        _progress_bar(canvas, bx, 42, bar_w, 12, float(step) / max_step)
        pct = f"{100.0 * step / max_step:.0f}%"
        _put(canvas, pct, (bx + bar_w - _text_w(pct, 0.42), 74), 0.42, RGB['title_dim'], 1)

    def _draw_rgb_panel(self, canvas, rgb_display, step):
        x0, y0, w = 0, MAIN_Y0, RGB_W
        canvas[y0:MAIN_Y1, x0:x0 + w] = RGB['panel']

        bbox = None
        if self._last_bbox is not None and self._last_bbox[0] == step:
            bbox = self._last_bbox[1]
        _panel_header(canvas, x0, y0, w, 'EGOCENTRIC RGB', RGB['hdr_rgb'],
                      right='grounding box' if bbox else 'no grounding box this step')

        cy0 = y0 + PANEL_HDR_H
        img, ox, oy, scale = _fit_into(rgb_display, w - 16, (MAIN_Y1 - cy0) - 16)
        px, py = x0 + 8 + ox, cy0 + 8 + oy
        ih, iw = img.shape[:2]
        canvas[py:py + ih, px:px + iw] = img
        cv2.rectangle(canvas, (px - 1, py - 1), (px + iw, py + ih), RGB['sep'], 1)

        if bbox is None:
            return
        try:
            bx = px + int(round(float(bbox['x']) * scale))
            by = py + int(round(float(bbox['y']) * scale))
            bw = int(round(float(bbox['width']) * scale))
            bh = int(round(float(bbox['height']) * scale))
            decision = bbox.get('action', 'NAVIGATE')
            color = RGB['bbox_nav'] if decision == 'NAVIGATE' else RGB['bbox_stop']
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, 3)

            label = f"{bbox.get('target', 'target')}  [{decision}]"
            lw = min(_text_w(label, 0.5) + 14, iw)
            ly = by - 24 if by - 24 >= py else min(by + bh + 2, py + ih - 24)
            lx = min(bx, px + iw - lw)
            cv2.rectangle(canvas, (lx, ly), (lx + lw, ly + 22), color, -1)
            _put(canvas, label, (lx + 7, ly + 16), 0.5, (255, 255, 255), 1)
        except Exception as e:
            logger.warning(f'failed to overlay grounding bbox on combined frame: {e}')

    def _draw_map_panel(self, canvas, vlm_map_display, note=None):
        x0, y0, w = RGB_W, MAIN_Y0, MAP_W
        canvas[y0:MAIN_Y1, x0:x0 + w] = RGB['panel']
        _panel_header(canvas, x0, y0, w, 'TOPDOWN VLM MAP', RGB['hdr_map'], right=note)

        cy0 = y0 + PANEL_HDR_H
        cy1 = MAIN_Y1 - MAP_LEGEND_H
        img, ox, oy, _ = _fit_into(_autocrop_map(vlm_map_display), w - 16, (cy1 - cy0) - 14)
        px, py = x0 + 8 + ox, cy0 + 7 + oy
        ih, iw = img.shape[:2]
        canvas[py:py + ih, px:px + iw] = img
        cv2.rectangle(canvas, (px - 1, py - 1), (px + iw, py + ih), RGB['sep'], 1)
        _draw_map_legend(canvas, x0, cy1, w)
        canvas[y0:MAIN_Y1, x0] = RGB['sep']

    def _draw_side_panel(self, canvas, plan_fields, telemetry):
        x0, y0, w = RGB_W + MAP_W, MAIN_Y0, SIDE_W
        canvas[y0:MAIN_Y1, x0:x0 + w] = RGB['panel']
        _panel_header(canvas, x0, y0, w, 'HARNESS STATE', RGB['hdr_side'])
        canvas[y0:MAIN_Y1, x0] = RGB['sep']

        pad = 12
        tx, rx = x0 + pad, x0 + w - pad
        state = {'y': y0 + PANEL_HDR_H + 26}

        def section(title):
            if state['y'] > y0 + PANEL_HDR_H + 30:
                state['y'] += 10
            if state['y'] > MAIN_Y1 - 24:
                return
            _put(canvas, title, (tx, state['y']), 0.46, RGB['accent'], 2)
            state['y'] += 7
            cv2.line(canvas, (tx, state['y']), (rx, state['y']), RGB['sep'], 1)
            state['y'] += 21

        def row(key, val, color=None):
            if state['y'] > MAIN_Y1 - 10:
                return
            _put(canvas, key, (tx, state['y']), 0.43, RGB['muted'], 1)
            s = str(val)
            _put(canvas, s, (rx - _text_w(s, 0.45), state['y']), 0.45, color or RGB['text'], 1)
            state['y'] += 21

        def badge(text, color):
            cv2.rectangle(canvas, (tx, state['y'] - 4), (rx, state['y'] + 24), color, -1)
            _put(canvas, text, (tx + (rx - tx - _text_w(text, 0.52, 2)) // 2,
                                state['y'] + 16), 0.52, (255, 255, 255), 2)
            state['y'] += 38

        def num(d, *keys):
            for k in keys:
                if isinstance(d, dict) and d.get(k) is not None:
                    return d[k]
            return '-'

        def hot(v, color):
            """Colour a counter only when it actually fired."""
            return color if isinstance(v, int) and v else None

        stop = telemetry.get('stop') or {}
        graph = telemetry.get('graph') or {}
        nav = telemetry.get('nav') or {}

        section('DECISION')
        row('action', plan_fields.get('action', '-'),
            RGB['bad'] if plan_fields.get('stop_signal') else RGB['text'])
        target = str(plan_fields.get('target', '-'))
        if _text_w(target, 0.45) > (rx - tx) * 0.62:
            _put(canvas, 'target', (tx, state['y']), 0.43, RGB['muted'], 1)
            state['y'] += 19
            for line in wrap_text(target, w - 2 * pad, FONT, 0.44, 1)[:2]:
                _put(canvas, line, (tx, state['y']), 0.44, RGB['text'], 1)
                state['y'] += 19
            state['y'] += 2
        else:
            row('target', target)

        section('STOP GATE')
        requested = num(stop, 'requested', 'total_stop_requested')
        gate_pass = num(stop, 'gate_pass', 'total_stop_gate_pass')
        gate_fail = num(stop, 'gate_fail', 'total_stop_gate_fail')
        rejected = num(stop, 'rejected', 'total_stop_rejected')
        if not isinstance(requested, int) or requested == 0:
            # The frame is drawn before the iteration's decide, so a planner stop
            # signal shows up here one decide before the gate ever sees it.
            if plan_fields.get('stop_signal'):
                badge('STOP ASKED', RGB['warn'])
            else:
                badge('NO STOP YET', RGB['muted'])
        elif isinstance(gate_pass, int) and gate_pass > 0:
            badge('GATE PASS', RGB['ok'])
        elif isinstance(rejected, int) and rejected > 0:
            badge('GATE REJECT', RGB['bad'])
        else:
            badge('STOP PENDING', RGB['warn'])
        row('requested', requested)
        row('gate pass', gate_pass, hot(gate_pass, RGB['ok']))
        row('gate fail', gate_fail, hot(gate_fail, RGB['warn']))
        row('rejected', rejected, hot(rejected, RGB['bad']))
        row('cooldown supp.', num(stop, 'cooldown_suppressed', 'total_stop_cooldown_suppressed'))
        row('forced / auto', f"{num(stop, 'forced', 'total_stop_forced')}"
                             f" / {num(stop, 'autostop', 'total_autostop')}")

        section('SPATIOTEMPORAL GRAPH')
        row('places', num(graph, 'n_places'))
        row('objects', num(graph, 'n_objects'))
        row('failure nodes', num(graph, 'n_failures'), hot(graph.get('n_failures'), RGB['warn']))
        row('edges', num(graph, 'n_edges'))
        row('revisited', num(graph, 'revisited_places'), hot(graph.get('revisited_places'), RGB['warn']))

        section('EXECUTOR')
        row('backtracks', num(nav, 'backtracks', 'total_backtracks'))
        row('waypoints set', num(nav, 'waypoints', 'total_waypoints'))
        row('targets set', num(nav, 'target_set', 'total_target_set'))
        row('target timeout', num(nav, 'target_timeout', 'total_target_timeout'),
            hot(nav.get('total_target_timeout', nav.get('target_timeout')), RGB['warn']))
        row('proj. pullback', num(nav, 'projection_pullback', 'total_projection_pullback'))

    def _draw_todo_strip(self, canvas, instruction, plan_phrase, todo_list):
        y0 = STRIP_Y0
        canvas[y0:CANVAS_H, :] = RGB['bg']

        todo_lines = [ln.strip() for ln in (todo_list or '').split('\n') if ln.strip()]
        if len(todo_lines) == 1 and todo_lines[0] in ('(empty)', '(none)'):
            todo_lines = []
        parsed = [_parse_todo_line(ln) for ln in todo_lines]
        counts = {}
        for _, status, _b in parsed:
            counts[status] = counts.get(status, 0) + 1
        if parsed:
            note = f"{counts.get('COMPLETED', 0)}/{len(parsed)} done"
            for status, word in (('ACTIVE', 'active'), ('BLOCKED', 'blocked')):
                if counts.get(status):
                    note += f"   |   {counts[status]} {word}"
        else:
            note = 'TODO memory off / empty'
        _panel_header(canvas, 0, y0, CANVAS_W, 'INSTRUCTION  /  TODO MEMORY', RGB['accent'],
                      right=note)

        pad = 18
        y = y0 + PANEL_HDR_H + 26
        label = 'Instruction:'
        _put(canvas, label, (pad, y), 0.6, RGB['accent'], 2)
        ix = pad + _text_w(label, 0.6, 2) + 12
        for line in wrap_text(instruction or '(none)', CANVAS_W - pad - ix, FONT, 0.6, 1)[:2]:
            _put(canvas, line, (ix, y), 0.6, RGB['text'], 1)
            y += 28

        if plan_phrase:
            _put(canvas, f"Action:  {plan_phrase}", (pad, y), 0.56, RGB['warn'], 1)
            y += 26

        cv2.line(canvas, (pad, y - 6), (CANVAS_W - pad, y - 6), RGB['sep'], 1)
        y += 14

        if not parsed:
            _put(canvas, '(no TODO items yet)', (pad, y), 0.56, RGB['muted'], 1)
            return

        # Bumped from 0.5/20: at 1920 wide the strip is the block people actually
        # read word-for-word, and 0.5 Hershey is thin on a projector.
        row_h, scale = 25, 0.6
        box_sz, id_scale = 15, 0.5
        # widest id keeps the bodies left-aligned across rows
        id_w = max([_text_w(f"{t[0]}", id_scale) for t in parsed if t[0]] or [0])
        gut = pad + box_sz + 12
        body_x = gut + (id_w + 12 if id_w else 0)
        shown = 0
        for idx, (todo_id, status, body) in enumerate(parsed):
            if y > CANVAS_H - 16 - row_h:
                break
            key, glyph = _TODO_STYLE.get(status, _TODO_STYLE['PENDING'])
            color = RGB[key]
            box_y = y - box_sz + 2
            b0, b1 = (pad, box_y), (pad + box_sz, box_y + box_sz)
            m = box_sz // 4
            if glyph == 'check':
                cv2.rectangle(canvas, b0, b1, RGB['ok'], -1)
                cv2.line(canvas, (pad + m, box_y + box_sz // 2),
                         (pad + box_sz // 2, box_y + box_sz - m), (255, 255, 255), 2)
                cv2.line(canvas, (pad + box_sz // 2, box_y + box_sz - m),
                         (pad + box_sz - m, box_y + m), (255, 255, 255), 2)
            elif glyph == 'dot':
                cv2.rectangle(canvas, b0, b1, RGB['accent'], 1)
                cv2.circle(canvas, (pad + box_sz // 2, box_y + box_sz // 2),
                           max(3, box_sz // 4), RGB['accent'], -1)
            elif glyph == 'cross':
                cv2.rectangle(canvas, b0, b1, RGB['bad'], 1)
                cv2.line(canvas, (pad + m, box_y + m),
                         (pad + box_sz - m, box_y + box_sz - m), RGB['bad'], 2)
                cv2.line(canvas, (pad + box_sz - m, box_y + m),
                         (pad + m, box_y + box_sz - m), RGB['bad'], 2)
            else:
                cv2.rectangle(canvas, b0, b1, RGB['sep'], 1)
            if todo_id:
                _put(canvas, todo_id, (gut, y), id_scale, RGB['muted'], 1)
            wrapped = wrap_text(body, CANVAS_W - body_x - pad, FONT, scale, 1)
            for wline in wrapped:
                if y > CANVAS_H - 16 - row_h:
                    break
                _put(canvas, wline, (body_x, y), scale, color, 1)
                y += row_h
            shown = idx + 1

        remaining = len(parsed) - shown
        if remaining > 0:
            msg = f"... +{remaining} more"
            _put(canvas, msg, (body_x, min(y, CANVAS_H - 10)), 0.56, RGB['muted'], 1)

    def _render_map(self, goal_tensor, visited_targets, navdp_traj):
        """Render the topdown map for the frame.

        marker_scale is passed by keyword and dropped if the mapping module
        predates it, so this file never hard-depends on that signature.
        """
        kwargs = dict(output_size=(1024, 1024), visited_targets=visited_targets,
                      display_last=True, navdp_traj=navdp_traj)
        try:
            return self.mapping_module.create_vlm_map_from_state(
                self.current_episode_id, 0, goal_tensor,
                marker_scale=MAP_MARKER_SCALE, **kwargs)
        except TypeError:
            return self.mapping_module.create_vlm_map_from_state(
                self.current_episode_id, 0, goal_tensor, **kwargs)

    def _create_combined_image(self, rgb_image, metadata, step, visited_targets, target_coords=None,
                               reasoning_info=None, todo_list=None, navdp_traj=None, plan_output=None,
                               telemetry=None):
        """Render the fixed 1920x1080 per-step frame.

        Layout: title bar / (RGB | topdown map | harness state) / instruction+TODO.

        Args:
            rgb_image: RGB observation image
            metadata: dict with episode_id, instruction, destination, action
            step: current step number
            visited_targets: waypoints to mark on the topdown map
            target_coords: (target_map_x, target_map_y) or None
            reasoning_info: legacy alias for plan_output
            todo_list: Markdown checklist string
            navdp_traj: list of (x, y) world coords for the NavDP plan overlay
            plan_output: dict of the planner's decision for this step
            telemetry: optional harness state for the side panel; every key is
                optional so callers that predate it keep working
        """
        if plan_output:
            reasoning_info = plan_output
        telemetry = telemetry or {}

        goal_tensor = None
        if target_coords is not None and target_coords[0] is not None and target_coords[1] is not None:
            goal_tensor = torch.tensor([target_coords[0], target_coords[1]], dtype=torch.float32)

        vlm_map = self._render_map(goal_tensor, visited_targets, navdp_traj).copy()

        # When DEBUG_LOGGING is on, dump a per-step topdown map next to the
        # model prompt/response under the structured debug log dir.
        if self.debug_log_dir is not None and self.current_episode_id is not None:
            try:
                _step_dir = os.path.join(self.debug_log_dir, str(self.current_episode_id), f'step_{step}')
                os.makedirs(_step_dir, exist_ok=True)
                # vlm_map is BGR from create_vlm_map_from_state; cv2.imwrite expects BGR.
                cv2.imwrite(os.path.join(_step_dir, 'topdown_map.png'), vlm_map)
                # Also save the raw RGB observation alongside for parity with the map.
                if isinstance(rgb_image, np.ndarray):
                    rgb_to_save = rgb_image
                    if rgb_to_save.dtype != np.uint8:
                        rgb_to_save = (rgb_to_save * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(_step_dir, 'rgb_obs.png'),
                                cv2.cvtColor(rgb_to_save, cv2.COLOR_RGB2BGR))
            except Exception as _e:
                logger.warning(f'failed to save step debug map/rgb: {_e}')

        rgb_display = rgb_image.copy() if isinstance(rgb_image, np.ndarray) else np.array(rgb_image)
        if rgb_display.dtype != np.uint8:
            rgb_display = np.clip(rgb_display * 255, 0, 255).astype(np.uint8)
        if rgb_display.ndim == 2:
            rgb_display = np.stack([rgb_display] * 3, axis=2)
        elif rgb_display.ndim == 3 and rgb_display.shape[2] == 4:
            rgb_display = rgb_display[:, :, :3]
        elif rgb_display.ndim != 3 or rgb_display.shape[2] != 3:
            logger.info(f"Warning: Unexpected RGB shape: {rgb_display.shape}")

        # create_vlm_map_from_state returns BGR; the canvas below is RGB.
        if vlm_map.ndim == 3:
            vlm_map_display = cv2.cvtColor(vlm_map, cv2.COLOR_BGR2RGB)
        else:
            vlm_map_display = np.stack([vlm_map] * 3, axis=2)

        plan_phrase, plan_fields = self._describe_plan(reasoning_info)
        action_display = self._resolve_action_display(metadata.get('action', 'unknown'))
        map_note = None
        if isinstance(target_coords, (tuple, list)) and len(target_coords) == 2 \
                and target_coords[0] is not None:
            map_note = f"nav target ({int(target_coords[0])}, {int(target_coords[1])})"

        canvas = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)
        self._draw_title_bar(canvas, metadata, step, action_display, telemetry)
        self._draw_rgb_panel(canvas, rgb_display, step)
        self._draw_map_panel(canvas, vlm_map_display, note=map_note)
        self._draw_side_panel(canvas, plan_fields, telemetry)
        self._draw_todo_strip(canvas, metadata.get('instruction', ''), plan_phrase, todo_list)
        cv2.line(canvas, (0, STRIP_Y0), (CANVAS_W, STRIP_Y0), RGB['sep'], 1)

        return Image.fromarray(canvas)

    def _save_rgb_frame(self, obs: Observations, step: int, visited_targets, episode_id: str = None, target_coords: tuple = None, todo_list: str = None, navdp_traj: list = None, plan_output: dict = None, telemetry: dict = None):
        """Save RGB frame with metadata, instruction text and top-down map"""
        if not self.visualize:
            return

        rgb_image = obs['rgb']
        episode_id = episode_id or self.current_episode_id
        episode_id = str(episode_id) if not isinstance(episode_id, str) else episode_id

        metadata = {
            'episode_id': episode_id,
            'step': step,
            'timestamp': time.time(),
            'instruction': getattr(self, 'instruction', ''),
            'pose': obs.get('sensor_pose', None),
            'destination': getattr(self, 'destination', 'unknown'),
            'action': getattr(self, '_action', 'unknown')
        }

        self._save_individual_rgb(rgb_image, episode_id, step)

        # logger.info(target_coords)
        # Create combined image with RGB, top-down map, and value map heatmap
        combined_image = self._create_combined_image(rgb_image, metadata, step, visited_targets, target_coords, todo_list=todo_list, navdp_traj=navdp_traj, plan_output=plan_output, telemetry=telemetry)

        img_folder = os.path.join(self.save_dir, episode_id)
        img_path = os.path.join(img_folder, f"combined_step{step:04d}.png")
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        combined_image.save(img_path)

        # Remember the call so redraw_last_frame() can re-render this same step
        # once the caller has resolved things that only exist after it (see there).
        self._last_frame = {'obs': obs, 'step': step, 'visited_targets': visited_targets,
                            'episode_id': episode_id, 'target_coords': target_coords,
                            'todo_list': todo_list, 'navdp_traj': navdp_traj,
                            'plan_output': plan_output, 'telemetry': telemetry}
        self._last_frame_sig = self._frame_sig(target_coords, todo_list, plan_output, telemetry)

    def _frame_sig(self, target_coords, todo_list, plan_output, telemetry):
        """Cheap identity of everything a redraw could change."""
        tel = telemetry or {}
        return (self._last_bbox[0] if self._last_bbox else None,
                tuple(target_coords) if target_coords is not None else None,
                str(todo_list), repr(plan_output),
                repr(tel.get('stop')), repr(tel.get('graph')), repr(tel.get('terminal')))

    def redraw_last_frame(self, **overrides):
        """Re-render the frame most recently written, with updated inputs.

        The evaluator draws a step's frame at the top of its loop, before that
        step's plan/decide runs -- so the grounding's grounding box, the planner's decision,
        the target it picked and the STOP gate's verdict for the step all arrive
        *after* the file is on disk. Calling this once the decide has finished
        rewrites the same combined_step<N>.png with them in place. Skips the
        work when nothing that shows up in the frame actually changed, which is
        every step where no decision was made.
        """
        if not self.visualize or not getattr(self, '_last_frame', None):
            return
        args = dict(self._last_frame)
        args.update(overrides)
        try:
            sig = self._frame_sig(args['target_coords'], args['todo_list'],
                                  args['plan_output'], args['telemetry'])
            if sig == self._last_frame_sig:
                return
            self._save_rgb_frame(**args)
        except Exception as e:
            logger.warning(f'failed to redraw visualization frame: {e}')

    def _save_individual_rgb(self, rgb_image, episode_id, step):
        """Disabled debug hook: per-step RGB-only dump. No-op by default; the
        combined visualization frame is saved instead."""

    def _save_waypoint_panorama_rgb(self, panorama_frames, waypoint_id, step):
        """Disabled debug hook: per-waypoint 4-direction RGB dump. No-op by default."""
