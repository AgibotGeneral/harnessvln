from openai import OpenAI
from PIL import Image
from habitat import logger
import io
import base64
import os
import json
import numpy as np

# Sentinel strings `generate()` returns when it could not get a real completion.
# Callers MUST check with `is_api_failure()` before treating the return value as
# model output: these contain no JSON, so a caller that skips the check will fail
# to parse them and silently degrade to a hardcoded fallback (a blind 'forward',
# a center bbox) while the log shows only a parse warning.
API_REJECTED = "Error: API rejected request (non-recoverable)"
API_MAX_RETRIES = "Error: Failed to get response from API after max retries"


def is_api_failure(text):
    """True if `text` is a generate() failure sentinel rather than model output."""
    return (not isinstance(text, str)) or text.startswith("Error: ")


def qwen_thinking_enabled():
    """Whether Qwen endpoints are allowed to think, from `MODEL_ENABLE_THINKING`.

    Off by default: on a trivial bbox query qwen3.7-plus spends ~2100 reasoning
    tokens and ~30s thinking, versus ~3s and 0 with thinking off, for the same
    answer (see the measurement table in `generate()`). Set MODEL_ENABLE_THINKING=1
    for a round that deliberately trades that wall-clock for reasoning depth.

    Read per call rather than at import so the multiprocessing workers see
    whatever the launcher exported, regardless of when `load_dotenv()` ran.
    """
    return os.environ.get('MODEL_ENABLE_THINKING', '').strip().lower() in (
        '1', 'true', 'yes', 'on')


def model_bbox_scale(model_name):
    """The coordinate convention the model returns `bbox_2d` in.

    Returns 1000 when the model normalises to a 0-1000 grid (the Qwen-VL
    convention), or None when it returns absolute pixel coordinates, in which
    case the caller must not rescale.

    Measured by compositing a marker at a known pixel box into a *real* habitat
    frame and scoring IoU under both interpretations
    (`tools/probe_bbox_real.py`): `qwen3.7-plus` and `qwen3.8-flash` use the
    0-1000 grid, while `qwen3.8-max`, `gpt-5.5`, `gpt-5.6-terra`, `gpt-5.6-luna`,
    `gpt-5.6-sol`, `gpt-6-astra`, `gpt-6-luna`, `kimi-k2.6` and `claude-opus-5` return pixels.
    One Qwen model breaks the family convention, so this dispatches on the exact
    model name rather than on a 'qwen' substring.

    Probe on a real frame, not a blank canvas. The older synthetic probe
    (one rectangle on an empty background) called
    `qwen3.8-flash` pixel-convention at IoU 1.000, and on that basis it was
    registered here -- then every real frame came back on the 0-1000 grid
    (68/68 boxes with y2 > 480 in a 640x480 image, i.e. off-frame waypoints at
    every step). An easy synthetic scene is not evidence about the workload.

    Guessing wrong is silent: the JSON still parses and the box is still
    well-formed, but dividing a pixel box by 1000 collapses it toward the
    top-left and the robot is steered at the wrong target every step. So an
    unprobed model gets the legacy /1000 behaviour only if it is Qwen-family
    (where the convention is at least likely, and where this code has always
    assumed it); any other unknown model defaults to pixels, the majority
    convention. Either way it logs that the convention was assumed rather than
    verified. Override with `MODEL_BBOX_SCALE=1000` or `MODEL_BBOX_SCALE=pixel` after
    probing a new model.
    """
    override = os.environ.get('MODEL_BBOX_SCALE', '').strip().lower()
    if override:
        if override in ('pixel', 'pixels', 'none', 'absolute'):
            return None
        try:
            return float(override)
        except ValueError:
            logger.warning(f"Ignoring unparseable MODEL_BBOX_SCALE={override!r}")

    name = (model_name or '').lower()
    if name in _MODEL_BBOX_SCALE_0_1000:
        return 1000.0
    if name in _MODEL_BBOX_SCALE_PIXEL:
        return None

    if 'qwen' in name:
        logger.warning(
            f"Model {model_name!r} bbox convention is unverified; assuming the "
            "Qwen 0-1000 grid. Probe it and set MODEL_BBOX_SCALE if that is wrong."
        )
        return 1000.0
    logger.warning(
        f"Model {model_name!r} bbox convention is unverified; assuming absolute "
        "pixels. Probe it and set MODEL_BBOX_SCALE=1000 if that is wrong."
    )
    return None


# Verified by probe, not inferred from the model family — see `model_bbox_scale`.
_MODEL_BBOX_SCALE_0_1000 = frozenset({'qwen3.7-plus', 'qwen3.8-flash'})
_MODEL_BBOX_SCALE_PIXEL = frozenset({
    'qwen3.8-max', 'gpt-5.5', 'gpt-5.6-terra', 'gpt-5.6-luna',
    'gpt-5.6-sol', 'gpt-6-astra', 'gpt-6-luna', 'kimi-k2.6', 'claude-opus-5',
})


class ModelClient:

    def __init__(self, api_key, base_url, model_name, image_cap=50,
                 token_parameter='max_tokens', client=None):
        if not model_name:
            raise ValueError('model_name is required')
        if token_parameter not in ('max_tokens', 'max_completion_tokens'):
            raise ValueError('Unsupported token parameter')
        self.image_cap = image_cap
        self.model_name = model_name
        self.token_parameter = token_parameter
        self.client = client if client is not None else OpenAI(
            api_key=api_key, base_url=base_url, timeout=120, max_retries=0)
        self.reset_stats()

    def image_to_base64(self, image):
        """Convert a PIL Image or numpy array to a base64-encoded string."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _save_debug_info(self, log_path, messages, response_text):
        if not log_path:
            return
        
        try:
            os.makedirs(log_path, exist_ok=True)
            
            # Save images and create a clean message list for saving
            saved_messages = []
            img_count = 0
            
            for msg in messages:
                new_msg = {'role': msg['role'], 'content': []}
                if isinstance(msg['content'], list):
                    for item in msg['content']:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            url = item['image_url']['url']
                            if url.startswith('data:image/'):
                                try:
                                    # Extract base64
                                    header, encoded = url.split(',', 1)
                                    data = base64.b64decode(encoded)
                                    img_filename = f"image_{img_count}.png"
                                    img_path = os.path.join(log_path, img_filename)
                                    with open(img_path, 'wb') as f:
                                        f.write(data)
                                    
                                    new_msg['content'].append({
                                        'type': 'image_url',
                                        'image_url': {'url': img_filename}
                                    })
                                    img_count += 1
                                except Exception as e:
                                    logger.error(f"Failed to save image: {e}")
                                    new_msg['content'].append({'type': 'image_url', 'image_url': {'url': 'FAILED_TO_SAVE'}})
                            else:
                                new_msg['content'].append(item)
                        else:
                            new_msg['content'].append(item)
                else:
                     new_msg['content'] = msg['content']
                saved_messages.append(new_msg)
                
            # Save prompt text
            with open(os.path.join(log_path, 'prompt.json'), 'w') as f:
                json.dump(saved_messages, f, indent=2)
                
            # Save response
            with open(os.path.join(log_path, 'response.txt'), 'w') as f:
                f.write(str(response_text))
        except Exception as e:
            logger.error(f"Failed to save debug info: {e}")

    @staticmethod
    def _parse_image_cap(err_str):
        """Extract the provider's image cap from an over-limit error, if that's what
        this error is. Returns None for unrelated errors."""
        import re
        if 'image' not in err_str.lower():
            return None
        for pat in (r'maximum allowed:\s*(\d+)',
                    r'maximum number of images \((\d+)\)',
                    r'at most (\d+) image'):
            m = re.search(pat, err_str)
            if m:
                return int(m.group(1))
        return None

    @staticmethod
    def _parse_unknown_param(err_str):
        """Extract the parameter name from an 'unknown parameter' rejection.

        Providers differ in which non-standard fields they accept (e.g. Qwen takes
        `enable_thinking`, OpenAI-compatible proxies reject it with
        `{'param': 'enable_thinking', 'code': 'unknown_parameter'}`). Resending the
        same payload is futile, but the request is salvageable by dropping the one
        field the provider named. Returns None for unrelated errors.
        """
        import re
        low = err_str.lower()
        if 'unknown_parameter' not in low and 'unknown parameter' not in low:
            return None
        for pat in (r"[Uu]nknown parameter: '([^']+)'",
                    r'"param"\s*:\s*"([^"]+)"',
                    r"'param'\s*:\s*'([^']+)'"):
            m = re.search(pat, err_str)
            if m and m.group(1):
                return m.group(1)
        return None

    @staticmethod
    def _drop_oldest_images(messages, keep):
        """Return a copy of `messages` with only the newest `keep` image parts.

        Callers append images oldest-first (navigation history, then the current
        views the decision actually hinges on), so dropping from the front sheds
        stale context and preserves the current observation. Text parts are kept
        so the prompt and its section headers stay intact.
        """
        total = sum(
            1 for msg in messages if isinstance(msg.get('content'), list)
            for item in msg['content']
            if isinstance(item, dict) and item.get('type') == 'image_url'
        )
        to_drop = max(0, total - keep)
        if to_drop == 0:
            return messages

        trimmed = []
        dropped = 0
        for msg in messages:
            if not isinstance(msg.get('content'), list):
                trimmed.append(msg)
                continue
            new_content = []
            for item in msg['content']:
                is_img = isinstance(item, dict) and item.get('type') == 'image_url'
                if is_img and dropped < to_drop:
                    dropped += 1
                    continue
                new_content.append(item)
            trimmed.append({**msg, 'content': new_content})
        return trimmed

    def generate(self, messages, images=None, max_new_tokens=1024, temperature=0.7,
                 log_path=None, max_retries=5, **kwargs):
        """Send one task-stage request through the shared model client.

        Retries are bounded here (SDK retries are disabled). Image-cap and
        parameter rejections repair the payload; transient/empty responses use
        backoff. Each returned response contributes usage, including empty ones.
        All exhausted paths return the same detectable failure sentinel.
        """
        import time
        if max_retries < 0:
            raise ValueError('max_retries must be nonnegative')
        rejected = set()
        extra_body = dict(kwargs.pop('extra_body', None) or {})
        token_parameter = self.token_parameter
        if self.image_cap:
            messages = self._drop_oldest_images(messages, self.image_cap)

        for attempt in range(max_retries + 1):
            body = {k: v for k, v in extra_body.items() if k not in rejected}
            options = {k: v for k, v in kwargs.items() if k not in rejected}
            if 'qwen' in self.model_name.lower() and 'enable_thinking' not in rejected:
                body.setdefault('enable_thinking', qwen_thinking_enabled())
            if 'enable_thinking' in body:
                # These provider-specific controls conflict when sent together.
                options.pop('reasoning_effort', None)
            elif 'reasoning_effort' not in rejected:
                options.setdefault('reasoning_effort', 'low')
            payload = dict(options, model=self.model_name, messages=messages,
                           temperature=temperature, timeout=120, extra_body=body)
            payload[token_parameter] = max_new_tokens
            repair = False
            try:
                response = self.client.chat.completions.create(**payload)
                self.stats['calls'] += 1
                usage = getattr(response, 'usage', None)
                if usage:
                    for target, source in (('input_tokens', 'prompt_tokens'),
                                           ('output_tokens', 'completion_tokens'),
                                           ('total_tokens', 'total_tokens')):
                        self.stats[target] += getattr(usage, source, 0) or 0
                if isinstance(response, str):
                    content = response
                else:
                    choices = getattr(response, 'choices', None) or []
                    content = getattr(choices[0].message, 'content', None) if choices else None
                if isinstance(content, str) and content.strip():
                    self._save_debug_info(log_path, messages, content)
                    return content
                logger.warning(f'Model {self.model_name} returned an empty response ({attempt + 1}/{max_retries + 1})')
            except Exception as exc:
                error = str(exc)
                logger.error(f'API Call error with model {self.model_name}: {error}')
                cap = self._parse_image_cap(error)
                if cap is not None:
                    trimmed = self._drop_oldest_images(messages, cap)
                    if trimmed is not messages:
                        messages = trimmed
                        repair = True
                bad_param = self._parse_unknown_param(error)
                if bad_param in ('max_tokens', 'max_completion_tokens') and bad_param == token_parameter:
                    other = 'max_completion_tokens' if token_parameter == 'max_tokens' else 'max_tokens'
                    rejected.add(bad_param)
                    if other not in rejected:
                        token_parameter = other
                        repair = True
                elif bad_param and (bad_param in body or bad_param in options):
                    rejected.add(bad_param)
                    repair = True
                # Gateway safety rejections repeat on every retry of the same frames.
                if ('data_inspection_failed' in error.lower()
                        or 'datainspectionfailed' in error.lower()
                        or 'inappropriate content' in error.lower()
                        or 'content safety system' in error.lower()
                        or 'request blocked by the gateway' in error.lower()):
                    logger.error('API rejected request: non-recoverable content rejection')
                    return API_REJECTED
            if attempt < max_retries:
                logger.warning(f'API retry {attempt + 1}/{max_retries} (payload_repaired={repair})')
                if not repair:
                    time.sleep(min(2 ** attempt, 10))
        logger.error(f'Model {self.model_name}: exhausted {max_retries + 1} attempts')
        return API_MAX_RETRIES

    def get_model_info(self):
        return {'model_name': self.model_name}

    def get_usage_stats(self):
        return {
            'model_name': self.model_name,
            'model': self.stats.copy(),
            'total_calls': self.stats['calls'],
            'total_tokens': self.stats['total_tokens'],
        }

    def print_usage_stats(self):
        summary = self.get_usage_stats()
        logger.info(f"Model {self.model_name}: {summary['total_calls']} calls, "
                    f"{summary['total_tokens']} tokens")
        return summary

    def reset_stats(self):
        self.stats = dict(calls=0, input_tokens=0, output_tokens=0, total_tokens=0)

    def eval(self):
        """Compatibility no-op for the remote model interface."""
