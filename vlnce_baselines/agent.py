"""model reasoning agent (VLMReasoningAgent).

Split out of ZS_Evaluator_mp.py: this module holds the language/vision reasoning
agent (prompt construction, navigate/backtrack, TODO-list memory, STOP double-check,
replanning). The evaluator in ZS_Evaluator_mp.py instantiates it per episode.
"""
import base64
import io
import json
import os
import re
from typing import List, Dict

import numpy as np
from PIL import Image

from habitat import logger

from .prompts import get_prompts
from .harness.todo import TodoMemory
from .objnav_categories import negative_constraints as _objnav_negative_constraints
from .utils.api import ModelClient, is_api_failure, model_bbox_scale
from .model_config import ModelConfig
from .utils.visualization import LaViRAVisualizer

class VLMReasoningAgent:
    def __init__(
        self,
        visualizer: LaViRAVisualizer,
        task_type: str = "VLN",
        use_guideline=True,
        use_working_memory=True,
        allow_move_behind=True,
        debug_logging=False,
        log_dir="logs/debug_logs",
        use_todo_list=True,
        backtrack_second_chance=True,
        use_backtrack=True,
        max_images_per_request=50,
        max_backtrack_distance_m=6.0,
        goal_vocabulary=None,
        stop_check_temperature=0.7,
    ):
        model_config = ModelConfig.from_env()
        logger.info(f"Model: {model_config.model_name}")
        self.bbox_scale = model_bbox_scale(model_config.model_name)
        self.model = ModelClient(
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            model_name=model_config.model_name,
            image_cap=max_images_per_request,
            token_parameter=model_config.token_parameter,
        )
        self.model.eval()
        self.visualizer = visualizer
        self.stair = False
        self.use_guideline = False
        self.use_working_memory = use_working_memory
        self.use_negative_constraints = True  # Switch to control negative constraints
        # Goal categories declared by the episode dataset, from the evaluator.
        # Negative constraints are built against this rather than a hard-coded
        # list, so MP3D's 21 categories are not scored against HM3D's 6.
        self.goal_vocabulary = list(goal_vocabulary or [])
        # Sampling temperature for the terminal STOP verdict; see
        # EVAL.STOP_CHECK_TEMPERATURE. Held on the agent rather than read from
        # config here because this class takes no config object.
        self.stop_check_temperature = float(stop_check_temperature)
        self.allow_move_behind = allow_move_behind
        self.use_todo_list = use_todo_list
        self.backtrack_second_chance = backtrack_second_chance
        self.task_type = task_type
        self.P = get_prompts(task_type)
        self.guideline = ""
        # TODO memory lives in the harness: stable todo_id addressing,
        # four statuses, evidence-checked completion. `self.todo_list` below is a
        # read/write view onto it, kept so existing call sites and logs work
        # unchanged -- there is deliberately no second copy of the list.
        self.todo = TodoMemory(logger=logger)
        self.todo_verification_feedback = ""
        self.todo_update_feedback = ""
        # One line of room semantics the planner writes about where it is
        # standing. The evaluator copies it onto the current PlaceNode, where it
        # becomes the text side of retrieval.
        # Asking for it in the existing nav response costs no extra API call.
        self.last_place_summary = ""
        self.debug_logging = debug_logging
        self.log_dir = log_dir
        self.last_object = ''
        self.last_grounding_log_dir = None  # cached path of most recent grounding save dir
        self.max_images_per_request = max_images_per_request
        # Shared with the memory readout, which only prints a "backtrack id" for
        # places this same threshold would let the planner name. Two literals
        # would drift and the readout would advertise an illegal action.
        self.max_backtrack_distance_m = float(max_backtrack_distance_m)

    def _negative_constraints(self, instruction):
        """The `{negative_constraints}` prompt line, or "" when it does not apply.

        ObjectNav only. On VLN the instruction is a whole sentence, so any
        route that happened to mention a goal category ("turn left at the chair")
        used to pick up a warning not to confuse that chair with a bed or a
        toilet -- advice about an object that is a landmark, not the target.
        """
        if not self.use_negative_constraints or self.task_type != "OBJNAV":
            return ""
        return _objnav_negative_constraints(instruction, self.goal_vocabulary)

    # TODO memory.
    @property
    def todo_list(self):
        """Legacy view of the TODO memory: list of dicts, or None before init.

        Kept as a property rather than a field so there is exactly one store.
        Two copies is how the index-addressing race stayed invisible: the
        prompt was rendered from one view and the updates applied to another.
        """
        return self.todo.as_list()

    @todo_list.setter
    def todo_list(self, value):
        if value is None:
            self.todo.reset()
        elif isinstance(value, str):
            self.todo.load_legacy_text(value)
        elif isinstance(value, list):
            self.todo.load(value)
        else:
            logger.warning(f"Ignoring todo_list assignment of type {type(value)}")

    def attach_harness(self, harness, strict_evidence: bool = False) -> None:
        """Point TODO evidence checks at a live HarnessRuntime.

        Without this the memory can only check that an ``evidence_ids`` entry
        has the shape of an id; with it, the id must have actually been issued
        for this episode. The evaluator wires this once, at construction: the
        runtime object outlives the episode but its id counters reset in
        ``start_episode``, and ``TodoMemory.reset()`` deliberately keeps the
        wiring, so a per-episode re-attach would be a no-op.
        """
        if harness is None:
            self.todo.evidence_resolver = None
            self.todo.evidence_step_resolver = None
            self.todo.on_event = None
            self.todo.on_note = None
            return
        self.todo.evidence_resolver = harness.has_evidence
        # Membership says the id is real; the step says whether it was available
        # before the obligation existed. Only the freshness *report* uses this --
        # see TodoMemory._check_freshness.
        self.todo.evidence_step_resolver = getattr(harness, 'evidence_step', None)
        self.todo.strict_evidence = bool(strict_evidence)
        # TodoMemory passes the step it was last applied at as the first
        # argument, which is exactly HarnessRuntime.violation's signature.
        self.todo.on_event = harness.violation
        # Freshness reports go to note(), not violation(): they are a new
        # measurement, and counting them as violations would move that metric
        # for every future round while the recorded rounds keep the old number.
        self.todo.on_note = getattr(harness, 'note', None)

    def _thin_history_images(self, history_images, reserved):
        """Subsample `history_images` so the request stays under the provider's image cap.

        `reserved` is how many non-history images the caller will also send (current
        views, working memory, failed-path frames). The newest frames matter most for
        "where am I now", so they are kept intact; the older tail is uniformly
        subsampled to preserve the shape of the trajectory. Returns the list
        unchanged when it already fits.
        """
        if not history_images:
            return history_images

        budget = self.max_images_per_request - reserved
        if budget <= 0:
            # Pathological config: no room for history at all. Keep the single newest
            # frame so the planner still gets some temporal context, and let the caller's
            # own images be what they are.
            logger.warning(
                f"Image budget exhausted by {reserved} non-history images "
                f"(cap {self.max_images_per_request}); keeping 1 history frame"
            )
            return history_images[-1:]

        if len(history_images) <= budget:
            return history_images

        # Keep the most recent 60% of the budget as a contiguous dense window, and
        # spread the rest evenly over everything older.
        n_recent = max(1, int(budget * 0.6))
        n_old = budget - n_recent
        recent = history_images[-n_recent:]
        older = history_images[:-n_recent]

        if n_old > 0 and older:
            step = len(older) / float(n_old)
            idxs = sorted({min(len(older) - 1, int(i * step)) for i in range(n_old)})
            sampled_old = [older[i] for i in idxs]
        else:
            sampled_old = []

        thinned = sampled_old + recent
        logger.info(
            f"Thinned history images {len(history_images)} -> {len(thinned)} "
            f"(budget {budget} = cap {self.max_images_per_request} - {reserved} reserved)"
        )
        return thinned

    @staticmethod
    def _count_images(content):
        """Number of image parts in an OpenAI-style content list."""
        return sum(
            1 for item in content
            if isinstance(item, dict) and item.get('type') == 'image_url'
        )

    def _generate_json(self, messages, max_new_tokens, temperature,
                       log_path=None, purpose="LLM", max_json_retries=5):
        """Call the model and return (parsed_dict_or_None, raw_text).

        Every decision site needs the same three things: retry when the reply has no
        JSON object, distinguish an API failure sentinel from a model that merely
        rambled, and never hand an error string to a JSON parser. Doing that once
        here keeps the sites consistent -- previously three of them retried and four
        degraded to a fallback on the first hiccup, so a single transient 400 could
        turn into a blind 'forward' action.
        """
        raw = ""
        for attempt in range(max_json_retries + 1):
            raw = self.model.generate(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                log_path=log_path,
            )
            if is_api_failure(raw):
                # generate() already exhausted its own transport-level retries and
                # logged the cause; retrying the identical payload here would just
                # replay it. Report the failure instead of parsing the sentinel.
                logger.error(f"{purpose}: API failure, no usable response ({raw})")
                return None, raw

            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                try:
                    return json.loads(match.group()), raw
                except Exception as e:
                    logger.warning(f"{purpose}: JSON decode failed ({e}); "
                                   f"attempt {attempt + 1}/{max_json_retries + 1}")
            else:
                logger.warning(f"{purpose}: no JSON object in response; "
                               f"attempt {attempt + 1}/{max_json_retries + 1}")

        logger.error(f"{purpose}: giving up after {max_json_retries + 1} attempts")
        return None, raw

    def _ensure_grounding_in_step(self, episode_id, step):
        """Ensure step_<step>/ has a grounding save. If grounding wasn't called this step, copy
        the last grounding save dir as `grounding_query_cached_<ts>/`.
        """
        if not self.debug_logging or episode_id is None or step is None:
            return
        if not self.last_grounding_log_dir or not os.path.isdir(self.last_grounding_log_dir):
            return
        try:
            import glob, shutil, time
            step_dir = os.path.join(self.log_dir, str(episode_id), f"step_{step}")
            existing = glob.glob(os.path.join(step_dir, 'grounding_query_*'))
            if existing:
                return
            os.makedirs(step_dir, exist_ok=True)
            dst = os.path.join(step_dir, f"grounding_query_cached_{time.time()}")
            shutil.copytree(self.last_grounding_log_dir, dst)
        except Exception as e:
            logger.warning(f'failed to mirror grounding into step_{step}: {e}')

    def _ensure_last_object(self, instruction: str, initial_views: List[Dict] = None,
                             episode_id=None, step=None):
        if self.last_object:
            return
        # Extracting a "last object" from a free-text instruction is VLN-specific;
        # for ObjectNav the target is already structured, so skip.
        if self.task_type != "VLN":
            self.last_object = ''
            return

        prompt_text = self.P.DESTINATION_PROMPT.format(instruction=instruction)
        logger.info("Generating Last Object...")
        logger.info(prompt_text)

        content = [{"type": "text", "text": prompt_text}]
        if initial_views:
            content.extend(initial_views)

        messages = [{"role": "user", "content": content}]

        log_path = None
        if self.debug_logging and episode_id is not None:
            import time
            log_path = os.path.join(self.log_dir, str(episode_id),
                                    f"step_{step if step is not None else 0}",
                                    f"destination_{time.time()}")

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=4096,
            temperature=0.7,
            log_path=log_path,
        )

        logger.info('Last Object:')
        logger.info("%s", output_text)
        # This one consumes raw text rather than JSON, so a failure sentinel would be
        # stored verbatim and then interpolated into later prompts as the target's
        # name ("Navigate to: Error: API rejected request"). Keep it empty instead.
        if is_api_failure(output_text):
            logger.error("Last object: API failure; leaving last_object empty")
            self.last_object = ''
            return
        self.last_object = output_text.strip()

    def offered_backtrack_ids(self, visited_targets):
        """Exactly the waypoint ids the action list will print this turn.

        ``backtrackable_waypoint_ids`` applies the distance rule; this adds the
        other condition the prompt builder has always used -- at least one prior
        waypoint has been grounded (``description``), i.e. there is something to
        go back *to*. The memory readout calls this so it can never advertise a
        `backtrack id` the action list omits.
        """
        if not any('description' in t for t in visited_targets[:-1]):
            return []
        return self.backtrackable_waypoint_ids(visited_targets)

    def _view_label(self, label, view_obs_ids, idx):
        """Stamp a view's caption with the ``ob-`` id issued for that frame.

        The planner is asked to cite an observation id when it claims a TODO item
        is done, so it has to be able to read one somewhere. This is where it can:
        `Current FORWARD view [ob-000021]:`. Falls back to the bare caption when
        no id is available (harness off, or an older waypoint recorded before the
        ids existed), which is why the old prompt shape is still reachable.
        """
        if not view_obs_ids or idx >= len(view_obs_ids) or not view_obs_ids[idx]:
            return label
        obs_id = view_obs_ids[idx]
        if label.endswith(':'):
            return f"{label[:-1]} [{obs_id}]:"
        return f"{label} [{obs_id}]"

    def backtrackable_waypoint_ids(self, visited_targets):
        """Waypoint ids the planner may legally name in `backtrack to <id>`.

        Excludes the waypoint just created (the last entry) and anything further
        than ``max_backtrack_distance_m`` away. Exposed as a method because the
        memory readout has to agree with it exactly: printing a "backtrack id"
        the action list does not offer would invite an illegal action, and
        recomputing the same filter in two places is how the two drift.

        ``world_coords`` are map cells at 5 cm each, which is where the 0.05
        comes from.
        """
        if not visited_targets:
            return []
        current_pos = visited_targets[-1].get('world_coords')
        if not current_pos:
            # No coords: keep the pre-existing permissive fallback rather than
            # silently removing the backtrack action.
            return list(range(len(visited_targets) - 1))
        ids = []
        for i in range(len(visited_targets) - 1):
            target_pos = visited_targets[i].get('world_coords')
            if not target_pos:
                continue
            dist_cells = np.sqrt((current_pos[0] - target_pos[0]) ** 2
                                 + (current_pos[1] - target_pos[1]) ** 2)
            if dist_cells * 0.05 <= self.max_backtrack_distance_m:
                ids.append(i)
        return ids

    def _build_nav_prompt(self, use_backtrack_prompt, negative_constraints, action_list_items, action_desc, feedback, memory=""):
        prompt_kwargs = {
            "negative_constraints": negative_constraints,
            "action_list": "\n".join(action_list_items),
            "action_desc": action_desc,
            "feedback": feedback,
            "memory": memory or "",
        }

        if self.use_todo_list:
            prompt_kwargs["todo_list"] = self._format_todo_for_prompt()
        else:
            prompt_kwargs["todo_list"] = "(none)"
        template = self.P.NAVIGATION_WITH_BACKTRACK_PROMPT if use_backtrack_prompt else self.P.NAVIGATION_PROMPT

        return template.format(**prompt_kwargs)

    def img_to_base64(self, img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64

    def _get_initial_views(self, visited_targets):
        """Helper to extract initial views from visited_targets."""
        initial_views = []
        if visited_targets and len(visited_targets) > 0:
            panorama_images = visited_targets[0]['panorama_frames']
            view_definitions = [
                {'angle': 0, 'name': 'forward', 'label': 'Initial FORWARD view'},
                {'angle': 90, 'name': 'left', 'label': 'Initial LEFT view'},
                {'angle': 180, 'name': 'behind', 'label': 'Initial BEHIND view'},
                {'angle': 270, 'name': 'right', 'label': 'Initial RIGHT view'}
            ]
            for view in view_definitions:
                angle = view['angle']
                frame_idx = angle // 90
                if frame_idx < len(panorama_images):
                    rgb_image = panorama_images[frame_idx]['rgb']
                    if isinstance(rgb_image, np.ndarray):
                        if rgb_image.dtype != np.uint8:
                            rgb_image = (rgb_image * 255).astype(np.uint8)
                        img = Image.fromarray(rgb_image)
                    else:
                        img = rgb_image
                    initial_views.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
                    initial_views.append({"type": "text", "text": view['label']})
        return initial_views

    def _check_blocked_directions(self, panorama_images):
        """
        Helper to check blocked directions based on depth.
        Returns a set of blocked direction names.
        """
        blocked_directions = set()
        num_frames = len(panorama_images)
        
        view_definitions = [
            {'angle': 0, 'name': 'forward'},
            {'angle': 90, 'name': 'left'},
            {'angle': 180, 'name': 'behind'},
            {'angle': 270, 'name': 'right'}
        ]
        
        for view in view_definitions:
            angle = view['angle']
            if num_frames > 0:
                frame_idx = int((angle / 360.0) * num_frames) % num_frames
            else:
                frame_idx = 0

            if frame_idx < len(panorama_images):
                # Check depth for blockage
                if 'depth' in panorama_images[frame_idx]:
                    depth_image = panorama_images[frame_idx]['depth']
                    # Handle potential (H, W, 1) shape
                    if len(depth_image.shape) == 3:
                        d_img = depth_image[:, :, 0]
                    else:
                        d_img = depth_image
                    
                    # Check center region (middle 1/3)
                    h, w = d_img.shape
                    center_d = d_img[h//3:2*h//3, w//3:2*w//3]
                    
                    # Assuming depth is normalized [0, 1] where small value means close obstacle
                    # Filter out 0.0 if it represents invalid/far, but keep small positive values
                    valid_mask = center_d > 0.01
                    if np.any(valid_mask):
                        mean_d = np.mean(center_d[valid_mask])
                        # Threshold 0.15 corresponds to roughly 0.8m - 1.5m depending on max_depth
                        if mean_d < 0.15: 
                            blocked_directions.add(view['name'])
                            
        return blocked_directions

    def navigate_or_backtrack(self, instruction, visited_targets, feedback="", episode_id=None, step=None, history_images=None, memory_readout=""):
        """Decide the next language-level action from instruction + history + 4-dir images.

        `memory_readout` is the recalled-places text block from
        :mod:`vlnce_baselines.harness.retrieval`. Empty by default, and an
        empty string renders the prompt exactly as before, so the feature is a
        config flag away from being a no-op.

        Returns one of: `navigate to {left,right,forward,behind}` or
        `backtrack to <waypoint_id>`.
        """

        panorama_images = visited_targets[-1]['panorama_frames'] if visited_targets else []

        # Cleared per turn so a failed planner call cannot leave the previous
        # waypoint's room description to be written onto this one.
        self.last_place_summary = ""

        history_content = []

        if history_images is not None:
            # Reserve room for the images appended after the history block: the 4
            # current-direction views, plus 4 more if working memory is on. Thin the
            # history to fit, so the provider never rejects the whole request.
            reserved = 4 if panorama_images else 0
            if self.use_working_memory and len(visited_targets) >= 2:
                reserved += 4 if panorama_images else 0
            history_images = self._thin_history_images(history_images, reserved)

        if history_images is not None:
            current_waypoint_idx = 0
            
            # Create a map of step -> waypoint_id
            step_to_waypoint = {}
            
            sorted_waypoints = sorted(visited_targets, key=lambda x: x['step'])
            
            for i, target in enumerate(sorted_waypoints):
                step_to_waypoint[target['step']] = i
            
            last_wp_idx = -1
            
            for item in history_images:
                img_step = item['step']
                
                # Find the most recent waypoint index <= img_step
                current_wp_segment_idx = -1
                for wp_step, wp_idx in step_to_waypoint.items():
                    if wp_step <= img_step:
                        if wp_idx > current_wp_segment_idx:
                            current_wp_segment_idx = wp_idx
                
                # If we moved to a new segment, insert header
                if current_wp_segment_idx > last_wp_idx:
                    leg_idx = -1
                    for i in range(len(sorted_waypoints) - 1):
                        wp_curr = sorted_waypoints[i]
                        wp_next = sorted_waypoints[i+1]
                        if wp_curr['step'] <= img_step < wp_next['step']:
                            leg_idx = i
                            break
                    
                    # Handle last segment (after last completed waypoint, before current)
                    if leg_idx == -1:
                        if len(sorted_waypoints) > 0 and img_step >= sorted_waypoints[-1]['step']:
                            leg_idx = len(sorted_waypoints) - 1
 

            last_leg_idx = -2
            
            for item in history_images:
                img_step = item['step']
                img = item['image']
                
                leg_idx = -1
                
                # Find interval [WP_i, WP_i+1)
                for i in range(len(sorted_waypoints) - 1):
                    if sorted_waypoints[i]['step'] <= img_step < sorted_waypoints[i+1]['step']:
                        leg_idx = i
                        break
                
                # If not found in intervals, check if it's after the last waypoint
                if leg_idx == -1 and len(sorted_waypoints) > 0:
                    if img_step >= sorted_waypoints[-1]['step']:
                        leg_idx = len(sorted_waypoints) - 1
                        
                # If leg changed, insert header
                if leg_idx != last_leg_idx:
                    if leg_idx >= 0:
                        # Determine label

                        start_wp = sorted_waypoints[leg_idx]
                        start_id = leg_idx
                        
                        if leg_idx < len(sorted_waypoints) - 1:
                            end_id = leg_idx + 1
                            label = f"Waypoint {start_id} -> Waypoint {end_id}: "
                        else:
                            # Last segment
                            label = f"Waypoint {start_id} -> Current Position: "
                            
                        history_content.append({"type": "text", "text": label})
                    last_leg_idx = leg_idx

                history_content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
        else:
            # Add history waypoint images and action descriptions
            for i, target in enumerate(visited_targets[:-1]):  # exclude the in-progress waypoint
                if 'init_image' in target:
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: Arrival view"})

                if 'dir_image' in target:
                    turn_direction = target.get('direction_decision', '')
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: After turn {turn_direction} view"})

                if 'description' in target:
                    history_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})

        current_views = []
        view_definitions = [
            {'angle': 0, 'name': 'forward', 'label': 'Current FORWARD view:'},
            {'angle': 90, 'name': 'left', 'label': 'Current LEFT view:'},
            {'angle': 180, 'name': 'behind', 'label': 'Current BEHIND view:'},
            {'angle': 270, 'name': 'right', 'label': 'Current RIGHT view:'}
        ]

        blocked_directions = self._check_blocked_directions(panorama_images)
        num_frames = len(panorama_images)

        # The `ob-` ids issued for this waypoint's four cardinal frames, in the
        # same forward/left/behind/right order they are shown in. Labelling the
        # views with them is what makes `evidence_ids` usable at all: the planner
        # is asked to cite an observation id, so it has to be able to see one.
        view_obs_ids = []
        if visited_targets:
            view_obs_ids = list(visited_targets[-1].get('obs_ids') or [])

        for idx, view in enumerate(view_definitions):
            angle = view['angle']
            # Calculate frame index based on total frames (assuming uniform distribution)
            # For 12 frames: 0->0, 90->3, 180->6, 270->9
            if num_frames > 0:
                frame_idx = int((angle / 360.0) * num_frames) % num_frames
            else:
                frame_idx = 0

            if frame_idx < len(panorama_images):
                rgb_image = panorama_images[frame_idx]['rgb']
                if isinstance(rgb_image, np.ndarray):
                    if rgb_image.dtype != np.uint8:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_image)
                else:
                    img = rgb_image
                label = self._view_label(view['label'], view_obs_ids, idx)
                current_views.append({"type": "text", "text": label})
                current_views.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})

        # Backtrack is offered when a prior waypoint was grounded and is close
        # enough; `offered_backtrack_ids` is the single definition of that set,
        # shared with the memory readout.
        available_ids = [str(i) for i in self.offered_backtrack_ids(visited_targets)]

        content = [{"type": "text", "text": f"Navigation Task: \"{instruction}\"\n\nNavigation History:"}]
        content.extend(history_content)
        if self.use_guideline and hasattr(self, 'guideline') and self.guideline:
            pass
        if self.use_working_memory and len(visited_targets) >= 2:
            # Working memory
            content.append({"type": "text", "text": "\nWorking Memory (previous waypoint snapshot): compare these with current views to judge progress, repeated scenes."})
            target = visited_targets[-2]
            view_definitions = [
                {'angle': 0, 'name': 'forward', 'label': 'Previous FORWARD view'},
                {'angle': 90, 'name': 'left', 'label': 'Previous LEFT view'},
                {'angle': 180, 'name': 'behind', 'label': 'Previous BEHIND view'},
                {'angle': 270, 'name': 'right', 'label': 'Previous RIGHT view'}
            ]
            previous_views = []

            for view in view_definitions:
                angle = view['angle']
                frame_idx = angle // 90
                if frame_idx < len(panorama_images):
                    rgb_image = panorama_images[frame_idx]['rgb']
                    if isinstance(rgb_image, np.ndarray):
                        if rgb_image.dtype != np.uint8:
                            rgb_image = (rgb_image * 255).astype(np.uint8)
                        img = Image.fromarray(rgb_image)
                    else:
                        img = rgb_image
                    previous_views.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
                    previous_views.append({"type": "text", "text": view['label']})
                
            content.extend(previous_views)   


        content.append({"type": "text", "text": "\nCurrent 4-directional views:"})
        content.extend(current_views)

        # Ensure reasoning state is available for the current episode.
        if self.use_todo_list and not self.todo.initialised:
            initial_views = self._get_initial_views(visited_targets)
            self.generate_todo_list(instruction, initial_views, episode_id=episode_id, step=step)
        elif not self.last_object:
            initial_views = self._get_initial_views(visited_targets)
            self._ensure_last_object(instruction, initial_views, episode_id=episode_id, step=step)

        negative_constraints = self._negative_constraints(instruction)

        # Generate action list dynamically
        action_list_items = []
        action_desc_parts = []
        
        # Directions we want to consider
        candidate_directions = ['forward', 'left', 'right']
        
        # Check if we are at the initial position (only current waypoint exists)
        is_initial_position = len(visited_targets) <= 1
        if self.allow_move_behind or is_initial_position:
            candidate_directions.append('behind')

        # Build action list based on valid directions (using blocked_directions calculated earlier)
        valid_directions = [d for d in candidate_directions if d not in blocked_directions]
        
        # Fallback: if all valid directions are blocked, allow behind or forward to avoid getting stuck
        if not valid_directions:
            if 'behind' in candidate_directions:
                valid_directions.append('behind')
            elif 'forward' in candidate_directions:
                valid_directions.append('forward')

        for direction in valid_directions:
            if direction == 'forward':
                action_list_items.append("   - navigate to forward - continue straight ahead")
            elif direction == 'left':
                action_list_items.append("   - navigate to left - turn left and go forward")
            elif direction == 'right':
                action_list_items.append("   - navigate to right - turn right and go forward")
            elif direction == 'behind':
                action_list_items.append("   - navigate to behind - turn around and go forward")
            
            action_desc_parts.append("navigate to " + direction)


        if not action_desc_parts:
             # Should be covered by fallback, but just in case
             action_desc = "navigate to behind" 
             action_list_items.append("   - navigate to behind - turn around and go forward (Emergency)")
        else:
             action_desc = "|".join(action_desc_parts)



        if available_ids:
            id_list_str = ", ".join(available_ids)
            action_list_items.append(f"   - backtrack to <waypoint_id> - return to a previous waypoint (Available IDs: {id_list_str})")
            action_desc += ' or "backtrack to <waypoint_id>"'
            prompt = self._build_nav_prompt(
                use_backtrack_prompt=True,
                negative_constraints=negative_constraints,
                action_list_items=action_list_items,
                action_desc=action_desc,
                feedback=feedback,
                memory=memory_readout,
            )
        else:
            prompt = self._build_nav_prompt(
                use_backtrack_prompt=False,
                negative_constraints=negative_constraints,
                action_list_items=action_list_items,
                action_desc=action_desc,
                feedback=feedback,
                memory=memory_readout,
            )

        logger.info(prompt)

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        log_path = None
        if self.debug_logging and episode_id is not None and step is not None:
             import time
             log_path = os.path.join(self.log_dir, str(episode_id), f"step_{step}", f"planning_{time.time()}")
             # mirror most recent grounding into this step folder if grounding wasn't called here
             self._ensure_grounding_in_step(episode_id, step)

        response_data, output_text = self._generate_json(
            messages=messages,
            max_new_tokens=8192,
            temperature=0.7,
            log_path=log_path,
            purpose="planner nav decision",
        )

        logger.info('planner-response:')
        logger.info(f"{output_text}")

        if response_data is None:
            logger.error("planner nav decision: no usable decision; falling back to 'forward'")
            return {
                'action': 'NAVIGATE',
                'direction': 'forward',
                'progress_analysis': 'Failed to parse JSON',
                'reasoning': 'Max retries reached'
            }

        if self.use_todo_list:
            # New diff-based mechanism: apply todo_updates if present
            if 'todo_updates' in response_data and response_data['todo_updates']:
                self._apply_todo_updates(response_data['todo_updates'], step=step)
                logger.info(f"Applied TODO updates. Current list:\n{self.todo.render()}")
            # Legacy fallback: support old 'updated_todo_list' field (full markdown rewrite)
            elif 'updated_todo_list' in response_data:
                legacy = response_data['updated_todo_list']
                if isinstance(legacy, list):
                    legacy = "\n".join(legacy)
                if self.todo.initialised and self.todo.legacy_text is None:
                    # Old format received but we have new structured - ignore
                    logger.info("Ignoring legacy updated_todo_list field (using structured format)")
                else:
                    self.todo.load_legacy_text(legacy)
                    logger.info(f"Updated TODO List (legacy): {legacy}")

        action = response_data.get('action', 'navigate to forward')
        progress_analysis = response_data.get('progress_analysis', '')
        # Room semantics for the place the agent is standing at. Read here and
        # left on the agent for the evaluator to copy onto the PlaceNode: the
        # planner is the only thing in the loop that can name a room, and it has
        # just been shown the four views it would name it from.
        place_summary = response_data.get('place_summary', '')
        self.last_place_summary = str(place_summary or '').strip()
        # Two-stage reasoning fields: reasoning_todo (for todo updates) and
        # reasoning_action (for the chosen action). Fall back to legacy 'reasoning'
        # if either is missing.
        legacy_reasoning = response_data.get('reasoning', '')
        self.reasoning_todo = response_data.get('reasoning_todo', legacy_reasoning)
        reasoning = response_data.get('reasoning_action', legacy_reasoning)
        self.stair = response_data.get('stair', False)

        if self.stair:
            logger.info(f"DEBUG: planner Model detected stairs: {self.stair}")
        stop_signal = response_data.get('stop', False)
        action = action.lower()

        # Models sometimes echo the prompt's action template verbatim, e.g.
        #   "navigate to forward|navigate to left|navigate to right or "backtrack..."
        # The direction parsing below is substring-based ('left' in action), so a
        # string naming every direction silently resolves to whichever branch is
        # checked first -- an arbitrary choice dressed up as a decision. Detect the
        # echo and fall back explicitly instead.
        if '|' in action or action.rstrip().endswith(' or'):
            logger.warning(
                f"planner echoed the action template instead of choosing "
                f"({action!r}); treating as no decision and going forward."
            )
            action = 'navigate to forward'

        if action == 'stop' or 'stop' in action.split():
             stop_signal = True
             if action == 'stop':
                 action = 'navigate to forward'

        # The planner's stop=true is trusted unconditionally; the only remaining
        # gate is the double-check STOP LLM pass below (visual verification
        # that the target is actually reached).

        if action.startswith('backtrack to'):
            waypoint_id = action.split('backtrack to ')[-1].strip()
            if waypoint_id.startswith('waypoint'):
                waypoint_id = waypoint_id.split('waypoint')[-1].strip()
            try:
                waypoint_id = int(waypoint_id)
                return {
                    'action': 'BACKTRACK',
                    'waypoint': waypoint_id,
                    'progress_analysis': progress_analysis,
                    'reasoning': reasoning
                }
            except:
                pass  # parse failure falls through to direction parsing below

        if 'forward' in action:
            direction = 'forward'
        elif 'left' in action:
            direction = 'left'
        elif 'right' in action:
            direction = 'right'
        elif 'behind' in action:
            direction = 'behind'
        else:
            direction = 'forward'

        return {
            'action': 'NAVIGATE',
            'direction': direction,
            'progress_analysis': progress_analysis,
            'reasoning': reasoning,
            'stop_signal': stop_signal
        }

    def generate_todo_list(self, instruction: str, initial_views: List[Dict] = None,
                           episode_id=None, step=None):
        """
        Generate initial TODO list as a structured list of dicts.
        Each item: {"content": str, "status": "pending"|"in_progress"|"completed", "result": str}
        """
        if not self.use_todo_list:
            self.todo.reset()
            self._ensure_last_object(instruction, initial_views, episode_id=episode_id, step=step)
            return

        # For VLN, merge TODO generation and last-object extraction into one planner call.
        # For other task types, last_object is not used (see _ensure_last_object).
        if self.task_type == "VLN" and hasattr(self.P, "INITIAL_PLAN_PROMPT"):
            prompt_text = self.P.INITIAL_PLAN_PROMPT.format(instruction=instruction)
            logger.info("Generating initial TODO list + last object (merged)...")
            log_purpose = "initial_plan"
        else:
            prompt_text = self.P.TODO_PLAN_PROMPT.format(instruction=instruction)
            logger.info("Generating initial TODO list...")
            log_purpose = "todo_plan"
        logger.info(prompt_text)

        content = [{"type": "text", "text": prompt_text}]
        if initial_views:
            content.extend(initial_views)

        messages = [{"role": "user", "content": content}]

        log_path = None
        if self.debug_logging and episode_id is not None:
            import time
            log_path = os.path.join(self.log_dir, str(episode_id),
                                    f"step_{step if step is not None else 0}",
                                    f"{log_purpose}_{time.time()}")

        data, output_text = self._generate_json(
            messages=messages,
            max_new_tokens=4096,
            temperature=0.7,
            log_path=log_path,
            purpose="TODO list",
        )

        logger.info('TODO+LastObject Output:' if self.task_type == "VLN" else 'TODO List Output:')
        logger.info("%s", output_text)

        if data is None:
            # A degenerate one-item TODO ("Find <instruction>") still lets the episode
            # run, but the plan is gone -- surface it rather than logging a warning
            # that reads like a mildly malformed reply.
            logger.error("TODO list: no usable response; falling back to a single-item plan")

        # Parse JSON TODO list (also picks up last_object if present)
        # `step` matters: the plan is written from the opening panorama, so items
        # created here must not look as if they predate it (TodoMemory.load).
        self.todo.load(self._parse_todo_json(output_text, instruction), turn=0, step=step)
        logger.info(f"Parsed TODO list ({len(self.todo.visible())} items):\n{self.todo.render()}")

        if self.task_type == "VLN" and data is not None:
            try:
                lo = data.get("last_object", "")
                if isinstance(lo, str) and lo.strip():
                    self.last_object = lo.strip()
                    logger.info(f"Last Object (merged): {self.last_object}")
            except Exception as e:
                logger.warning(f"Failed to read last_object from merged output: {e}")

        # Fallback: if last_object still empty (parse failed or non-VLN), call separate API
        self._ensure_last_object(instruction, initial_views, episode_id=episode_id, step=step)

    def _parse_todo_json(self, text: str, instruction: str) -> List[Dict]:
        """Parse the initial JSON TODO plan from planner output.

        Returns raw dicts; ids and status normalisation are TodoMemory's job.
        Statuses are passed through as written and normalised there, so a plan
        that comes back with "in_progress" is understood instead of being
        silently flattened to pending.
        """
        import json as _json
        import re as _re
        fallback = [{"content": f"Find {instruction}", "status": "pending", "result": ""}]
        m = _re.search(r'\{[\s\S]*\}', text)
        if not m:
            logger.warning(f"Failed to parse TODO JSON, using fallback. Raw: {text[:200]}")
            return fallback
        try:
            data = _json.loads(m.group())
            todos = data.get("todos", [])
            if not isinstance(todos, list) or not todos:
                return fallback
            parsed = []
            for t in todos:
                if isinstance(t, dict) and "content" in t:
                    parsed.append({
                        "content": str(t["content"]),
                        "status": t.get("status", "pending"),
                        "result": str(t.get("result", "")),
                    })
            return parsed or fallback
        except Exception as e:
            logger.warning(f"TODO JSON parse error: {e}. Raw: {text[:200]}")
            return fallback

    def _format_todo_for_prompt(self) -> str:
        """Render the TODO block for prompts, addressed by stable ``todo_id``."""
        return self.todo.render()

    def _apply_todo_updates(self, updates, step=None) -> None:
        """Apply planner updates using stable IDs or pre-batch index resolution.
        
        Rejected updates are returned to the planner as feedback.
        """
        self.todo.apply(updates, step=step)
        feedback = self.todo.rejection_feedback()
        self.todo_update_feedback = feedback
        if feedback:
            logger.info(feedback)

    def query_llm(self,
                  instruction: str,
                  visited_targets: List[Dict[str, str]],
                  rgb_image: np.ndarray,
                  depth_image: np.ndarray,
                  width: int,
                  height: int,
                  current_step: int,
                  progress_analysis: str = None,
                  planned_action: str = 'NAVIGATE',
                  episode_id=None):
        try:
            if isinstance(rgb_image, np.ndarray):
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                img = Image.fromarray(rgb_image)
            else:
                img = rgb_image

            if isinstance(depth_image, np.ndarray):
                if depth_image.dtype != np.uint8:
                    depth_image = (depth_image * 255).astype(np.uint8)
                if depth_image.ndim == 3 and depth_image.shape[2] == 1:
                    depth_image = np.squeeze(depth_image, axis=2)
                depth_img = Image.fromarray(depth_image)
            else:
                depth_img = depth_image

            content = [{
                'type': 'text',
                'text': self.P.ROBOT_NAVIGATION_SYSTEM_PROMPT,
            }]

            img_base64 = self.img_to_base64(img)
            depth_base64 = self.img_to_base64(depth_img)

            content.append({"type": "image_url", 'image_url': {'url': f"data:image/png;base64,{img_base64}"}})
            content.append({"type": "image_url", 'image_url': {'url': f"data:image/png;base64,{depth_base64}"}})

            progress_info = ""
            if progress_analysis:
                progress_info = f"\nProgress Analysis from Navigation Decision: {progress_analysis}\n"
            
            negative_constraints = self._negative_constraints(instruction)

            if self.use_todo_list:
                if not self.todo.initialised:
                    todo_text = "No TODO list available."
                else:
                    todo_text = self._format_todo_for_prompt()
            else:
                todo_text = "(none)"
            prompt = self.P.GROUNDING_PROMPT.format(
                instruction=instruction,
                current_step=current_step,
                width=width,
                height=height,
                todo_list=todo_text,
                progress_info=progress_info,
                planned_action=planned_action,
            )
            logger.info(prompt)
            content.append({
                "type": "text",
                "text": prompt
            })

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            log_path = None
            if self.debug_logging and episode_id is not None:
                import time, glob, shutil
                step_dir = os.path.join(self.log_dir, str(episode_id), f"step_{current_step}")
                # remove any prior cached grounding mirror in this step — a real grounding call now happens
                for _stale in glob.glob(os.path.join(step_dir, 'grounding_query_cached_*')):
                    try:
                        shutil.rmtree(_stale)
                    except Exception:
                        pass
                log_path = os.path.join(step_dir, f"grounding_query_{time.time()}")
                self.last_grounding_log_dir = log_path

            # Use the API client to generate response
            response_data, output_text = self._generate_json(
                messages=messages,
                max_new_tokens=10240,
                temperature=0,
                log_path=log_path,
                purpose="grounding bbox",
            )
            logger.info('LLM Output:')
            logger.info("%s", output_text)

            if response_data is not None:                # Extract action decision
                action_decision = response_data.get('action', planned_action).upper()

                # Extract bbox_2d in [x1, y1, x2, y2] format
                bbox_2d = response_data.get('bbox_2d')

                # Ensure we have 4 coordinates
                if isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) >= 4:
                    x1, y1, x2, y2 = bbox_2d[:4]
                    # Only rescale for models that normalise to a 0-1000 grid;
                    # pixel-convention models (most of them) are already in frame
                    # coordinates and dividing would collapse the box top-left.
                    if self.bbox_scale:
                        s = self.bbox_scale
                        x1, y1 = x1 / s * width, y1 / s * height
                        x2, y2 = x2 / s * width, y2 / s * height
                else:
                    # Missing or malformed bbox: fall back to a centre box. Already
                    # in pixels, so it must not go through the rescale above.
                    x1, y1, x2, y2 = width // 4, height // 4, 3 * width // 4, 3 * height // 4

                # Convert to x, y, width, height format for internal use
                x = int(x1)
                y = int(y1)
                img_w, img_h = width, height
                width = int(x2 - x1)
                height = int(y2 - y1)

                bbox = {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'target': response_data.get('target', 'unknown target'),
                    'action': action_decision,
                    'reasoning': response_data.get('reasoning', 'No reasoning provided'),
                    'progress': response_data.get('progress', 'Unknown progress')
                }

                # Ensure bbox is within image bounds
                bbox['x1'] = max(0, min(bbox['x1'], img_w - 1))
                bbox['y1'] = max(0, min(bbox['y1'], img_h - 1))
                bbox['x2'] = max(bbox['x1'] + 1, min(bbox['x2'], img_w))
                bbox['y2'] = max(bbox['y1'] + 1, min(bbox['y2'], img_h))

                # Update x, y, width, height based on bounded coordinates
                bbox['x'] = bbox['x1']
                bbox['y'] = bbox['y1']
                bbox['width'] = bbox['x2'] - bbox['x1']
                bbox['height'] = bbox['y2'] - bbox['y1']

                # Record this target if it's a new navigation target
                if action_decision == 'NAVIGATE' and bbox['target'] != 'unknown target':
                    target_record = {
                        'step': current_step,
                        'description': bbox['target'],
                        'bbox': {
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height'],
                            'x1': bbox['x1'],
                            'y1': bbox['y1'],
                            'x2': bbox['x2'],
                            'y2': bbox['y2']
                        },
                        'reasoning': bbox['reasoning']
                        # world_coords is set at waypoint creation, not here.
                    }

                    if visited_targets is not None:
                        visited_targets[-1].update(target_record)

                # Calculate target coordinates for visualization
                if not self.stair:
                    coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), min(int(bbox.get('y2', 0)), self.visualizer.height - 1))
                else:
                    # Stair navigation
                    # If 'up' (or True for legacy), target top (y1). If 'down', target bottom (y2).
                    if self.stair == 'down':
                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y2', 0)))
                    else:
                        # Default to 'up' behavior (y1)
                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y1', 0)))

                # Save RGB image with bounding box annotation
                self.visualizer._save_rgb_with_bbox(rgb_image, bbox, target_coords=coords)

                return bbox
            else:
                logger.error("grounding bbox: no usable response; using center-image fallback bbox")

        except Exception as e:
            logger.info(f"Error in query_llm: {e}")

        # Fallback bbox (center of image)
        x1_fallback = width // 4
        y1_fallback = height // 4
        x2_fallback = 3 * width // 4
        y2_fallback = 3 * height // 4

        fallback_bbox = {
            'x': x1_fallback,
            'y': y1_fallback,
            'width': x2_fallback - x1_fallback,
            'height': y2_fallback - y1_fallback,
            'x1': x1_fallback,
            'y1': y1_fallback,
            'x2': x2_fallback,
            'y2': y2_fallback,
            'target': 'fallback target',
            'action': 'NAVIGATE',
            'reasoning': 'Fallback due to parsing error',
            'progress': 'Unknown due to error'
        }

        # Calculate fallback coords
        if not self.stair:
            coords_fallback = (int((fallback_bbox.get('x1', 0) + fallback_bbox.get('x2', 0)) / 2.0), min(int(fallback_bbox.get('y2', 0)), self.visualizer.height - 1))
        else:
            if self.stair == 'down':
                coords_fallback = (int((fallback_bbox.get('x1', 0) + fallback_bbox.get('x2', 0)) / 2.0), int(fallback_bbox.get('y2', 0)))
            else:
                coords_fallback = (int((fallback_bbox.get('x1', 0) + fallback_bbox.get('x2', 0)) / 2.0), int(fallback_bbox.get('y1', 0)))

        # Save RGB image with fallback bounding box
        self.visualizer._save_rgb_with_bbox(rgb_image, fallback_bbox, target_coords=coords_fallback)

        return fallback_bbox

    def double_check_stop(self, instruction, panorama_frames, visited_targets, episode_id=None, step=None, history_images=None):
        """
        Double check if the agent should really stop based on 4-directional views.
        Returns: True if should stop, False otherwise.
        """
        # Construct current views
        current_views = []
        view_definitions = [
            {'angle': 0, 'name': 'forward', 'label': 'Current FORWARD view'},
            {'angle': 90, 'name': 'left', 'label': 'View after turning LEFT'},
            {'angle': 180, 'name': 'behind', 'label': 'View after turning BEHIND'},
            {'angle': 270, 'name': 'right', 'label': 'View after turning RIGHT'}
        ]

        for view in view_definitions:
            angle = view['angle']
            frame_idx = angle // 90
            if frame_idx < len(panorama_frames):
                rgb_image = panorama_frames[frame_idx]['rgb']
                if isinstance(rgb_image, np.ndarray):
                    if rgb_image.dtype != np.uint8:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_image)
                else:
                    img = rgb_image
                current_views.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
                current_views.append({"type": "text", "text": view['label']})

        history_content = []

        if history_images is not None:
            # This prompt sends only `current_views` (see the content assembly below),
            # but thin anyway so we don't burn CPU base64-encoding hundreds of frames
            # into a `history_content` that goes unused.
            history_images = self._thin_history_images(
                history_images, reserved=self._count_images(current_views)
            )

        if history_images is not None:
            current_waypoint_idx = 0
            
            # Create a map of step -> waypoint_id
            step_to_waypoint = {}
            
            sorted_waypoints = sorted(visited_targets, key=lambda x: x['step'])
            
            for i, target in enumerate(sorted_waypoints):
                step_to_waypoint[target['step']] = i
            
            last_wp_idx = -1
            
            for item in history_images:
                img_step = item['step']
                
                # Find the most recent waypoint index <= img_step
                current_wp_segment_idx = -1
                for wp_step, wp_idx in step_to_waypoint.items():
                    if wp_step <= img_step:
                        if wp_idx > current_wp_segment_idx:
                            current_wp_segment_idx = wp_idx
                
                # If we moved to a new segment, insert header
                if current_wp_segment_idx > last_wp_idx:
                    leg_idx = -1
                    for i in range(len(sorted_waypoints) - 1):
                        wp_curr = sorted_waypoints[i]
                        wp_next = sorted_waypoints[i+1]
                        if wp_curr['step'] <= img_step < wp_next['step']:
                            leg_idx = i
                            break
                    
                    # Handle last segment (after last completed waypoint, before current)
                    if leg_idx == -1:
                        if len(sorted_waypoints) > 0 and img_step >= sorted_waypoints[-1]['step']:
                            leg_idx = len(sorted_waypoints) - 1
 

            last_leg_idx = -2
            
            for item in history_images:
                img_step = item['step']
                img = item['image']
                
                leg_idx = -1
                
                # Find interval [WP_i, WP_i+1)
                for i in range(len(sorted_waypoints) - 1):
                    if sorted_waypoints[i]['step'] <= img_step < sorted_waypoints[i+1]['step']:
                        leg_idx = i
                        break
                
                # If not found in intervals, check if it's after the last waypoint
                if leg_idx == -1 and len(sorted_waypoints) > 0:
                    if img_step >= sorted_waypoints[-1]['step']:
                        leg_idx = len(sorted_waypoints) - 1
                        
                # If leg changed, insert header
                if leg_idx != last_leg_idx:
                    if leg_idx >= 0:
                        # Determine label

                        start_wp = sorted_waypoints[leg_idx]
                        start_id = leg_idx
                        
                        if leg_idx < len(sorted_waypoints) - 1:
                            end_id = leg_idx + 1
                            label = f"Waypoint {start_id} -> Waypoint {end_id}: "
                        else:
                            # Last segment
                            label = f"Waypoint {start_id} -> Current Position: "
                            
                        history_content.append({"type": "text", "text": label})
                    last_leg_idx = leg_idx

                history_content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
        else:
            # Add history waypoint images and action descriptions
            for i, target in enumerate(visited_targets[:-1]):  # exclude the in-progress waypoint
                if 'init_image' in target:
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: Arrival view"})


                if 'dir_image' in target:
                    turn_direction = target.get('direction_decision', '')
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: After turn {turn_direction} view"})

                if 'description' in target:
                    history_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})


        # Construct prompt (task-specific: VLN / OBJNAV)
        parts = self.P.STOP_CHECK_PROMPT.split("{current_views}")
        part1 = parts[0].format(instruction=instruction, target=self.last_object)
        rest = parts[1]
        
        content = [{"type": "text", "text": part1}]
        content.extend(current_views)
        content.append({"type": "text", "text": rest})

        messages = [{"role": "user", "content": content}]
        
        logger.info("Double-checking stop decision...")
        
        log_path = None
        if self.debug_logging and episode_id is not None and step is not None:
            import time
            log_path = os.path.join(self.log_dir, str(episode_id), f"step_{step}", f"verify_stop_{time.time()}")

        # Call LLM
        response_data, output_text = self._generate_json(
            messages=messages,
            max_new_tokens=8192,
            temperature=self.stop_check_temperature,
            log_path=log_path,
            purpose="Double-check stop",
        )

        logger.info('Double-check response:')
        logger.info(f"{output_text}")

        if response_data is not None:
            decision = response_data.get('decision', 'CONTINUE').upper()
            response_data['stop_evidence'] = self._parse_stop_evidence(response_data)
            return decision == 'STOP', response_data

        # No usable verdict. Returning False keeps the agent moving, which is the
        # safer default (a wrong STOP ends the episode outright, a wrong CONTINUE
        # only costs steps) -- but it is a guess, not a decision, so log it.
        logger.error("Double-check stop: no usable verdict; defaulting to CONTINUE")
        return False, {}

    # Which of the 4 panorama views the double-check named, as an index into the
    # list get_panorama() returns. That list is rotated so index 0 is the 0-degree
    # view and the stride is 90 degrees, which is the same mapping the view
    # labels in double_check_stop are built from.
    STOP_VIEW_INDEX = {'forward': 0, 'left': 1, 'behind': 2, 'right': 3}

    def _parse_stop_evidence(self, response_data):
        """Normalize the double-check's target_view/bbox_2d into a gate-ready dict.

        The gate measures depth at this bbox instead of trusting a stored
        projection, so a malformed box has to come back as None rather than as a
        default full-frame box: a fabricated box would silently measure the
        distance to whatever happens to sit in the middle of the view, and the
        gate would read that as a confident distance. Returning None makes the
        gate fall back to the map-target basis and say so in the trace.
        """
        view = response_data.get('target_view')
        view = view.strip().lower() if isinstance(view, str) else None
        view_idx = self.STOP_VIEW_INDEX.get(view)

        bbox_2d = response_data.get('bbox_2d')
        if view_idx is None or not isinstance(bbox_2d, (list, tuple)) or len(bbox_2d) < 4:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox_2d[:4])
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None

        return {
            'target_view': view,
            'view_idx': view_idx,
            # Kept on the 0-1000 scale the prompt asks for; the evaluator scales
            # to the frame it actually reads, so this stays resolution-agnostic.
            'bbox_norm': [x1, y1, x2, y2],
        }

    def replan_at_backtrack(self, instruction, visited_targets_up_to_backtrack, failed_path, episode_id=None, step=None, history_images=None):
        """
        Re-plan navigation from a backtrack point.
        """
        # 1. Construct History (up to backtrack point)
        history_content = []
        failed_path_content = []

        if history_images is not None:
             # Reserve the 4 current-direction views at the backtrack waypoint. Both
             # history_content and failed_path_content are drawn from history_images,
             # so a single thinning pass bounds the whole request.
             _panorama = visited_targets_up_to_backtrack[-1].get('panorama_frames') \
                 if visited_targets_up_to_backtrack else None
             history_images = self._thin_history_images(
                 history_images, reserved=4 if _panorama else 0
             )

        if history_images is not None:
             # Logic for continuous history images
             # Determine backtrack step (step when the backtrack waypoint was created/visited)
             backtrack_target = visited_targets_up_to_backtrack[-1]
             backtrack_step = backtrack_target['step']
             
             # Group logic similar to navigate_or_backtrack
             sorted_waypoints = sorted(visited_targets_up_to_backtrack, key=lambda x: x['step'])
             last_leg_idx = -2
             
             for item in history_images:
                 img_step = item['step']
                 img = item['image']
                 
                 # Determine if this image is part of history or failed path
                 if img_step <= backtrack_step:
                     target_list = history_content
                     
                     # Add segment header for history
                     leg_idx = -1
                     for i in range(len(sorted_waypoints) - 1):
                         if sorted_waypoints[i]['step'] <= img_step < sorted_waypoints[i+1]['step']:
                             leg_idx = i
                             break
                     if leg_idx == -1 and len(sorted_waypoints) > 0:
                         if img_step >= sorted_waypoints[-1]['step']:
                             leg_idx = len(sorted_waypoints) - 1
                             
                     if leg_idx != last_leg_idx:
                         if leg_idx >= 0:
                             start_id = leg_idx
                             if leg_idx < len(sorted_waypoints) - 1:
                                 end_id = leg_idx + 1
                                 label = f"Waypoint {start_id} -> Waypoint {end_id}: "
                             else:
                                 # For history up to backtrack, the last segment ends at the backtrack waypoint
                                 label = f"Waypoint {start_id} -> Waypoint {start_id} (Backtrack Point): "
                             target_list.append({"type": "text", "text": label})
                         last_leg_idx = leg_idx
                 else:
                     target_list = failed_path_content
                     
                     if not failed_path_content: # First item
                         target_list.append({"type": "text", "text": "Trajectory after Backtrack Point (Failed Path):"})

                 img_data = {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(img)}"
                 }}
                 target_list.append(img_data)
        else:
            for i, target in enumerate(visited_targets_up_to_backtrack):
                 if 'init_image' in target:
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: Arrival view"})
                 if 'dir_image' in target:
                    turn_direction = target.get('direction_decision', '')
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: After turn {turn_direction} view"})
                 if 'description' in target:
                    history_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})

            # 2. Construct Failed Path Description (Images + Text)
            for i, target in enumerate(failed_path):
                idx = len(visited_targets_up_to_backtrack) + i
                if 'init_image' in target:
                    failed_path_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                    failed_path_content.append({"type": "text", "text": f"Waypoint {idx}: Arrival view"})

                if 'dir_image' in target:
                    turn_direction = target.get('direction_decision', '')
                    failed_path_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                    failed_path_content.append({"type": "text", "text": f"Waypoint {idx}: After turn {turn_direction} view"})
                if 'description' in target:
                    failed_path_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})

        if not failed_path_content and not history_images:
             failed_path_content.append({"type": "text", "text": "None (Immediate failure)"})

        # 3. Construct Backtrack Point Views (Current Views) and Action List
        backtrack_point = visited_targets_up_to_backtrack[-1]
        panorama_frames = backtrack_point['panorama_frames']
        
        current_views = []
        view_definitions = [
            {'angle': 0, 'name': 'forward', 'label': 'Current FORWARD view:'},
            {'angle': 90, 'name': 'left', 'label': 'View after turning LEFT:'},
            {'angle': 180, 'name': 'behind', 'label': 'View after turning BEHIND:'},
            {'angle': 270, 'name': 'right', 'label': 'View after turning RIGHT:'}
        ]
        
        # Calculate blocked directions for backtrack point
        blocked_directions = self._check_blocked_directions(panorama_frames)
        
        # Generate Action List
        candidate_directions = ['forward', 'left', 'right', 'behind']
        valid_directions = [d for d in candidate_directions if d not in blocked_directions]
        
        # Fallback if all blocked
        if not valid_directions:
            if 'behind' in candidate_directions: valid_directions.append('behind')
            elif 'forward' in candidate_directions: valid_directions.append('forward')

        action_list_items = []
        action_desc_parts = []
        
        for direction in valid_directions:
            if direction == 'forward':
                action_list_items.append("   - navigate to forward - continue straight ahead")
            elif direction == 'left':
                action_list_items.append("   - navigate to left - turn left and go forward")
            elif direction == 'right':
                action_list_items.append("   - navigate to right - turn right and go forward")
            elif direction == 'behind':
                action_list_items.append("   - navigate to behind - turn around and go forward")
            
            action_desc_parts.append("navigate to " + direction)
            
        action_list_str = "\n".join(action_list_items)
        action_desc_str = "|".join(action_desc_parts)

        # Build current views for prompt
        num_frames = len(panorama_frames)
        for view in view_definitions:
            angle = view['angle']
            if num_frames > 0:
                frame_idx = int((angle / 360.0) * num_frames) % num_frames
            else:
                frame_idx = 0

            if frame_idx < len(panorama_frames):
                rgb_image = panorama_frames[frame_idx]['rgb']
                if isinstance(rgb_image, np.ndarray):
                    if rgb_image.dtype != np.uint8:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_image)
                else:
                    img = rgb_image
                current_views.append({"type": "text", "text": view['label']})
                current_views.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})

        # 4. Construct Prompt
        backtrack_point = visited_targets_up_to_backtrack[-1]
        previous_action = backtrack_point.get('direction_decision', 'unknown')

        negative_constraints = self._negative_constraints(instruction)

        prompt = self.P.REPLAN_PROMPT.format(
            instruction=instruction,
            action_list=action_list_str,
            action_desc=action_desc_str,
            previous_action=previous_action,
            negative_constraints=negative_constraints
        )

        content = [{"type": "text", "text": "Navigation History:"}]
        content.extend(history_content)
        content.append({"type": "text", "text": "\nPrevious Trajectory (Path taken from here):"})
        content.extend(failed_path_content)
        content.append({"type": "text", "text": "\nCurrent 4-directional views at Backtrack Waypoint:"})
        content.extend(current_views)
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        logger.info("Replanning at backtrack point...")
        
        log_path = None
        if self.debug_logging and episode_id is not None and step is not None:
             import time
             log_path = os.path.join(self.log_dir, str(episode_id), f"step_{step}", f"replan_{time.time()}")

        response, output_text = self._generate_json(
            messages=messages,
            max_new_tokens=2048,
            temperature=0.7,
            log_path=log_path,
            purpose="Replan",
        )

        logger.info(f"Replan Output: {output_text}")

        if response is not None:
            try:
                self.stair = response.get('stair', False) # Update stair status

                # Handle stop signal
                stop_signal = response.get('stop', False)
                action = response.get('action', 'navigate to forward').lower()

                # Same template-echo guard as the main nav decision: a string
                # naming every direction would resolve to whichever branch below
                # is tested first.
                if '|' in action or action.rstrip().endswith(' or'):
                    logger.warning(
                        f"Replan: planner echoed the action template instead of choosing "
                        f"({action!r}); going forward."
                    )
                    action = 'navigate to forward'

                if action == 'stop' or 'stop' in action.split():
                    stop_signal = True
                    if action == 'stop':
                        action = 'navigate to forward'

                response['stop_signal'] = stop_signal
                response['action'] = action

                if 'left' in action: return 'left', response
                if 'right' in action: return 'right', response
                if 'behind' in action: return 'behind', response
                return 'forward', response
            except Exception as e:
                logger.error(f"Replan: failed to read a valid action from response: {e}")

        # Falling through means we are about to move without a real decision, at the
        # exact moment the agent had concluded it was going the wrong way. Log loudly
        # so these are countable in the run log rather than invisible.
        logger.error("Replan: no usable decision; falling back to a blind 'forward'")
        return 'forward', {}

    def reset(self):
        self.stair = False
        self.todo.reset()
        self.todo_update_feedback = ""
        self.last_place_summary = ""
        self.last_object = ''

