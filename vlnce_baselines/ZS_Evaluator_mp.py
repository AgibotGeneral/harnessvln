import copy
import gzip
import json
import os
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict

import numpy as np
from PIL import Image
from fastdtw import fastdtw
from skimage.morphology import binary_closing
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm

from habitat import logger
from habitat_extensions.measures import NDTW
from habitat.core.simulator import Observations
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.environments import get_env_class
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry

# Import Habitat visualization utilities

from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.Policy import FusionMapPolicy
from vlnce_baselines.env.env_utils import construct_envs, NoMatchingEpisodesError
from vlnce_baselines.utils.misc import get_device
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.utils.constant import base_classes, map_channels
from .utils.depth_utils import get_world_xz_from_pixel
from .utils.visualization import LaViRAVisualizer

import sys
# Dynamic path resolution for Docker/local compatibility
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

NAVDP_ROOT = os.path.join(PROJECT_ROOT, "navdp")
if NAVDP_ROOT not in sys.path:
    sys.path.append(NAVDP_ROOT)

try:
    from policy_agent import NavDP_Agent
except ImportError as _e:
    logger.warning(f"NavDP import failed ({_e}); --use-navdp will fail loudly when used.")
    NavDP_Agent = None

import warnings

warnings.filterwarnings('ignore')

from .agent import VLMReasoningAgent
from .objnav_categories import load_vocabulary as load_goal_vocabulary
from .harness import (
    DIRECTION_YAW,
    EventStream,
    HarnessRuntime,
    NavigationState,
    Panorama,
    WorldGraph,
    default_registry,
    safe_value,
)
from .harness import world_graph as wg
from .harness import retrieval as retrieval_mod


def _is_fatal_cuda_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "no kernel image is available",
            "invalid device function",
            "cuda out of memory",
            "cuda driver version is insufficient",
            "initialization error",
        )
    )



@baseline_registry.register_trainer(name="ZS-Evaluator-mp")
class ZeroShotVlnEvaluatorMP(BaseTrainer):
    def __init__(self, config, segment_module=None, mapping_module=None) -> None:
        super().__init__()
        self.device = get_device(config.TORCH_GPU_ID)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.config = config
        self.map_args = config.MAP
        self.resolution = config.MAP.MAP_RESOLUTION
        self.width = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        self.height = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
        self.max_step = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.map_shape = (config.MAP.MAP_SIZE_CM // self.resolution,
                          config.MAP.MAP_SIZE_CM // self.resolution)

        self.trans = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(
                                             (self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH),
                                             interpolation=Image.NEAREST)
                                         ])

        self.classes = []
        self.current_episode_id = None
        self.floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversable = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)


        self.visualize = True
        _rgb_base = getattr(config, 'RGB_SAVE_DIR', './saved_rgb_images')
        # Under EVAL_DATA_ROOT run_mp already points RGB_SAVE_DIR at
        # <root>/<exp>/rgb, so it is the final directory. In the legacy layout
        # it is the shared saved_rgb_images/ tree and still needs the exp_name
        # namespacing, or concurrent runs overwrite each other's frames.
        if getattr(config, 'EVAL_DATA_ROOT', ''):
            self.save_dir = _rgb_base
        else:
            _exp_name = self._exp_name()
            self.save_dir = os.path.join(_rgb_base, _exp_name) if _exp_name else _rgb_base
        # When DEBUG_LOGGING is on, also forward the debug_log_dir to the visualizer
        # so per-step topdown_map.png + rgb_obs.png land alongside the model prompts.
        _debug_logging = getattr(config, 'DEBUG_LOGGING', False)
        _debug_log_dir = getattr(config, 'DEBUG_LOG_DIR', 'logs/debug_logs') if _debug_logging else None
        self.visualizer = LaViRAVisualizer(None, self.visualize, self.save_dir, self.width, self.height,
                                           debug_log_dir=_debug_log_dir)
        # Shown as a chip in the per-step frame's title bar.
        try:
            _split = self.config.TASK_CONFIG.DATASET.SPLIT
            self.visualizer.dataset_label = f"{getattr(config, 'TASK_TYPE', 'VLN')} / {_split}"
        except Exception:
            self.visualizer.dataset_label = getattr(config, 'TASK_TYPE', 'VLN')

        self.visited_targets = []  # List of targets the agent has identified/visited
        self.history_images = [] # Initialize history images
        self.current_step = 0  # Track current step for navigation decisions

        # Distance thresholds for target management (in map units)
        # 0.75 meter threshold (75 cm / resolution)
        self.target_reached_threshold = getattr(config, 'TARGET_REACHED_THRESHOLD', 75.0 / self.resolution)
        
        # Statistics counters
        self.total_backtracks = 0
        self.total_waypoints = 0
        # STOP-gate counters.
        self.total_stop_requested = 0
        self.total_stop_rejected = 0
        self.total_stop_gate_pass = 0
        self.total_stop_gate_fail = 0
        self.total_stop_forced = 0
        self.total_stop_cooldown_suppressed = 0
        self.total_autostop = 0
        self.total_target_set = 0
        self.total_target_timeout = 0
        self.total_projection_pullback = 0

        self.harness_enabled = bool(getattr(config.EVAL, 'HARNESS_ENABLED', False))
        self.harness_trace_enabled = bool(getattr(config.EVAL, 'HARNESS_TRACE', True))
        self.harness = None
        if self.harness_enabled:
            self.harness = HarnessRuntime(
                emit=self._emit_event,
                strict_receipts=getattr(config.EVAL, 'HARNESS_STRICT_RECEIPTS', True),
                require_fresh_observation=getattr(
                    config.EVAL, 'HARNESS_REQUIRE_FRESH_OBSERVATION', True),
                # Every tool name reaching the trace is checked against this.
                # A name that is not registered produces a HARNESS_VIOLATION
                # record; it is not refused, because refusing would change
                # navigation. See harness/registry.py for why that line is
                # drawn here.
                registry=default_registry(),
                trace_phases=getattr(config.EVAL, 'HARNESS_TRACE_PHASES', False),
            )

        # Record places, objects and failed branches alongside the navigation waypoints.
        self.world_graph_enabled = bool(getattr(config.EVAL, 'WORLD_GRAPH_ENABLED', True))
        self.world_graph = WorldGraph(
            merge_radius_m=getattr(config.EVAL, 'WORLD_GRAPH_MERGE_RADIUS_M', 0.75),
            object_merge_radius_m=getattr(config.EVAL, 'WORLD_GRAPH_OBJECT_MERGE_RADIUS_M', 1.0),
            same_point_radius_m=getattr(config.EVAL, 'WORLD_GRAPH_SAME_POINT_RADIUS_M', 0.35),
            on_event=self._emit_event,
        ) if self.world_graph_enabled else None
        self.todo_evidence_strict = bool(getattr(config.EVAL, 'TODO_EVIDENCE_STRICT', False))

        # Memory retrieval requires an enabled world graph.
        self.memory_readout_enabled = bool(
            getattr(config.EVAL, 'MEMORY_READOUT_ENABLED', True)) and self.world_graph is not None
        self.memory_readout_seeds = int(getattr(config.EVAL, 'MEMORY_READOUT_SEEDS', 3))
        self.memory_readout_max_places = int(getattr(config.EVAL, 'MEMORY_READOUT_MAX_PLACES', 5))
        self.memory_recency_weight = float(getattr(config.EVAL, 'MEMORY_RECENCY_WEIGHT', 0.3))
        self.memory_salience_weight = float(getattr(config.EVAL, 'MEMORY_SALIENCE_WEIGHT', 0.3))
        self.memory_failure_penalty_per = float(
            getattr(config.EVAL, 'MEMORY_FAILURE_PENALTY_PER', 0.5))
        self.max_backtrack_distance_m = float(
            getattr(config.EVAL, 'MAX_BACKTRACK_DISTANCE_M', 6.0))

        self.use_continuous_history = getattr(config.EVAL, 'USE_CONTINUOUS_HISTORY', True)
        self.history_interval = getattr(config.EVAL, 'HISTORY_INTERVAL', 2)
        self.max_images_per_request = getattr(config.EVAL, 'MAX_IMAGES_PER_REQUEST', 50)

        # The legacy switch bypasses geometric gating and budget autostop.
        self.legacy_stop_enabled = getattr(config.EVAL, 'LEGACY_STOP_ENABLED', False)
        self.stop_geometric_gate_enabled = getattr(config.EVAL, 'STOP_GEOMETRIC_GATE', False)
        self.stop_gate_distance_m = getattr(config.EVAL, 'STOP_GEOMETRIC_GATE_DISTANCE_M', 1.0)
        self.stop_gate_max_target_age_steps = getattr(
            config.EVAL, 'STOP_GEOMETRIC_GATE_MAX_TARGET_AGE_STEPS', 80)
        self.stop_autostop_before_timeout = getattr(config.EVAL, 'STOP_AUTOSTOP_BEFORE_TIMEOUT', False)
        # Default 13, not 1: the trigger is a window and get_panorama() spends 12
        # steps atomically, so a narrower window can be jumped clean over. See
        # the config comment for the ep2701 case this cost.
        self.stop_autostop_reserve_steps = getattr(config.EVAL, 'STOP_AUTOSTOP_RESERVE_STEPS', 38)
        self.stop_autostop_max_attempts = max(
            1, int(getattr(config.EVAL, 'STOP_AUTOSTOP_MAX_ATTEMPTS', 2)))
        # Derive the attempt cost from the panorama step cost plus one STOP action.
        self.panorama_step_cost = 12
        self.autostop_attempt_cost = self.panorama_step_cost + 1
        self.stop_autostop_allow_vlm_override = getattr(
            config.EVAL, 'STOP_AUTOSTOP_ALLOW_VLM_OVERRIDE', True)
        self.stop_gate_distance_basis = getattr(
            config.EVAL, 'STOP_GATE_DISTANCE_BASIS', 'fresh_depth')
        self.stop_gate_depth_distance_m = getattr(
            config.EVAL, 'STOP_GATE_DEPTH_DISTANCE_M', 2.5)
        self.stop_gate_depth_window = getattr(config.EVAL, 'STOP_GATE_DEPTH_WINDOW', 5)
        self.stop_gate_depth_max_std_m = getattr(config.EVAL, 'STOP_GATE_DEPTH_MAX_STD_M', 0.5)
        self.stop_event_stream_enabled = getattr(config.EVAL, 'STOP_EVENT_STREAM', True)
        self.stop_event_flush_every = max(1, int(getattr(config.EVAL, 'STOP_EVENT_FLUSH_EVERY', 1)))
        self.stop_rejection_cooldown_steps = getattr(
            config.EVAL, 'STOP_REJECTION_PANORAMA_COOLDOWN_STEPS', 0)
        # Sampling temperature for the terminal STOP verdict. Logged because it
        # decides whether a round's stop decisions are reproducible at all, and a
        # round comparing against an earlier one has to know which side it ran on.
        self.stop_check_temperature = float(
            getattr(config.EVAL, 'STOP_CHECK_TEMPERATURE', 0.7))
        logger.info(f"stop double-check temperature: {self.stop_check_temperature}")

        # Per-episode stop-gate state; reset() re-initialises these.
        self.last_grounded_target = None
        self.last_stop_rejection_step = None
        # The append-only event stream is created after world_size/local_rank are
        # read from config below, since the filename encodes both.
        self.event_stream = None

        self.config.defrost()
        self.config.MAP.DEVICE = self.config.TORCH_GPU_ID
        self.config.MAP.HFOV = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        self.config.MAP.AGENT_HEIGHT = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
        self.config.MAP.NUM_ENVIRONMENTS = self.config.NUM_ENVIRONMENTS
        self.config.MAP.RESULTS_DIR = self.config.RESULTS_DIR
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.config.freeze()

        self.event_stream = EventStream(
            save_dir=self.save_dir,
            rank=self.local_rank,
            world_size=self.world_size,
            # Late-bound on purpose: the evaluator advances current_episode_id
            # as it goes, and the record must carry the id live at emit time.
            context=lambda: {
                'episode_id': getattr(self, 'current_episode_id', 'unknown'),
                'task_type': getattr(self, 'task_type', None),
            },
            enabled=self.stop_event_stream_enabled,
            flush_every=self.stop_event_flush_every,
            on_error=logger.warning,
        )

        # Initialize agent with specified configuration
        # use_guideline=False, use_working_memory=False as requested
        debug_logging = getattr(config, 'DEBUG_LOGGING', False)
        debug_log_dir = getattr(config, 'DEBUG_LOG_DIR', 'logs/debug_logs')
        self.task_type = getattr(config, 'TASK_TYPE', 'VLN')
        self.use_todo_list = getattr(config.EVAL, 'USE_TODO_LIST', True)
        self.backtrack_second_chance = getattr(config.EVAL, 'BACKTRACK_SECOND_CHANCE', True)
        # Read the goal vocabulary off the episode set itself, so the negative
        # constraints in the prompts name the categories actually being evaluated
        # (21 on MP3D, 6 on hm3d_v2) instead of a hard-coded six.
        goal_vocabulary = load_goal_vocabulary(config.TASK_CONFIG.DATASET.DATA_PATH)
        if self.task_type == "OBJNAV":
            logger.info(f"objnav goal vocabulary: {len(goal_vocabulary)} categories "
                        f"from {config.TASK_CONFIG.DATASET.DATA_PATH}")
        self.agent = VLMReasoningAgent(
            self.visualizer,
            task_type=self.task_type,
            use_guideline=True,
            use_working_memory=False,
            allow_move_behind=False,
            debug_logging=debug_logging,
            log_dir=debug_log_dir,
            use_todo_list=self.use_todo_list,
            backtrack_second_chance=self.backtrack_second_chance,
            max_images_per_request=self.max_images_per_request,
            max_backtrack_distance_m=self.max_backtrack_distance_m,
            goal_vocabulary=goal_vocabulary,
            stop_check_temperature=self.stop_check_temperature,
        )
        # Lets the TODO memory check a COMPLETED claim's evidence ids against the
        # ones the harness actually issued, and route its rejections into the
        # violation stream. Without a harness the memory falls back to shape-only
        # checking; the plan itself works either way.
        if self.harness_enabled and self.harness is not None:
            self.agent.attach_harness(self.harness, strict_evidence=self.todo_evidence_strict)

        # Initialize NavDP Agent
        self.use_navdp = getattr(config, 'USE_NAVDP', False)
        if self.use_navdp:
            if NavDP_Agent is None:
                raise ImportError(
                    "USE_NAVDP=True but NavDP_Agent failed to import. "
                    "Likely missing `diffusers` (see navdp/requirements.txt). "
                    "Run: pip install diffusers==0.33.1"
                )
            intrinsics = self._get_camera_intrinsics()
            model_path = os.path.join(NAVDP_ROOT, "navdp-cross-modal.ckpt")

            self.navdp_agent = NavDP_Agent(
                image_intrinsic=intrinsics,
                navi_model=model_path,
                device=self.device
            )
            self.navdp_agent.reset(batch_size=1, threshold=0.1)

        self.use_fmm = getattr(config, 'USE_FMM', True)

        # Validation Logic. The IPlanner backend was removed from the codebase,
        # so FMM and NavDP are the only controllers left; one of them must be on.
        if not self.use_fmm and not self.use_navdp:
            raise ValueError("Configuration Error: use_fmm and use_navdp cannot both be False.")

    def _init_envs(self) -> None:

        # Check if we have specific allowed episodes loaded from assignment file
        specific_episodes = getattr(self, 'allowed_episodes', None)
        if specific_episodes:
            # self.allowed_episodes is a set of tuples (id, scene_id)
            specific_episodes = list(specific_episodes)

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
            specific_episodes_allowed=specific_episodes,
        )
        logger.info(f"local rank: {self.local_rank}, num of episodes: {self.envs.number_of_episodes}")
        self.detected_classes = OrderedSet()

    def _collect_val_traj(self) -> None:
        """Load trajectory references only for language-navigation benchmarks."""
        self.gt_data = None
        if self.task_type != 'VLN':
            return
        dataset = self.config.TASK_CONFIG.DATASET
        roles = list(getattr(dataset, 'ROLES', ['guide']))
        gt_path = self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
            split=dataset.SPLIT, role=roles[0])
        if os.path.exists(gt_path):
            with gzip.open(gt_path, 'rt') as stream:
                self.gt_data = json.load(stream)
        else:
            logger.warning(f'Trajectory ground truth missing: {gt_path}; NDTW unavailable')

    def _finish_episode(self, step, reason, infos, **fields):
        """Finalize every navigation exit through one trace/metric path."""
        try:
            self._emit_event('TERMINAL', step, terminal_reason=reason)
            self._harness_terminal(step, reason, **fields)
            self._calculate_metric(infos)
        finally:
            self._close_episode_event_stream()

    def _calculate_metric(self, infos: List):
        curr_eps = self.envs.current_episodes()
        info = infos[0]
        ep_id = curr_eps[0].episode_id

        metric = {}
        metric['steps_taken'] = info.get('steps_taken', 0)
        
        # Get success distance from config
        success_distance = getattr(self.config.TASK_CONFIG.TASK, 'SUCCESS_DISTANCE', 
                                 getattr(self.config.TASK_CONFIG.TASK.SUCCESS, 'SUCCESS_DISTANCE', 3.0))

        if self.gt_data is not None and str(ep_id) in self.gt_data:
            gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
            pred_path = np.array(info['position']['position'])
            distances = np.array(info['position']['distance'])
            gt_length = distances[0]
            dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
            
            metric['distance_to_goal'] = distances[-1]
            metric['success'] = 1. if distances[-1] <= success_distance else 0.
            metric['oracle_success'] = 1. if (distances <= success_distance).any() else 0.
            metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
            metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
            metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
            metric['sdtw'] = metric['ndtw'] * metric['success']
        else:
            # Use environment provided metrics if available
            metric['distance_to_goal'] = info.get('distance_to_goal', 0.0)
            metric['success'] = info.get('success', 0.0)
            metric['oracle_success'] = info.get('oracle_success', 0.0)
            metric['path_length'] = info.get('path_length', 0.0)
            metric['spl'] = info.get('spl', 0.0)
            metric['ndtw'] = info.get('ndtw', 0.0)
            metric['sdtw'] = info.get('sdtw', 0.0)
            
        self.state_eps[ep_id] = metric
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)
        logger.info(f'ep{ep_id}:{self.state_eps[ep_id]}')

        # Visualize trajectory
        self.draw_agent_pos_and_ref_path(success=bool(metric['success']))

        # Rename the episode directory with success status
        try:
            is_success = bool(metric['success'])
            status_suffix = "_success" if is_success else "_failed"
            
            # The current save directory for this episode
            current_dir = os.path.join(self.save_dir, str(ep_id))
            
            if os.path.exists(current_dir):
                # The new directory name
                new_dir = os.path.join(self.save_dir, f"{ep_id}{status_suffix}")
                
                if os.path.exists(new_dir):
                    import shutil
                    shutil.rmtree(new_dir)
                
                os.rename(current_dir, new_dir)
                logger.info(f"Renamed episode directory to: {new_dir}")
        except Exception as e:
            logger.info(f"Error renaming episode directory: {e}")

    def _initialize_policy(self) -> None:
        self.segment_module = GroundedSAM(self.config, self.device)
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        self.visualizer.update_map(self.mapping_module)

        self.policy = FusionMapPolicy(self.config, self.mapping_module.map_shape[0])
        self.policy.reset()

    def _concat_obs(self, obs: Observations) -> np.ndarray:
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)  # (h, w, c)->(c, h, w)

        return state

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        state = state.transpose(1, 2, 0)
        rgb = state[:, :, :3].astype(np.uint8)  # [3, h, w]
        rgb = rgb[:, :, ::-1]  # RGB to BGR
        depth = state[:, :, 3:4]  # [1, h, w]
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        env_frame_width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH

        sem_seg_pred = self._get_sem_pred(rgb)  # [num_detected_classes, h, w]
        depth = self._preprocess_depth(depth, min_depth, max_depth)  # [1, h, w]

        """
        ds: Downscaling factor
        args.env_frame_width = 640, args.frame_width = 160
        """
        ds = env_frame_width // self.map_args.FRAME_WIDTH  # ds = 4
        if ds != 1:
            rgb = np.asarray(self.trans(rgb.astype(np.uint8)))  # resize
            depth = depth[ds // 2::ds, ds // 2::ds]  # down scaling start from 2, step=4
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)  # recover depth.shape to (height, width, 1)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)  # (4+num_detected_classes, h, w)

        return state

    def _get_sem_pred(self, rgb: np.ndarray) -> np.ndarray:
        """
        mask.shape=[num_detected_classes, h, w]
        labels looks like: ["kitchen counter 0.69", "floor 0.37"]
        """
        cls2 = self.classes.copy()
        if self.agent.stair:
            cls2 = ["stairs"]
        elif self.current_step <= 12:
            cls2.append('stairs')
        masks, labels, annotated_images, self.current_detections = \
            self.segment_module.segment(rgb, classes=cls2)
        if self.visualize:
            pass
        self.mapping_module.rgb_vis = annotated_images
        assert len(masks) == len(labels), f"The number of masks not equal to the number of labels!"

        class_names = self._process_labels(labels)
        masks = self._process_masks(masks, class_names)

        return masks.transpose(1, 2, 0)

    def _process_labels(self, labels: List[str]) -> List:
        class_names = []
        for label in labels:
            class_name = " ".join(label.split(' ')[:-1])
            class_names.append(class_name)
            self.detected_classes.add(class_name)

        return class_names

    def _process_masks(self, masks: np.ndarray, labels: List[str]):
        """Since we are now handling the open-vocabulary semantic mapping problem,
        we need to maintain a mask tensor with dynamic channels. The idea is to combine
        all same class tensors into one tensor, then let the "detected_classes" to 
        record all classes without duplication. Finally we can use each class's index
        in the detected_classes to determine as it's channel in the mask tensor.
        The organization of mask is similar to chaplot's Sem_Exp, please refer to this link:
        https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/agents/utils/semantic_prediction.py#L41
        
        Args:
            masks (np.ndarray): shape:(c,h,w), each instance(even the same class) has one channel
            labels (List[str]): masks' corresponding labels. len(masks) = len(labels)

        Returns:
            final_masks (np.ndarray): each mask will find their channel in self.detected_classes.
            len(final_masks) = len(self.detected_classes)
        """
        if masks.shape[0] > 0:  # Check if there are any masks
            same_label_indexs = defaultdict(list)
            for idx, item in enumerate(labels):
                same_label_indexs[item].append(idx)  # dict {class name: [idx]}
            combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
            for i, indexs in enumerate(same_label_indexs.values()):
                combined_mask[i] = np.sum(masks[indexs, ...], axis=0)

            idx = [self.detected_classes.index(label) for label in same_label_indexs.keys()]

            """
            max_idx = max(idx) + 1, attention: remember to add one becaure index start from 0
            init final masks as [max_idx + 1, h, w]; add not_a_category channel at last
            """
            final_masks = np.zeros((len(self.detected_classes), *masks.shape[1:]))
            final_masks[idx, ...] = combined_mask
        else:
            final_masks = np.zeros((len(self.detected_classes), self.height, self.width))

        return final_masks

    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        # Preprocesses a depth map by handling missing values, removing outliers, and scaling the depth values.
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99  # turn too far pixels to invalid
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 1.0  # then turn all invalid pixels to vision_range(100)
        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0

        return depth

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        concated_obs = self._concat_obs(obs)
        state = self._preprocess_state(concated_obs)

        return state  # state.shape=(c,h,w)

    def _batch_obs(self, n_obs: List[Observations]) -> Tensor:
        n_states = [self._preprocess_obs(obs) for obs in n_obs]
        max_channels = max([len(state) for state in n_states])
        batch = np.stack([np.pad(state,
                                 [(0, max_channels - state.shape[0]),
                                  (0, 0),
                                  (0, 0)],
                                 mode='constant')
                          for state in n_states], axis=0)

        return torch.from_numpy(batch).to(self.device)




    def _process_one_step_floor(self, one_step_full_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        one_step_full_map = remove_small_objects(one_step_full_map.astype(bool), min_size=64)

        obstacles = one_step_full_map[0, ...].astype(bool)
        explored_area = one_step_full_map[1, ...].astype(bool)

        objects = np.sum(one_step_full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)
        navigable = np.logical_or.reduce(one_step_full_map[map_channels:, ...][navigable_index])
        # stairs should remain navigable even if overlapped with objects
        # navigable = np.logical_or(navigable, stairs_mask)
        navigable = np.logical_and(navigable, np.logical_not(objects))

        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=disk(kernel_size))

        return floor

    def _process_map(self, step: int, full_map: np.ndarray, kernel_size: int = 3) -> tuple:
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        full_map = remove_small_objects(full_map.astype(bool), min_size=64)

        obstacles = full_map[0, ...].astype(bool)
        explored_area = full_map[1, ...].astype(bool)

        objects = np.sum(full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)

        selem = disk(3)
        obstacles_closed = binary_closing(obstacles, footprint=selem)
        objects_closed = binary_closing(objects, footprint=selem)
        navigable = np.logical_or.reduce(full_map[map_channels:, ...][navigable_index])
        # stairs should remain navigable even if overlapped with objects
        # navigable = np.logical_or(navigable, stairs_mask)
        navigable = np.logical_and(navigable, np.logical_not(objects))
        navigable_closed = binary_closing(navigable, footprint=selem)

        untraversable = np.logical_or(objects_closed, obstacles_closed)
        # ensure stairs override untraversable
        untraversable[navigable_closed == 1] = 0
        untraversable = remove_small_objects(untraversable, min_size=64)
        untraversable = binary_closing(untraversable, footprint=disk(3))
        traversable = np.logical_not(untraversable)

        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=selem)
        traversable = np.logical_or(floor, traversable)

        explored_area = binary_closing(explored_area, footprint=selem)
        contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = np.zeros(full_map.shape[-2:], dtype=np.uint8)
        image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)
        frontiers = np.logical_and(floor, image)
        frontiers = remove_small_objects(frontiers.astype(bool), min_size=64)

        return traversable, floor, frontiers.astype(np.uint8)

    def _maps_initialization(self):
        obs = self.envs.reset()  # type(obs): list
        
        # Handle instruction extraction for different tasks (VLN / ObjectNav)
        current_episode = self.envs.current_episodes()[0]
        if 'instruction' in obs[0]:
            self.instruction = obs[0]['instruction']['text']
        elif hasattr(current_episode, 'object_category'):
            self.instruction = f"Find the {current_episode.object_category}"
        elif 'objectgoal' in obs[0]:
            self.instruction = f"Find the object with ID {obs[0]['objectgoal'][0]}"
        else:
            self.instruction = "Explore the environment"
            
        self.destination = getattr(current_episode, 'object_category', None) or "goal"
        self.classes = self.base_classes.copy()
        self.current_episode_id = self.envs.current_episodes()[0].episode_id

        self.visualizer._save_rgb_frame(obs[0], 0, None, self.current_episode_id)

        # Initialize trajectory tracking
        self.pos_list = []
        # Get reference path if available
        self.ref_path = []
        if self.gt_data is not None and str(self.current_episode_id) in self.gt_data:
            self.ref_path = self.gt_data[str(self.current_episode_id)]['locations']

        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        batch_obs = self._batch_obs(obs)
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
        self.mapping_module(batch_obs, poses, self.current_step)
        full_map, full_pose, _ = self.mapping_module.update_map(0, self.detected_classes, self.current_episode_id)
        self.mapping_module.one_step_full_map.fill_(0.)
        self.mapping_module.one_step_local_map.fill_(0.)

        return obs, full_pose



    def reset(self) -> None:
        self._close_episode_event_stream()
        self.classes = []
        self.detected_classes = OrderedSet()
        self.floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversable = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)

        # Reset target tracking
        self.visited_targets = []
        if self.world_graph is not None:
            self.world_graph.reset()
        # total_* counters live on the worker, not the episode (see __init__), so
        # snapshot them here and report current-minus-snapshot in the per-step
        # visualization. world_graph.stats() needs no snapshot: reset() above
        # already clears it per episode.
        self._nav_stats_at_episode_start = self._nav_stats_dict()
        self._last_info = None
        self._last_obs = None
        self._terminal_frame_done = False
        self.history_images = []
        self.current_step = 0
        self.last_grounded_target = None
        self.last_stop_rejection_step = None

        self.policy.reset()
        self.mapping_module.reset()
        self.agent.reset()

    def _get_camera_intrinsics(self) -> np.ndarray:
        """Get camera intrinsics matrix for depth projection"""
        hfov = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        width = self.width
        height = self.height
        vfov = 2 * np.arctan(height / width * np.tan(hfov / 2))

        fx = width / (2 * np.tan(np.deg2rad(hfov / 2)))
        fy = height / (2 * np.tan(np.deg2rad(vfov / 2)))
        cx = width / 2
        cy = height / 2

        intrinsics = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]])
        return intrinsics

    def get_panorama(self, obs: Observations, step: int):
        """Collect a panorama by turning left in 12 steps of 30 degrees each.

        Returns the 4 cardinal frames on success, or a dict carrying
        `episode_finished` if the episode ended mid-spin. Either way
        `_last_panorama_steps_taken` reports how many sim steps were actually
        consumed, which is what callers must add to their step counter -- an
        unconditional +12 desynchronised the autostop window before.
        """

        panorama_frames = []
        self._last_panorama_steps_taken = 0
        self._last_panorama_obs_ids = []

        # Shared with autostop_attempt_cost so the reserve arithmetic and the
        # actual spin cannot disagree.
        for turn_step in range(1, self.panorama_step_cost + 1):
            turn_action = [{"action": HabitatSimActions.TURN_LEFT}]  # 30 deg per step
            turn_outputs = self._harness_step(
                turn_action, step + turn_step - 1,
                tool_name='observe_panorama', reason='panorama_turn',
                metadata={'turn_step': turn_step, 'turn_degrees': 30})
            self._last_panorama_steps_taken = turn_step
            turn_obs, _, turn_dones, turn_infos = [list(x) for x in zip(*turn_outputs)]
            if self.harness_enabled and self.harness is not None:
                self.harness.clear_fresh_observation(step + turn_step, 'observe_panorama')

            if turn_dones[0]:
                # Return signal that episode is done - caller should handle this
                return {'turn_direction': 'episode_done', 'episode_finished': True,
                        'steps_taken': turn_step}

            panorama_frames.append({
                'rgb': turn_obs[0]['rgb'].copy(),
                'depth': turn_obs[0]['depth'].copy(),
                'angle': turn_step * 30 % 360,
                'step': turn_step
            })

            # Update map state for consistency (not persisted).
            batch_obs = self._batch_obs(turn_obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in turn_obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses, self.current_step)
            full_map, full_pose, one_step_full_map = \
                self.mapping_module.update_map(step + turn_step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
        panorama_frames = [panorama_frames[-1]] + panorama_frames[:-1]

        cardinal = panorama_frames[::3]
        # One `ob-` id per cardinal view, in forward/left/behind/right order --
        # the same order the planner is shown them in. `_last_panorama_obs_ids`
        # is what the Place node cites and what a COMPLETED TODO item can point
        # at; it is deliberately parallel to the returned list rather than
        # stuffed into the frame dicts, which are handed to the visualizer and
        # to the planner prompt builder unchanged.
        self._last_panorama_obs_ids = self._harness_observations(
            step, 'panorama_view', len(cardinal))
        return cardinal

    def draw_agent_pos_and_ref_path(self, success: bool = False):
        """
        Visualize agent path, reference path, and target object/viewpoints using Matplotlib.
        """
        if not getattr(self, "visualize", False):
            return None

        pos_list = getattr(self, "pos_list", None) or []
        ref_path = getattr(self, "ref_path", None) or []
        
        # Get target object and viewpoints
        current_episode = self.envs.current_episodes()[0]
        goals = getattr(current_episode, 'goals', [])
        
        # Extract data for plotting
        # Habitat coordinates: Y is up. We visualize X-Z plane (Top-down).
        agent_x = [p[0] for p in pos_list]
        agent_z = [p[2] for p in pos_list]
        
        ref_x = [p[0] for p in ref_path]
        ref_z = [p[2] for p in ref_path]
        
        object_centers = []
        closest_viewpoints = []
        all_viewpoints = []
        radius_list = []
        
        for goal in goals:
            if hasattr(goal, 'position'):
                obj_pos = goal.position
                object_centers.append(obj_pos)
                # Check for radius
                if hasattr(goal, 'radius') and goal.radius is not None:
                     radius_list.append((obj_pos, goal.radius))
            
                # Find closest viewpoint for this goal
                if hasattr(goal, 'view_points') and len(goal.view_points) > 0:
                    best_vp = None
                    min_dist = float('inf')
                    
                    for vp in goal.view_points:
                        if hasattr(vp, 'agent_state') and hasattr(vp.agent_state, 'position'):
                            vp_pos = vp.agent_state.position
                            all_viewpoints.append(vp_pos)
                            # Calculate Euclidean distance in 3D
                            dist = np.linalg.norm(np.array(vp_pos) - np.array(obj_pos))
                            if dist < min_dist:
                                min_dist = dist
                                best_vp = vp_pos
                    
                    if best_vp is not None:
                        closest_viewpoints.append(best_vp)

        # Plotting
        plt.figure(figsize=(10, 8))
        
        # Plot Object Centers
        for obj in object_centers:
            plt.scatter(obj[0], obj[2], c='red', s=200, marker='*', label='Object Center')
            
        # Plot Radius zones
        for center, rad in radius_list:
             circle = plt.Circle((center[0], center[2]), rad, color='green', fill=True, alpha=0.2)
             plt.gca().add_patch(circle)

        # Plot ALL ViewPoints (but circle only closest)
        if all_viewpoints:
            vp_x_all = [vp[0] for vp in all_viewpoints]
            vp_z_all = [vp[2] for vp in all_viewpoints]
            plt.scatter(vp_x_all, vp_z_all, c='green', s=30, alpha=0.6, label='ViewPoints')

        # Draw Success Zones (1.0m) for closest viewpoints only
        if closest_viewpoints:
            vp_x_closest = [vp[0] for vp in closest_viewpoints]
            vp_z_closest = [vp[2] for vp in closest_viewpoints]
            
            for vx, vz in zip(vp_x_closest, vp_z_closest):
                circle = plt.Circle((vx, vz), 1.0, color='green', fill=False, linestyle='--', alpha=0.5)
                plt.gca().add_patch(circle)
        
        # Plot Reference Path
        if ref_x:
            plt.plot(ref_x, ref_z, c='green', linewidth=2, label='Reference Path', linestyle='-')
            # Start and End of Reference Path
            plt.scatter(ref_x[0], ref_z[0], c='orange', s=100, marker='o', label='Ref Start')
            plt.scatter(ref_x[-1], ref_z[-1], c='blue', s=100, marker='X', label='Ref Goal')

        # Plot Agent Trajectory
        if agent_x:
            plt.plot(agent_x, agent_z, c='red', linewidth=2, label='Agent Path', linestyle='-')
            if agent_x:
                plt.scatter(agent_x[0], agent_z[0], c='blue', s=100, marker='^', label='Agent Start')
                plt.scatter(agent_x[-1], agent_z[-1], c='red', s=80, marker='v', label='Agent End')

        plt.title(f'Episode {getattr(self, "current_episode_id", "unknown")} Trajectory')
        plt.xlabel('X (meters)')
        plt.ylabel('Z (meters) - Top Down View')
        
        # Avoid duplicate labels in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')
        
        episode_id = str(getattr(self, "current_episode_id", "unknown"))
        # out_dir = os.path.join(self.save_dir, episode_id)
        # Modified to save to traj_viz directory
        out_dir = os.path.join(self.save_dir, "traj_viz")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"traj_viz_success_{success}_{episode_id}.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved trajectory visualization to: {out_path}")

        return out_path


    # STOP instrumentation and geometry.

    def _emit_event(self, event_type: str, step: int, **fields) -> None:
        """Append one JSONL event for the current episode.

        Thin delegate to :class:`vlnce_baselines.harness.EventStream`, which
        owns the file handle, the flush cadence and the never-raise policy.
        """
        if self.event_stream is not None:
            self.event_stream.emit(event_type, step, **fields)

    def _nav_stats_dict(self) -> dict:
        """Per-worker navigation counters. Single source for both write sites."""
        return {
            'total_backtracks': self.total_backtracks,
            'total_waypoints': self.total_waypoints,
            'total_stop_requested': self.total_stop_requested,
            'total_stop_rejected': self.total_stop_rejected,
            'total_stop_gate_pass': self.total_stop_gate_pass,
            'total_stop_gate_fail': self.total_stop_gate_fail,
            'total_stop_forced': self.total_stop_forced,
            'total_stop_cooldown_suppressed': self.total_stop_cooldown_suppressed,
            'total_autostop': self.total_autostop,
            'total_target_set': self.total_target_set,
            'total_target_timeout': self.total_target_timeout,
            'total_projection_pullback': self.total_projection_pullback,
        }

    def _vis_telemetry(self, distance_to_goal=None) -> dict:
        """Harness state for the per-step visualization side panel.

        The nav counters are per-worker cumulative, so subtract the snapshot taken
        in reset() to get this episode's numbers. Everything here is best-effort:
        the visualizer treats every key as optional.
        """
        try:
            base = getattr(self, '_nav_stats_at_episode_start', None) or {}
            now = self._nav_stats_dict()
            delta = {k: v - base.get(k, 0) for k, v in now.items()}
            graph = self.world_graph.stats() if self.world_graph is not None else {}
            try:
                # habitat hands back numpy scalars; the visualizer wants a float.
                distance_to_goal = None if distance_to_goal is None else float(distance_to_goal)
            except (TypeError, ValueError):
                distance_to_goal = None
            return {
                'max_step': getattr(self, 'max_step', None),
                'distance_to_goal': distance_to_goal,
                'stop': delta,
                'nav': delta,
                'graph': graph,
            }
        except Exception as e:
            logger.warning(f'failed to build visualization telemetry: {e}')
            return {}

    def _close_episode_event_stream(self) -> None:
        """Flush and close the per-episode event file, if one is open."""
        if self.event_stream is not None:
            self.event_stream.close()

    def _harness_start_episode(self, step: int = 0) -> None:
        if self.harness_enabled and self.harness is not None:
            self.harness.start_episode(getattr(self, 'current_episode_id', 'unknown'), step)

    def _harness_logical_tool(self, name: str, step: int, ok: bool = True,
                              args=None, result=None, error=None, reason=None) -> None:
        if not (self.harness_enabled and self.harness is not None and self.harness_trace_enabled):
            return
        call = self.harness.call(
            name, step, args=safe_value(args or {}), read_only=True, reason=reason)
        if not ok:
            self.harness.violation(step, f'{name}_failed', str(error or 'logical tool failed'))
        self.harness.result(
            call, step, ok=ok, result=safe_value(result or {}),
            error=str(error) if error is not None else None)

    def _harness_violation(self, step: int, kind: str, message: str, **fields) -> None:
        """Record a violation, or do nothing when the harness is off."""
        if self.harness_enabled and self.harness is not None:
            self.harness.violation(step, kind, str(message), **safe_value(fields))

    def _harness_terminal(self, step: int, reason: str, **fields) -> None:
        if self.harness_enabled and self.harness is not None:
            self.harness.terminal(step, reason, **safe_value(fields))
        if self.world_graph is not None:
            # One record per episode. Emitted here rather than in the harness
            # block because the graph is built whether or not the harness trace
            # is on, and this aggregate is what report_rounds will read.
            self._emit_event('WORLD_GRAPH_STATS', step, **self.world_graph.stats())
        self._save_terminal_frame(step, reason)

    def _save_terminal_frame(self, step: int, reason: str) -> None:
        """Write one last visualization frame once the episode is decided.

        The per-step frame is drawn at the *top* of the loop, before that
        iteration's plan/decide -- so on the STOP path the last frame shows the
        planner asking to stop while the geometric gate still reads 0 requested: the
        gate runs, and the episode ends, inside the decide that follows. This
        adds a closing frame carrying the resolved gate verdict and terminal
        reason. Never lets a drawing failure take down metric computation.
        """
        if getattr(self, '_terminal_frame_done', False):
            return
        self._terminal_frame_done = True
        obs = getattr(self, '_last_obs', None)
        if obs is None or self.visualizer is None or not self.visualizer.visualize:
            return
        try:
            info = getattr(self, '_last_info', None) or {}
            d2g = info.get('distance_to_goal') if isinstance(info, dict) else None
            telemetry = self._vis_telemetry(d2g)
            telemetry['terminal'] = reason
            todo = self.agent._format_todo_for_prompt() if self.agent.use_todo_list else None
            # Carry over the nav target from the last live frame: without it the
            # map silently loses its green marker on the one frame people stop
            # and read, and the panel would disagree with the frame before it.
            prev = getattr(self.visualizer, '_last_frame', None) or {}
            self.visualizer._save_rgb_frame(
                obs, step, self.visited_targets, self.current_episode_id,
                target_coords=prev.get('target_coords'),
                todo_list=todo, plan_output=self.latest_plan, telemetry=telemetry)
        except Exception as e:
            logger.warning(f'failed to save terminal visualization frame: {e}')

    def _harness_observations(self, step: int, kind: str, count: int, **fields) -> list:
        """Issue `count` ``ob-`` ids for things just seen, or [] without a harness.

        These are what a COMPLETED TODO item cites and what a Place node hangs
        its ``OBSERVED_FROM`` edges off. The images themselves never enter the
        trace -- only the ids.
        """
        if not (self.harness_enabled and self.harness is not None):
            return []
        return self.harness.observations(step, kind, count, **fields)

    def _graph_failure(self, failure_type: str, step: int, **fields) -> None:
        """Record a failed attempt on the current place, if the graph is on."""
        if self.world_graph is None:
            return
        self.world_graph.record_failure(
            failure_type, step, subgoal=self._current_subgoal(), **fields)

    def _current_subgoal(self) -> str:
        """What the agent is currently trying to do, for scoping failures.

        The ACTIVE TODO item when there is one, else the episode instruction.
        Scoping matters: "could not reach the sofa from here" must not count
        against a later search for the TV (world_graph.failure_count).
        """
        todo = getattr(self.agent, 'todo', None)
        active = todo.active() if todo is not None else None
        if active:
            return str(active.get('content', ''))[:120]
        return str(getattr(self, 'instruction', '') or '')[:120]

    def _recall_memory(self, step: int, agent_map_x, agent_map_y,
                       available_waypoint_ids=None) -> str:
        """Run ``recall_place`` and return the planner-facing readout.

        Run once per planner turn without an additional model request.

        Traced as a call plus a result and deliberately **no receipt**:
        ``HarnessRuntime.receipt`` arms the fresh-observation obligation for any
        non-terminal tool, and a read-only lookup must not create an obligation.
        """
        if not self.memory_readout_enabled or self.world_graph is None:
            return ""
        query = self._current_subgoal()
        call = None
        if self.harness_enabled and self.harness is not None and self.harness_trace_enabled:
            call = self.harness.call('recall_place', step, args={'query': query},
                                     read_only=True, reason='planner_turn')
        t0 = time.time()
        try:
            result = retrieval_mod.recall(
                self.world_graph, query,
                current_xz=self._map_cell_to_world_m(agent_map_x, agent_map_y),
                seeds=self.memory_readout_seeds,
                max_places=self.memory_readout_max_places,
                recency_weight=self.memory_recency_weight,
                salience_weight=self.memory_salience_weight,
                failure_penalty_per=self.memory_failure_penalty_per,
                max_backtrack_distance_m=self.max_backtrack_distance_m,
                waypoint_ids_available=available_waypoint_ids,
            )
        except Exception as e:
            # Memory is an optimisation; losing it must not lose the episode.
            logger.error(f"recall_place failed: {e}")
            self._harness_violation(step, 'retrieval_failed', str(e))
            if call is not None:
                self.harness.result(call, step, ok=False, error=str(e))
            return ""
        latency_ms = (time.time() - t0) * 1000.0
        summary = result.summary()
        self._emit_event('MEMORY_READOUT', step, latency_ms=round(latency_ms, 3), **summary)
        if call is not None:
            self.harness.result(call, step, ok=True, result={
                'n_recalled': summary['n_recalled'],
                'n_chars': summary['n_chars'],
                'degraded': summary['degraded'],
            })
        if result.degraded:
            # a degraded ranking must not pass for a working one. Without
            # this, "retrieval quietly stopped matching" and "retrieval does not
            # help" are the same observation.
            self._harness_violation(step, 'retrieval_degraded',
                                    f"recall fell back to recency: {result.degraded_reason}",
                                    reason=result.degraded_reason)
        return result.text

    def _annotate_current_place(self, step: int) -> None:
        """Copy the planner's room description onto the place it describes.

        ``place_summary`` and the ACTIVE TODO id are the two fields that
        cannot be filled at ``observe_place`` time: both are only known after the
        planner has answered about the views it was just shown.
        """
        if self.world_graph is None:
            return
        summary = str(getattr(self.agent, 'last_place_summary', '') or '')
        todo = getattr(self.agent, 'todo', None)
        active = todo.active() if todo is not None else None
        place = self.world_graph.annotate_place(
            semantic_summary=summary,
            todo_progress=(active or {}).get('todo_id'))
        self.agent.last_place_summary = ""
        if place is not None and summary:
            self._emit_event('WORLD_GRAPH_PLACE_SUMMARY', step,
                             place_id=place.place_id, summary=summary[:120])

    def _expire_finished_subgoal_failures(self) -> None:
        """Retire the failure records of TODO items that are no longer pursued.

        Reflections are advisory, so the runtime must be able to
        clear them. The penalty is already scoped per (place, subgoal), which
        handles the planner *switching* sub-goals; this handles the sub-goal
        being finished or dropped, where the records would otherwise sit around
        for a query string nobody asks again.
        """
        if self.world_graph is None:
            return
        todo = getattr(self.agent, 'todo', None)
        if todo is None:
            return
        for entry in getattr(todo, 'last_report', []) or []:
            if not entry.get('ok'):
                continue
            if entry.get('op') == 'remove' or entry.get('status') == 'COMPLETED':
                item = todo.get(entry.get('todo_id') or '')
                if item is None:
                    continue
                n = self.world_graph.expire_failures(subgoal=str(item.get('content', ''))[:120])
                if n:
                    logger.info(f"expired {n} failure record(s) for finished sub-goal "
                                f"{entry.get('todo_id')}")

    def _action_tool_name(self, action_id) -> str:
        if action_id == HabitatSimActions.STOP or action_id == 0:
            return 'stop'
        if action_id == HabitatSimActions.MOVE_FORWARD or action_id == 1:
            return 'move'
        if action_id in (HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT, 2, 3):
            return 'turn'
        return 'action'

    def _harness_step(self, actions, step: int, tool_name: str = None,
                      reason: str = None, metadata=None):
        call = None
        if self.harness_enabled and self.harness is not None and self.harness_trace_enabled:
            action_values = []
            for action in actions:
                if isinstance(action, dict):
                    action_values.append(safe_value(action.get('action')))
                else:
                    action_values.append(safe_value(action))
            call = self.harness.call(
                tool_name or 'action', step,
                args={'actions': action_values, 'metadata': safe_value(metadata or {})},
                read_only=False, reason=reason)
        try:
            outputs = self.envs.step(actions)
        except Exception as e:
            if call is not None:
                self.harness.result(call, step, ok=False, error=str(e))
                self.harness.violation(step, 'env_step_failed', str(e), tool_name=tool_name)
            raise
        if call is not None:
            step_after = step + 1
            self.harness.receipt(
                call, step_before=step, step_after=step_after,
                mutation=tool_name or 'action', metadata=safe_value(metadata or {}))
            self.harness.result(call, step_after, ok=True, result={'step_after': step_after})
        return outputs

    def _record_grounded_target(self, step: int, target_map_x: int, target_map_y: int,
                                target_world_x: float, target_world_y: float,
                                bbox: dict, source: str, planned_action: str) -> None:
        """Persist the newest grounding-grounded target so STOP can be checked against it.

        The local `target_map_x/target_map_y` are cleared the moment the agent
        reaches a target, which is exactly when the STOP double-check runs. This
        instance field is what survives that reset and gives the gate something
        to measure.
        """
        self.last_grounded_target = {
            'step': int(step),
            'map': [int(target_map_x), int(target_map_y)],
            'world': [float(target_world_x), float(target_world_y)],
            'bbox': bbox if isinstance(bbox, dict) else None,
            'source': source,
            'planned_action': planned_action,
        }
        self.total_target_set += 1
        if self.world_graph is not None:
            # Merge grounded objects by category and position across waypoint observations.
            description = (bbox.get('target') if isinstance(bbox, dict) else '') or ''
            self.world_graph.observe_object(
                category=str(description).strip() or 'unknown',
                step=step,
                description=str(description),
                world_xz=(target_world_x, target_world_y),
                map_cell=(target_map_x, target_map_y),
                source=source,
            )
        self._emit_event(
            'TARGET_SET', step,
            target_map=self.last_grounded_target['map'],
            target_world=self.last_grounded_target['world'],
            source=source,
            planned_action=planned_action,
            target_description=(bbox.get('target') if isinstance(bbox, dict) else None),
        )

    def _stop_gate_fresh_depth(self, stop_evidence, panorama_frames):
        """Metric depth at the bbox the double-check just framed.

        The model identifies *what* it sees
        and *where*, the harness measures *how far*. Reading the depth sensor at
        gate time sidesteps both corruptions the stored projection carries --
        the grounding pullback loop that walks a target back along the view ray
        until it lands on traversable space, and the backtrack_replan path that
        projects from a cached panorama at a pose the agent never returned to.

        Returns (distance_m or None, info). None means "no usable reading", and
        the caller falls back to the map-target basis rather than guessing.
        """
        info = {'distance_fresh_depth_m': None}
        if not stop_evidence:
            info['depth_invalid_reason'] = 'no_stop_evidence'
            return None, info
        if not panorama_frames:
            info['depth_invalid_reason'] = 'no_panorama'
            return None, info

        view_idx = stop_evidence.get('view_idx')
        if view_idx is None or not (0 <= view_idx < len(panorama_frames)):
            info['depth_invalid_reason'] = 'view_out_of_range'
            return None, info

        raw_depth = panorama_frames[view_idx].get('depth')
        if raw_depth is None:
            info['depth_invalid_reason'] = 'frame_has_no_depth'
            return None, info

        sensor = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
        min_d = float(getattr(sensor, 'MIN_DEPTH', 0.1))
        max_d = float(getattr(sensor, 'MAX_DEPTH', 5.0))
        # Same conversion as the Step 2 reprojection, so the two bases are in
        # the same units and comparable in the trace. Copy first: these frames
        # are cached on the waypoint and re-read on every rejected STOP.
        try:
            depth_m = self._preprocess_depth(
                np.array(raw_depth, dtype=np.float32, copy=True), min_d, max_d) / 100.0
        except Exception as e:
            info['depth_invalid_reason'] = f'depth_preprocess_error:{e}'
            return None, info

        h, w = depth_m.shape[:2]
        x1, y1, x2, y2 = stop_evidence['bbox_norm']
        # 0-1000 normalized (prompt convention) -> pixels of this frame.
        px1, px2 = x1 / 1000.0 * w, x2 / 1000.0 * w
        py1, py2 = y1 / 1000.0 * h, y2 / 1000.0 * h
        # Anchor exactly where Step 2 anchors its projection (:1722): bottom-edge
        # midpoint, or the top edge on stairs where the far end is what matters.
        anchor_x = int(np.clip((px1 + px2) / 2.0, 0, w - 1))
        anchor_y = int(np.clip(py1 if self.agent.stair else py2, 0, h - 1))

        half = max(1, int(self.stop_gate_depth_window)) // 2
        patch = depth_m[max(0, anchor_y - half):anchor_y + half + 1,
                        max(0, anchor_x - half):anchor_x + half + 1]
        valid = patch[np.isfinite(patch)]
        if valid.size == 0:
            info['depth_invalid_reason'] = 'empty_depth_patch'
            return None, info

        median_m = float(np.median(valid))
        std_m = float(np.std(valid))
        info.update({
            'distance_fresh_depth_m': median_m,
            'depth_window_std_m': std_m,
            'depth_anchor_px': [anchor_x, anchor_y],
            'depth_view': stop_evidence.get('target_view'),
            'depth_bbox_norm': list(stop_evidence['bbox_norm']),
        })

        # _preprocess_depth maps both "invalid" and "too far" pixels onto
        # max_depth, so a reading pinned there is not a distance to anything.
        if median_m >= max_d - 0.05:
            info['depth_invalid_reason'] = 'depth_clipped_at_max'
            return None, info
        if std_m > float(self.stop_gate_depth_max_std_m):
            # The anchor straddles a depth discontinuity -- object edge, doorway
            # -- and the median is not measuring the target.
            info['depth_invalid_reason'] = 'depth_unstable'
            return None, info

        return median_m, info

    def _stop_gate_map_target(self, step: int, current_pose, agent_map_x: int,
                              agent_map_y: int):
        """Distance to the last stored grounded target.

        Retained as the fallback and, more importantly, recorded on every gate
        event next to the fresh-depth reading so the two bases can be compared
        offline without spending another full run.
        """
        info = {'distance_map_target_m': None}
        target = self.last_grounded_target
        if target is None:
            info['map_target_invalid_reason'] = 'no_grounded_target'
            return None, info

        age = int(step) - int(target['step'])
        info.update({
            'target_age_steps': age,
            'target_map': target['map'],
            'target_world': target['world'],
            'target_source': target['source'],
            'target_planned_action': target.get('planned_action'),
        })
        try:
            dx = float(current_pose[0]) - float(target['world'][0])
            dy = float(current_pose[1]) - float(target['world'][1])
            distance_m = float(np.sqrt(dx * dx + dy * dy))
        except Exception as e:
            info['map_target_invalid_reason'] = f'distance_error:{e}'
            return None, info

        info['distance_map_target_m'] = distance_m
        if not np.isfinite(distance_m):
            info['map_target_invalid_reason'] = 'non_finite_distance'
            return None, info
        if age > self.stop_gate_max_target_age_steps:
            info['map_target_invalid_reason'] = 'stale_target'
            return None, info
        return distance_m, info

    def _stop_geometric_gate(self, step: int, current_pose, agent_map_x: int,
                             agent_map_y: int, reason: str,
                             stop_evidence=None, panorama_frames=None):
        """Decide whether the agent is geometrically close enough to STOP.

        Two bases are computed every time and both are written to the trace:
        `fresh_depth` (metric depth at the bbox the double-check framed) and
        `map_target` (distance to the last stored projection). `EVAL.STOP_GATE_DISTANCE_BASIS` picks which one decides;
        the other rides along so the choice can be re-evaluated offline.

        Whichever basis is used, it must never consult `episode.goals`, goal
        view points, `infos[...]['distance_to_goal']` or `sim.geodesic_distance`:
        those are privileged goal state and using them would invalidate the
        zero-shot setting. Depth and the agent's own projections are fair game.

        Fails closed. A wrong STOP ends the episode at 0; a wrong CONTINUE only
        spends steps and can still recover.
        """
        info = {'reason': reason}
        if self.legacy_stop_enabled:
            info['mode'] = 'legacy_bypass'
            return True, info
        if not self.stop_geometric_gate_enabled:
            info['mode'] = 'gate_disabled'
            return True, info

        # Both radii go in the trace unconditionally; `threshold_m` below is set
        # to whichever one the chosen basis is actually judged against, so a
        # reader never has to guess which radius a decision used.
        info['threshold_map_target_m'] = float(self.stop_gate_distance_m)
        info['threshold_fresh_depth_m'] = float(self.stop_gate_depth_distance_m)
        info['agent_map'] = [int(agent_map_x), int(agent_map_y)]
        try:
            info['agent_pose'] = [float(current_pose[0]), float(current_pose[1]),
                                  float(current_pose[2])]
        except Exception:
            info['agent_pose'] = None

        fresh_m, fresh_info = self._stop_gate_fresh_depth(stop_evidence, panorama_frames)
        info.update(fresh_info)
        map_m, map_info = self._stop_gate_map_target(
            step, current_pose, agent_map_x, agent_map_y)
        info.update(map_info)

        # Preference order: the configured basis first, then the other one. The
        # fallback is not a technicality -- both are measured geometry, and
        # refusing to decide because the preferred sensor reading was unusable
        # would send the agent back to burning steps it does not have.
        prefer_fresh = str(self.stop_gate_distance_basis) == 'fresh_depth'
        candidates = ([('fresh_depth', fresh_m), ('map_target', map_m)] if prefer_fresh
                      else [('map_target', map_m), ('fresh_depth', fresh_m)])
        basis, distance_m = next(((b, d) for b, d in candidates if d is not None),
                                (None, None))
        info['distance_basis'] = basis
        if basis is None:
            # Report why the *preferred* basis was unusable; that is the one whose
            # failure modes we are trying to characterise.
            preferred_reason = (info.get('depth_invalid_reason') if prefer_fresh
                                else info.get('map_target_invalid_reason'))
            other_reason = (info.get('map_target_invalid_reason') if prefer_fresh
                            else info.get('depth_invalid_reason'))
            info['fail_reason'] = preferred_reason or other_reason or 'no_usable_distance'
            return False, info

        # Each basis carries its own threshold: map_target measures to a point
        # already pulled back onto standing-height floor, fresh_depth measures to
        # the object surface, and goal view points sit a median 1.57 m back from
        # the object. Sharing one radius would make the depth basis reject stops
        # taken from the best possible standing position.
        threshold = (float(self.stop_gate_depth_distance_m) if basis == 'fresh_depth'
                     else float(self.stop_gate_distance_m))
        info['threshold_m'] = threshold

        info['distance_m'] = distance_m
        if distance_m > threshold:
            info['fail_reason'] = 'distance_exceeds_threshold'
            return False, info
        return True, info

    def _bbox_anchor(self, bbox):
        """The pixel a grounding bounding box is projected from.

        Normally the bottom edge of the box, clamped into the image: that is
        where the object meets the floor, so the depth read there is a distance
        to standable ground. On a staircase the bottom edge is the tread the
        agent is already on, so the top edge is used instead.
        """
        anchor_x = int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0)
        if not self.agent.stair:
            return (anchor_x, min(int(bbox.get('y2', 0)), self.height - 1))
        return (anchor_x, int(bbox.get('y1', 0)))

    def _world_to_map_cell(self, target):
        """World metres (x, z) -> (col, row) map indices, clamped to the map.

        Clamping rather than rejecting is deliberate and pre-existing: a target
        projected off the edge of the map becomes a target on the edge of the
        map, which FMM can still path to.
        """
        cell_x = int(target[0] * 100.0 / self.resolution)
        cell_y = int(target[1] * 100.0 / self.resolution)
        return (max(0, min(cell_x, self.map_shape[0] - 1)),
                max(0, min(cell_y, self.map_shape[1] - 1)))

    def _map_cell_to_world_m(self, cell_x, cell_y):
        """(col, row) map indices -> world metres (x, z). Inverse of the above.

        The world graph merges places by metric distance, so a caller holding
        map cells has to convert rather than pass cells and let a radius in
        metres be compared against a count of 5 cm cells.
        """
        return (float(cell_x) * self.resolution / 100.0,
                float(cell_y) * self.resolution / 100.0)

    def _save_goal_star_viz(self, ob, step, current_pose, target_map_x, target_map_y):
        """Write the depth frame with the current FMM goal marked by a yellow star.

        Diagnostics only: reads no navigation state and writes none. This is the
        single copy of what used to be inlined identically in the Step 2 and
        Step 3 FMM branches, down to the ``navdp_traj_step{step}.png`` filename
        (shared with the NavDP trajectory viz, which overwrites the same path
        when NavDP is the active controller).
        """
        if not self.visualize:
            return
        try:
            viz_depth = self.visualizer._save_depth(ob['depth'], step, use_colormap=False)
            if viz_depth is None:
                return
            if len(viz_depth.shape) == 2:
                viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
            elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)

            intrinsics = self._get_camera_intrinsics()
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            h, w = viz_depth.shape[:2]

            # Goal in the agent's body frame.
            ax = current_pose[0]
            ay = current_pose[1]
            ah = np.deg2rad(current_pose[2])

            tx_viz = target_map_x * self.resolution / 100.0
            ty_viz = target_map_y * self.resolution / 100.0
            dx_viz = tx_viz - ax
            dy_viz = ty_viz - ay
            gx = dx_viz * np.cos(ah) + dy_viz * np.sin(ah)
            gy = -dx_viz * np.sin(ah) + dy_viz * np.cos(ah)
            gz = 0.0

            if gx > 0.1:  # at or behind the image plane -> nothing to draw
                gu = fx * (-gy) / gx + cx
                gv = fy * (self.config.MAP.AGENT_HEIGHT - gz) / gx + cy
                gu, gv = int(gu), int(gv)
                if 0 <= gu < w and 0 <= gv < h:
                    cv2.drawMarker(viz_depth, (gu, gv), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
                    cv2.putText(viz_depth, "Goal", (gu + 10, gv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            episode_dir = os.path.join(self.save_dir, str(self.current_episode_id))
            os.makedirs(episode_dir, exist_ok=True)
            cv2.imwrite(os.path.join(episode_dir, f"navdp_traj_step{step}.png"), viz_depth)
        except Exception as e:
            logger.info(f"Error in FMM viz: {e}")

    def _select_controller(self):
        """Pick the local controller for this step. Shared by Step 2 and Step 3.

        FMM is the default. NavDP takes over either when FMM is disabled
        outright, or -- in hybrid mode -- while the agent is on a staircase,
        where FMM's 2D map has no useful geometry. ``__init__`` rejects
        ``use_fmm=False and use_navdp=False``, so ``use_special`` implies NavDP
        is available.

        Returns:
            (use_special_controller, selected_controller), the latter being
            ``'navdp'`` or ``None``.
        """
        use_special = (not self.use_fmm) or (self.use_navdp and self.agent.stair)
        return use_special, ('navdp' if use_special else None)

    def rollout(self):
        """
        Execute a whole episode using bounding box target navigation
        """
        if self.use_navdp:
            self.navdp_agent.reset_env(0)

        obs, full_pose = self._maps_initialization()

        # Check allowed episodes
        if getattr(self, 'allowed_episodes', None) is not None:
             current_ep = self.envs.current_episodes()[0]
             
             # Robust matching for (id, scene_id) to handle path differences
             is_allowed = False
             for allowed_id, allowed_scene_id in self.allowed_episodes:
                 if str(current_ep.episode_id) == allowed_id:
                     # Check Scene match (basename or suffix)
                     # allowed_scene_id is usually relative, current_ep.scene_id is absolute
                     if current_ep.scene_id.endswith(allowed_scene_id) or \
                        os.path.basename(current_ep.scene_id) == os.path.basename(allowed_scene_id):
                         is_allowed = True
                         break
             
             if not is_allowed:
                 ep_info = f"{current_ep.episode_id} ({os.path.basename(current_ep.scene_id)})"
                 logger.info(f"Skipping episode {ep_info} (Not in assignment)")
                 self._close_episode_event_stream()
                 return



        dones = [False] * self.config.NUM_ENVIRONMENTS
        infos = [{}] * self.config.NUM_ENVIRONMENTS
        self._harness_start_episode(step=0)


        # --- Initialization ---
        self._action = None
        action_list = []
        # Keep navigation state and transitions in one object.
        nav = NavigationState()
        collided = 0
        # Legacy flag, permanently False: FusionMapPolicy._get_action takes it as
        # a positional argument and nothing in this file ever sets it True.
        search_destination = False
        current_pose = full_pose[0] if full_pose is not None else None

        self.latest_plan = {}

        # New: Track consecutive stop failures
        self.stop_feedback = ""
        self.consecutive_stop_failures = 0

        # Target step tracking: re-navigate if the agent has been pursuing the same
        # target for more than max_steps_to_target steps (timeout escape hatch).
        max_steps_to_target = 15

        # The panorama-backed budget autostop gets one shot per episode; the
        # reserve only pays for a single confirming spin.
        autostop_unaffordable_logged = False
        autostop_attempts_made = 0

        full_map = self.mapping_module.get_full_map()
        step = 0
        # Each exit shares the same trace, metrics and stream cleanup path.
        terminal_reason = 'timeout'
        while step < self.max_step:
            pos = self.envs.call_at(0,'_env')._sim.get_agent_state().position
            self.pos_list.append(pos)
            # =================================================================
            # 1. ANALYZE STATE for step N
            #    (consume the result of the previous step's action)
            # =================================================================
            self.visualizer.instruction = self.instruction
            self.visualizer.destination = self.destination
            self.visualizer._action = self._action

            logger.info(f"\nepisode:{self.current_episode_id}, step:{step}")

            self.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)

            # Update pose / position.
            last_pose = current_pose
            current_pose = full_pose[0]
            self.current_step = step
            self.visualizer.sync(step, self.current_episode_id)

            position = current_pose[:2] * 100 / self.resolution
            agent_map_x, agent_map_y = int(position[0]), int(position[1])

            # Reserve steps for geometrically verified STOP attempts before timeout.
            # Each attempt needs a fresh panorama plus the STOP action.
            if (self.stop_autostop_before_timeout
                    and not self.legacy_stop_enabled
                    and not dones[0]
                    and step >= self.max_step - self.stop_autostop_reserve_steps):
                autostop_evidence = None
                autostop_frames = None
                vlm_declined = None
                # 12 for the panorama + 1 for STOP itself, per attempt.
                can_afford_panorama = (self.max_step - step) >= self.autostop_attempt_cost
                # Each attempt needs its own panorama to obtain fresh depth evidence.
                attempts_left = (self.stop_autostop_max_attempts
                                 - autostop_attempts_made)
                if not can_afford_panorama:
                    if not autostop_unaffordable_logged:
                        autostop_unaffordable_logged = True
                        self._emit_event(
                            'AUTOSTOP_UNAFFORDABLE', step,
                            steps_remaining=int(self.max_step - step),
                            confirm_cost=int(self.autostop_attempt_cost),
                            attempts_made=int(autostop_attempts_made),
                            reserve_steps=int(self.stop_autostop_reserve_steps),
                        )
                        logger.info(
                            f"Budget autostop skipped at step {step}: only "
                            f"{self.max_step - step} steps left, confirming "
                            f"panorama needs {self.autostop_attempt_cost}.")
                elif attempts_left > 0:
                    autostop_attempts_made += 1
                    autostop_frames = self.get_panorama(obs[0], step)
                    step += self._last_panorama_steps_taken
                    if isinstance(autostop_frames, dict):
                        terminal_reason = 'env_done_during_panorama'
                        break
                    self._emit_event(
                        'STOP_REQUESTED', step, source='budget_autostop',
                        attempt=int(autostop_attempts_made))
                    self.total_stop_requested += 1
                    try:
                        should_stop, stop_response = self.agent.double_check_stop(
                            self.instruction, autostop_frames, self.visited_targets,
                            episode_id=self.current_episode_id, step=step,
                            history_images=self.history_images)
                        self._harness_logical_tool(
                            'verify_stop', step, ok=True,
                            args={'source': 'budget_autostop',
                                  'attempt': int(autostop_attempts_made)},
                            result={'should_stop': bool(should_stop), 'has_response': stop_response is not None},
                            reason='budget_autostop')
                        self.latest_plan = stop_response
                        autostop_evidence = (stop_response or {}).get('stop_evidence')
                        vlm_declined = not should_stop
                    except Exception as e:
                        logger.error(f"Budget autostop double-check failed: {e}")
                        self._harness_logical_tool(
                            'verify_stop', step, ok=False,
                            args={'source': 'budget_autostop',
                                  'attempt': int(autostop_attempts_made)}, error=e,
                            reason='budget_autostop')
                        vlm_declined = None

                # Skip the verdict without fresh evidence; stored projections may be stale or displaced.
                run_gate = autostop_frames is not None
                if run_gate:
                    gate_ok, gate_info = self._stop_geometric_gate(
                        step=step,
                        current_pose=current_pose,
                        agent_map_x=agent_map_x,
                        agent_map_y=agent_map_y,
                        reason='budget_autostop',
                        stop_evidence=autostop_evidence,
                        panorama_frames=autostop_frames,
                    )
                    gate_info['attempt'] = int(autostop_attempts_made)
                    if vlm_declined is not None:
                        gate_info['vlm_declined'] = bool(vlm_declined)
                        # Budget autostop may override visual confirmation when configured.
                        # Record the override separately from normal verification.
                        if vlm_declined and not self.stop_autostop_allow_vlm_override:
                            gate_ok = False
                            gate_info['fail_reason'] = 'vlm_declined'

                    if gate_ok:
                        # Any queued turn/forward would consume the last slot and make
                        # STOP impossible, so drop them.
                        action_list.clear()
                        self._action = 0
                        outputs = self._harness_step(
                            [{"action": 0}], step, tool_name='stop', reason='budget_autostop',
                            metadata=gate_info)
                        step += 1
                        obs, _, dones, infos = [list(x) for x in zip(*outputs)]
                        self.total_autostop += 1
                        self._emit_event('AUTOSTOP', step, **gate_info)
                        logger.info(
                            f"Budget autostop at step {step}: distance="
                            f"{gate_info.get('distance_m')} basis={gate_info.get('distance_basis')} "
                            f"threshold={gate_info.get('threshold_m')} "
                            f"vlm_declined={gate_info.get('vlm_declined')}")
                        self._finish_episode(step, 'budget_autostop', infos, gate_info=gate_info)
                        return
                    self._emit_event('AUTOSTOP_SKIPPED', step, **gate_info)

            # Prepare NavDP trajectory for visualization, if available.
            navdp_traj_list = None
            if getattr(self, 'navdp_traj_global', None) is not None:
                navdp_traj_list = [(pt[0], pt[1]) for pt in self.navdp_traj_global]

            _last_info = getattr(self, '_last_info', None) or {}
            _d2g = _last_info.get('distance_to_goal') if isinstance(_last_info, dict) else None
            # kept for the closing frame drawn from _harness_terminal
            self._last_obs = obs[0]

            self.visualizer._save_rgb_frame(obs[0], step, self.visited_targets, self.current_episode_id, (nav.target_map_x, nav.target_map_y), todo_list=self.agent._format_todo_for_prompt() if self.agent.use_todo_list else None, plan_output=self.latest_plan, navdp_traj=navdp_traj_list, telemetry=self._vis_telemetry(_d2g))

            # Save continuous history images
            if self.use_continuous_history and step % self.history_interval == 0:
                rgb_to_save = obs[0]['rgb'].copy()
                if isinstance(rgb_to_save, np.ndarray):
                    if rgb_to_save.dtype != np.uint8:
                        rgb_to_save = (rgb_to_save * 255).astype(np.uint8)
                    img_save = Image.fromarray(rgb_to_save)
                else:
                    img_save = rgb_to_save
                
                self.history_images.append({'step': step, 'image': img_save})

            # =================================================================
            # 2. PLAN/DECIDE for step N (4-step navigation cycle):
            #    Step 1: no target  -> spin to collect panorama
            #    Step 2: panorama done -> query LLM for the next target
            #    Step 3: have target -> navigate toward it
            #    Step 4: target reached or timed out -> reset and go back to Step 1
            # =================================================================

            # High-level decisions only happen when action_list is empty.
            if not action_list:
                # Target timeout check.
                if nav.has_target and nav.target_set_step is not None:
                    steps_since_target_set = nav.steps_since_target_set(step)
                    if steps_since_target_set >= max_steps_to_target:
                        if nav.backtracking_only:
                            logger.info("Backtrack-only return timed out, resetting navigation state.")

                        self.total_target_timeout += 1
                        self._emit_event(
                            'TARGET_TIMEOUT', step,
                            target_map=[int(nav.target_map_x), int(nav.target_map_y)],
                            agent_map=[int(agent_map_x), int(agent_map_y)],
                            steps_since_target_set=int(steps_since_target_set),
                            max_steps_to_target=int(max_steps_to_target),
                            backtracking_only=bool(nav.backtracking_only),
                        )

                        if len(self.visited_targets) > 0:
                            self.visited_targets.pop()

                        # Retain the failed branch in the graph after removing its navigation waypoint.
                        self._graph_failure(
                            wg.FAIL_NO_PROGRESS, step, detail='target_timeout')

                        # Step 4: reset state, prepare to restart. Deliberately
                        # leaves is_backtracking and planned_action alone --
                        # see NavigationState.on_target_timeout.
                        nav.on_target_timeout()

                # Distance-to-target check.
                if nav.has_target:
                    distance_to_target = np.sqrt((nav.target_map_x - agent_map_x) ** 2 + (nav.target_map_y - agent_map_y) ** 2)

                    # Determine threshold based on navigation type
                    # For backtracking, use same threshold as normal nav (or specific if needed)
                    # User requested: Normal 0.75m, Backtrack 0.75m. So just use self.target_reached_threshold (which is 0.75m)
                    current_threshold = self.target_reached_threshold

                    if distance_to_target < current_threshold:
                        if nav.backtracking_only:
                            logger.info("Reached backtrack waypoint. Returning to normal decision flow.")

                        # Deduplicate waypoints that ended up too close to the just-reached one.
                        if len(self.visited_targets) > 0:
                            dist_calc = lambda target: np.sqrt(
                                (target['world_coords'][0] - self.visited_targets[-1]['world_coords'][0]) ** 2 + (
                                            target['world_coords'][1] - self.visited_targets[-1]['world_coords'][
                                        1]) ** 2) if 'world_coords' in target else float('inf')
                            for target in self.visited_targets[:-1]:
                                if dist_calc(target) < self.target_reached_threshold:
                                    self.visited_targets.pop()
                                    break

                        # Step 4: reset state, prepare for next target.
                        nav.on_target_reached()

                # Step 1: no target and not spinning yet -> spin to collect panorama and decide direction.
                if nav.target_map_x is None and not nav.panorama_got and nav.going_to_stop:
                    # Double-check stop: acquire a fresh panorama and verify with the planner model.
                    panorama_frames = self.get_panorama(obs[0], step)
                    step += self._last_panorama_steps_taken

                    # get_panorama returns a dict only on the episode-done path;
                    # the normal return is a list of frames. Test the type, not
                    # membership -- `'x' in [list]` silently scans elements.
                    if isinstance(panorama_frames, dict):
                        terminal_reason = 'env_done_during_panorama'
                        break

                    self._emit_event('STOP_REQUESTED', step, source='double_check')
                    should_stop, stop_response = self.agent.double_check_stop(self.instruction, panorama_frames, self.visited_targets, episode_id=self.current_episode_id, step=step, history_images=self.history_images)
                    self._harness_logical_tool(
                        'verify_stop', step, ok=True,
                        args={'source': 'double_check'},
                        result={'should_stop': bool(should_stop), 'has_response': stop_response is not None},
                        reason='double_check')
                    self.latest_plan = stop_response
                    self.total_stop_requested += 1

                    # Use the model for visibility and category, and sensor depth for distance.
                    gate_ok, gate_info = (True, {'reason': 'double_check_stop', 'mode': 'legacy_bypass'})
                    if should_stop:
                        gate_ok, gate_info = self._stop_geometric_gate(
                            step=step,
                            current_pose=current_pose,
                            agent_map_x=agent_map_x,
                            agent_map_y=agent_map_y,
                            reason='double_check_stop',
                            stop_evidence=(stop_response or {}).get('stop_evidence'),
                            panorama_frames=panorama_frames,
                        )
                        if gate_ok:
                            self.total_stop_gate_pass += 1
                            self._emit_event('STOP_GATE_PASS', step, **gate_info)
                        else:
                            self.total_stop_gate_fail += 1
                            self._emit_event('STOP_GATE_FAIL', step, **gate_info)
                            logger.info(
                                f"STOP rejected by geometric gate: {gate_info.get('fail_reason')} "
                                f"(distance={gate_info.get('distance_m')}, "
                                f"threshold={gate_info.get('threshold_m')})")
                            should_stop = False

                    if should_stop:
                        action_list.append(0)  # STOP action
                        self.consecutive_stop_failures = 0 # Success
                    else:
                        # Rolls back the stop intent and the panorama progress so
                        # Step 1 is re-entered; the current target survives.
                        nav.on_stop_rejected()

                        # A rejected STOP is the agent having been convinced it
                        # had arrived and been wrong -- the single most useful
                        # thing to remember about this place.
                        self._graph_failure(
                            wg.FAIL_STOP_REJECTED, step,
                            detail=str(gate_info.get('fail_reason') or 'double_check_declined'))

                        # Handle rejection feedback
                        self.consecutive_stop_failures += 1
                        self.total_stop_rejected += 1
                        rejected_by = 'geometric_gate' if not gate_ok else 'double_check'
                        self.last_stop_rejection_step = step
                        if rejected_by == 'geometric_gate':
                            measured = gate_info.get('distance_m')
                            if isinstance(measured, float):
                                distance_note = (
                                    f" Measured distance to the last grounded target was "
                                    f"{measured:.2f} m, and STOP requires "
                                    f"{gate_info.get('threshold_m')} m or closer.")
                            else:
                                distance_note = (
                                    f" There is no recent grounded target to verify against "
                                    f"({gate_info.get('fail_reason')}).")
                            self.stop_feedback = (
                                f"System: Your previous decision to STOP was rejected because the agent "
                                f"is not geometrically close enough to the target (Failure count: "
                                f"{self.consecutive_stop_failures})."
                                f"{distance_note} Navigate closer to the target before requesting STOP again.")
                        else:
                            self.stop_feedback = f"System: Your previous decision to STOP was rejected by the double-check mechanism because the target was not clearly visible or not close enough (Failure count: {self.consecutive_stop_failures}). Please navigate closer to the target or verify your position."
                        self._emit_event(
                            'STOP_REJECTED', step,
                            source=rejected_by,
                            consecutive_stop_failures=self.consecutive_stop_failures,
                            **{k: v for k, v in gate_info.items() if k != 'reason'},
                        )

                        # No forced STOP here on purpose. Terminating after N
                        # rejections converted 7 correct rejections into scored-0
                        # episodes, all of them >=1.24 m from the goal. The agent
                        # keeps navigating instead; if it really is in position, the
                        # gate or the budget autostop will let it stop.
                        if self.legacy_stop_enabled and self.consecutive_stop_failures >= 3:
                            logger.info(f"Forcing STOP after {self.consecutive_stop_failures} consecutive double-check rejections.")
                            action_list.append(0) # STOP action
                            self.total_stop_forced += 1
                            self._emit_event('STOP_FORCED', step, mode='legacy')
                            # Reset flags to prevent loop
                            self.stop_feedback = ""
                            self.consecutive_stop_failures = 0

                elif nav.target_map_x is None and not nav.navigate_or_not:
                    # Step 1: collect panorama and decide direction.
                    current_rgb = obs[0]['rgb'].copy()

                    panorama_frames = self.get_panorama(obs[0], step)
                    step += self._last_panorama_steps_taken

                    if isinstance(panorama_frames, dict):
                        terminal_reason = 'env_done_during_panorama'
                        break

                    # Create a new waypoint anchored at the current map position.
                    waypoint_id = len(self.visited_targets)
                    self.visited_targets.append({
                        'step': step,
                        'init_image': Image.fromarray(current_rgb) if isinstance(current_rgb,
                                                                                 np.ndarray) else current_rgb,
                        'panorama_frames': panorama_frames,
                        'world_coords': (agent_map_x, agent_map_y),
                        'full_pose': current_pose,  # full (x, y, heading)
                        # The `ob-` ids of the four cardinal frames, in the order
                        # they are shown to the planner. Carried on the waypoint so
                        # the prompt can label each view with the id that a
                        # COMPLETED TODO item is supposed to cite.
                        'obs_ids': list(getattr(self, '_last_panorama_obs_ids', []) or []),
                    })
                    self.total_waypoints += 1

                    # Convert waypoint map cells to metres before merging graph nodes.
                    if self.world_graph is not None:
                        self.world_graph.observe_place(
                            self._map_cell_to_world_m(agent_map_x, agent_map_y),
                            step=step,
                            map_cell=(agent_map_x, agent_map_y),
                            heading=(float(current_pose[2]) if current_pose is not None else None),
                            observation_ids=list(getattr(self, '_last_panorama_obs_ids', []) or []),
                            waypoint_id=waypoint_id,
                        )

                    self.visualizer._save_waypoint_panorama_rgb(panorama_frames, waypoint_id, step)
                    # Combine stop feedback and todo verification feedback
                    combined_feedback = self.stop_feedback
                    todo_feedback = getattr(self.agent, 'todo_verification_feedback', "")
                    if todo_feedback:
                        combined_feedback = (combined_feedback + "\n" + todo_feedback).strip()
                        # Consume the feedback (don't repeat it forever)
                        self.agent.todo_verification_feedback = ""
                    # Same channel for the update-level rejections (unknown id,
                    # second ACTIVE, COMPLETED without evidence). Telling the
                    # planner is the point: a silently dropped update looks, from
                    # its side, exactly like one that was accepted and forgotten.
                    todo_update_feedback = getattr(self.agent, 'todo_update_feedback', "")
                    if todo_update_feedback:
                        combined_feedback = (combined_feedback + "\n" + todo_update_feedback).strip()
                        self.agent.todo_update_feedback = ""

                    # Recall after recording the current place and before planning.
                    # Only expose backtrack IDs that the action filter accepts.
                    memory_readout = self._recall_memory(
                        step, agent_map_x, agent_map_y,
                        available_waypoint_ids=self.agent.offered_backtrack_ids(
                            self.visited_targets))

                    decision = self.agent.navigate_or_backtrack(
                        instruction=self.instruction,
                        visited_targets=self.visited_targets,
                        feedback=combined_feedback,
                        episode_id=self.current_episode_id,
                        step=step,
                        history_images=self.history_images if self.use_continuous_history else None,
                        memory_readout=memory_readout,
                    )
                    # The planner has now named the room it is standing in and
                    # which TODO item it is on: both belong on the place node, and
                    # neither existed when observe_place ran.
                    self._annotate_current_place(step)
                    # Its TODO updates have been applied by now, so any sub-goal
                    # that just finished can stop costing its places a penalty.
                    self._expire_finished_subgoal_failures()
                    self._harness_logical_tool(
                        'plan_navigation', step, ok=isinstance(decision, dict),
                        args={'visited_targets': len(self.visited_targets), 'feedback': bool(combined_feedback)},
                        result={
                            'action': decision.get('action') if isinstance(decision, dict) else None,
                            'direction': decision.get('direction') if isinstance(decision, dict) else None,
                            'waypoint': decision.get('waypoint') if isinstance(decision, dict) else None,
                        },
                        reason='navigate_or_backtrack')
                    self.latest_plan = decision

                    # If planner decides to NAVIGATE (and not STOP), clear the feedback
                    if decision.get('action') == 'NAVIGATE' or decision.get('action') == 'BACKTRACK':
                        self.stop_feedback = ""
                        # If we had stop failures but now planner decided to move, we can reset the counter
                        # self.consecutive_stop_failures = 0 # Optional: reset or keep counting? Reset seems fair.
                        if decision.get('action') == 'NAVIGATE':
                             # Only reset if it's a new navigation decision, backtrack might still be "stuck" logic
                             pass

                    if decision.get('action') == 'STOP':
                        # Cooldown after a rejected STOP. Without the forced-STOP
                        # escape hatch, an planner that keeps asking to stop would spend
                        # the budget on reject -> re-spin panorama (12 steps) ->
                        # reject cycles instead of closing the remaining distance.
                        if (not self.legacy_stop_enabled
                                and self.last_stop_rejection_step is not None
                                and (step - self.last_stop_rejection_step)
                                < self.stop_rejection_cooldown_steps):
                            self.total_stop_cooldown_suppressed += 1
                            self._emit_event(
                                'STOP_COOLDOWN_SUPPRESSED', step,
                                steps_since_rejection=step - self.last_stop_rejection_step,
                                cooldown_steps=self.stop_rejection_cooldown_steps,
                            )
                            logger.info(
                                "planner requested STOP during post-rejection cooldown; "
                                "continuing to navigate instead.")
                            decision['action'] = 'NAVIGATE'
                            nav.planned_action = 'NAVIGATE'
                        else:
                            nav.planned_action = 'STOP'

                    if decision.get('action', 'NAVIGATE') == 'BACKTRACK':
                        self.total_backtracks += 1
                        target_waypoint_id = decision.get('waypoint', 0)
                        if isinstance(target_waypoint_id, int) and target_waypoint_id < len(self.visited_targets) - 1:
                            visited_targets_up_to_backtrack = self.visited_targets[:target_waypoint_id + 1]
                            failed_path = self.visited_targets[target_waypoint_id + 1:]
                            backtrack_point = visited_targets_up_to_backtrack[-1]

                            if self.agent.backtrack_second_chance:
                                logger.info(f"Initiating Backtrack Re-planning to Waypoint {target_waypoint_id}")

                                # 2. Replan
                                new_direction, replan_response = self.agent.replan_at_backtrack(
                                    self.instruction,
                                    visited_targets_up_to_backtrack,
                                    failed_path,
                                    episode_id=self.current_episode_id,
                                    step=step,
                                    history_images=self.history_images if self.use_continuous_history else None
                                )
                                self.latest_plan = replan_response
                                self._harness_logical_tool(
                                    'plan_navigation', step, ok=True,
                                    args={'source': 'backtrack_replan', 'visited_targets': len(visited_targets_up_to_backtrack)},
                                    result={'direction': new_direction, 'has_response': replan_response is not None},
                                    reason='backtrack_replan')
                                logger.info(f"Re-planned direction: {new_direction}")

                                # 3. Get corresponding image from backtrack point
                                panorama_frames = backtrack_point['panorama_frames']
                                # Use the shared legacy panorama index convention.
                                frame_idx = Panorama(panorama_frames).legacy_index(
                                    DIRECTION_YAW.get(new_direction, 0))

                                if frame_idx < len(panorama_frames):
                                    # 4. Get BBox from grounding model
                                    dir_rgb = panorama_frames[frame_idx]['rgb']
                                    dir_depth = panorama_frames[frame_idx]['depth']

                                    bbox = self.agent.query_llm(
                                        instruction=self.instruction,
                                        visited_targets=visited_targets_up_to_backtrack,
                                        rgb_image=dir_rgb,
                                        depth_image=dir_depth,
                                        width=self.width,
                                        height=self.height,
                                        current_step=self.current_step,
                                        progress_analysis="Backtracking and Replanning",
                                        episode_id=self.current_episode_id
                                    )

                                    bbox_save_path = os.path.join(self.save_dir, str(self.current_episode_id), f'bbox_step{step}.png')
                                    os.makedirs(os.path.dirname(bbox_save_path), exist_ok=True)

                                    d_img = dir_depth
                                    if len(d_img.shape) == 3:
                                        d_img = d_img[:, :, 0]

                                    dummy_depth = np.expand_dims(d_img, axis=2) # (H,W,1)
                                    processed_depth = self._preprocess_depth(dummy_depth, 0.1, 5.0) / 100.0

                                    coords = self._bbox_anchor(bbox)

                                    base_pose = backtrack_point.get('full_pose')
                                    if base_pose is None:
                                        nav.target_map_x, nav.target_map_y = backtrack_point['world_coords']
                                    else:
                                        frame_angle = panorama_frames[frame_idx]['angle']

                                        if isinstance(base_pose, torch.Tensor):
                                            view_pose = base_pose.clone().cpu().numpy()
                                        else:
                                            view_pose = base_pose.copy()

                                        if view_pose.ndim == 2:
                                            view_pose = view_pose[0]

                                        view_pose[2] += frame_angle
                                        view_pose[2] = view_pose[2] % 360

                                        target = get_world_xz_from_pixel(
                                            pixel_coords=coords,
                                            depth_image=processed_depth,
                                            full_pose=view_pose,
                                            camera_intrinsics=self._get_camera_intrinsics(),
                                        )

                                        new_target_x, new_target_y = self._world_to_map_cell(target)

                                        nav.target_map_x, nav.target_map_y = new_target_x, new_target_y
                                        self._record_grounded_target(
                                            step=step,
                                            target_map_x=nav.target_map_x,
                                            target_map_y=nav.target_map_y,
                                            target_world_x=target[0],
                                            target_world_y=target[1],
                                            bbox=bbox,
                                            source='backtrack_replan',
                                            planned_action=nav.planned_action,
                                        )

                                revisit_waypoint = {
                                    'step': step,
                                    'init_image': backtrack_point['init_image'],
                                    'panorama_frames': backtrack_point['panorama_frames'],
                                    'world_coords': backtrack_point['world_coords'],
                                    'full_pose': backtrack_point['full_pose'],
                                    'is_revisit': True,
                                    'original_waypoint_id': target_waypoint_id,
                                    'backtrack_from_waypoint_id': len(self.visited_targets) - 1
                                }

                                self.visited_targets.append(revisit_waypoint)
                                self.total_waypoints += 1

                                self.visited_targets[-1]['direction_decision'] = new_direction
                                self.visited_targets[-1]['dir_image'] = Image.fromarray(dir_rgb) if isinstance(dir_rgb, np.ndarray) else dir_rgb
                                self.visited_targets[-1]['bbox'] = bbox
                                self.visited_targets[-1]['progress_analysis'] = "Backtracking Re-plan"

                                nav.is_backtracking = False # Treat as new target
                                nav.backtracking_only = False
                            else:
                                logger.info(f"Initiating Backtrack-only return to Waypoint {target_waypoint_id}")
                                if backtrack_point.get('world_coords') is not None:
                                    nav.target_map_x, nav.target_map_y = backtrack_point['world_coords']
                                else:
                                    decision['action'] = 'NAVIGATE'
                                    nav.panorama_got = True
                                    continue

                                revisit_waypoint = {
                                    'step': step,
                                    'init_image': backtrack_point['init_image'],
                                    'panorama_frames': backtrack_point['panorama_frames'],
                                    'world_coords': backtrack_point['world_coords'],
                                    'full_pose': backtrack_point['full_pose'],
                                    'is_revisit': True,
                                    'original_waypoint_id': target_waypoint_id,
                                    'backtrack_from_waypoint_id': len(self.visited_targets) - 1,
                                    'progress_analysis': "Backtracking Only"
                                }

                                self.visited_targets.append(revisit_waypoint)
                                self.total_waypoints += 1
                                nav.is_backtracking = True
                                nav.backtracking_only = True

                            nav.target_set_step = step

                        else:
                            decision['action'] = 'NAVIGATE'
                        nav.panorama_got = True
                    if decision.get('action', 'NAVIGATE') == 'NAVIGATE':
                        nav.navigate_or_not = True
                        direction = decision.get('direction', 'forward')

                        if decision.get('stop_signal', False):
                            # Same post-rejection cooldown as the explicit STOP action
                            # above; stop_signal is the other way a STOP intent enters.
                            if (not self.legacy_stop_enabled
                                    and self.last_stop_rejection_step is not None
                                    and (step - self.last_stop_rejection_step)
                                    < self.stop_rejection_cooldown_steps):
                                self.total_stop_cooldown_suppressed += 1
                                self._emit_event(
                                    'STOP_COOLDOWN_SUPPRESSED', step,
                                    source='stop_signal',
                                    steps_since_rejection=step - self.last_stop_rejection_step,
                                    cooldown_steps=self.stop_rejection_cooldown_steps,
                                )
                                nav.planned_action = 'NAVIGATE'
                            else:
                                nav.planned_action = 'STOP'
                        else:
                            nav.planned_action = 'NAVIGATE'

                        progress_analysis = decision.get('progress_analysis', '')
                        reasoning = decision.get('reasoning', '')

                        self.visited_targets[-1].update({
                            'progress_analysis': progress_analysis,
                            'reasoning': reasoning,
                            'direction_decision': direction
                        })

                        # Pull the panorama frame matching the chosen direction for visualization.
                        frame_idx = Panorama(panorama_frames).legacy_index(
                            DIRECTION_YAW.get(direction, 0))

                        if frame_idx < len(panorama_frames):
                            dir_rgb = panorama_frames[frame_idx]['rgb']
                            self.visited_targets[-1]['dir_image'] = Image.fromarray(dir_rgb) if isinstance(dir_rgb,
                                             np.ndarray) else dir_rgb
                            self.visited_targets[-1]['turn_action'] = f"turn {direction}"

                        # Each TURN_* action is 30 deg; chain them to reach the target heading.
                        if direction == 'left':
                            action_list.extend([2] * 3)  # TURN_LEFT 90 deg (3 * 30 deg)
                        elif direction == 'right':
                            action_list.extend([3] * 3)  # TURN_RIGHT 90 deg
                        elif direction == 'behind':
                            action_list.extend([2] * 6)  # TURN_LEFT 180 deg (6 * 30 deg)
                        # forward needs no turn

                        nav.panorama_got = True

                # Step 2: panorama done, no queued actions -> ask the grounding for a target bbox.
                elif nav.target_map_x is None and nav.panorama_got and not action_list:
                    progress_analysis = self.visited_targets[-1].get('progress_analysis', '')

                    bbox = self.agent.query_llm(
                        instruction=self.instruction,
                        visited_targets=self.visited_targets,
                        rgb_image=obs[0]['rgb'],
                        depth_image=obs[0]['depth'],
                        width=self.width,
                        height=self.height,
                        current_step=self.current_step,
                        progress_analysis=progress_analysis,
                        planned_action=nav.planned_action,
                        episode_id=self.current_episode_id
                    )
                    self._harness_logical_tool(
                        'ground_target', step, ok=isinstance(bbox, dict),
                        args={'planned_action': nav.planned_action},
                        result={'has_bbox': isinstance(bbox, dict), 'bbox': bbox if isinstance(bbox, dict) else None},
                        reason='query_llm')

                    self.visited_targets[-1].update({
                        'description': bbox.get('target', 'unknown target'),
                        'bbox': bbox,
                        'llm_reasoning': bbox.get('reasoning', ''),
                        'llm_progress': bbox.get('progress', '')
                    })

                    # If the current action is STOP, we are approaching the target.
                    # Once we reach this target (in Step 4), we should then trigger the double check.
                    if nav.planned_action == 'STOP':
                        nav.going_to_stop = True

                    # Convert the grounding bbox into a world coordinate target.
                    depth_image = self._preprocess_depth(obs[0]['depth'], 0.1, 5.0) / 100.0
                    coords = self._bbox_anchor(bbox)

                    # Pixel coords -> world coords (with retry loop, see body below).
                    pullback_count = 0
                    while True:
                        self._harness_logical_tool(
                            'project_target', step, ok=True,
                            args={'coords': coords, 'pullback_count': pullback_count},
                            result={'depth_adjusted': pullback_count > 0},
                            reason='pixel_to_world')
                        target = get_world_xz_from_pixel(
                            pixel_coords=coords,
                            depth_image=depth_image,
                            full_pose=current_pose,
                            camera_intrinsics=self._get_camera_intrinsics(),
                        )
                        new_target_x, new_target_y = self._world_to_map_cell(target)

                        if self.traversable[new_target_y, new_target_x] == 1 or depth_image.max() < 0.1:
                            nav.target_map_x, nav.target_map_y = new_target_x, new_target_y
                            nav.target_set_step = step

                            # Persist the projection: the local target_map_* are cleared
                            # on arrival, which is exactly when the STOP gate needs them.
                            self._record_grounded_target(
                                step=step,
                                target_map_x=nav.target_map_x,
                                target_map_y=nav.target_map_y,
                                target_world_x=target[0],
                                target_world_y=target[1],
                                bbox=bbox,
                                source='grounding',
                                planned_action=nav.planned_action,
                            )

                            nav.waypoint = np.array([nav.target_map_y, nav.target_map_x])

                            use_special_controller, selected_controller = self._select_controller()

                            if use_special_controller:
                                if selected_controller == 'navdp':
                                    tx = nav.target_map_x * self.resolution / 100.0
                                    ty = nav.target_map_y * self.resolution / 100.0
                                    ax = current_pose[0]
                                    ay = current_pose[1]
                                    ah = np.deg2rad(current_pose[2])
                                    
                                    dx = tx - ax
                                    dy = ty - ay
                                    
                                    x_body = dx * np.cos(ah) + dy * np.sin(ah)
                                    y_body = -dx * np.sin(ah) + dy * np.cos(ah)
                                    z_body = 0.0
                                    
                                    goals = np.array([[x_body, y_body, z_body]])
                                    rgb_img = obs[0]['rgb']
                                    min_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                                    max_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                                    depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                                    depth_img = depth_meters
                                    
                                    if depth_img.ndim == 2:
                                        depth_img = depth_img[:, :, np.newaxis]
                                    images = rgb_img[np.newaxis, ...]
                                    depths = depth_img[np.newaxis, ...]
                                    
                                    next_point, traj, val, mask = self.navdp_agent.step_pointgoal(goals, images, depths)
                                    
                                    # Visualization
                                    local_traj = next_point[0]
                                    global_traj = []
                                    for pt in local_traj:
                                        lb_x, lb_y = pt[:2]
                                        wx = ax + lb_x * np.cos(ah) - lb_y * np.sin(ah)
                                        wy = ay + lb_x * np.sin(ah) + lb_y * np.cos(ah)
                                        global_traj.append([wx, wy])
                                    self.navdp_traj_global = np.array(global_traj)

                                    x, y = local_traj[0][:2]
                                    angle = np.arctan2(y, x)
                                    if angle > 0.26:
                                        action_id = HabitatSimActions.TURN_LEFT
                                    elif angle < -0.26:
                                        action_id = HabitatSimActions.TURN_RIGHT
                                    else:
                                        action_id = HabitatSimActions.MOVE_FORWARD
                                    
                                    action_list.append(action_id)
                                    break


                            # Emit the first navigation action immediately so the agent doesn't idle.
                            if not use_special_controller and self.use_fmm:
                                self._save_goal_star_viz(obs[0], step, current_pose, nav.target_map_x, nav.target_map_y)

                                navigation_action = self.policy._get_action(
                                    current_pose, nav.waypoint, full_map[0], self.traversable,
                                    self.collision_map, step, self.current_episode_id,
                                    self.detected_classes, search_destination
                                )
                                action_list.append(navigation_action)
                            elif not use_special_controller and not self.use_fmm:
                                logger.warning("Step 2 - FMM is disabled and no other planner took over! This should not happen due to init validation.")
                            break
                        # Target landed on untraversable space: walk the projection
                        # back along the view ray by shrinking depth. Logged because
                        # a pulled-back target is a different failure from a wrong
                        # bbox, and today they are indistinguishable after the fact.
                        self.total_projection_pullback += 1
                        pullback_count += 1
                        self._emit_event(
                            'PROJECTION_PULLBACK', step,
                            pullback_count=pullback_count,
                            rejected_target_map=[int(new_target_x), int(new_target_y)],
                            depth_max_m=float(depth_image.max()),
                            source='grounding',
                        )
                        depth_image = depth_image - 0.1

                    nav.panorama_got = False  # reset so the next Step 1 re-spins

                # Step 3: have a target and no queued actions -> keep planning toward it.
                elif nav.has_target and not action_list:
                    nav.waypoint = np.array([nav.target_map_y, nav.target_map_x])
                    
                    use_special_controller, selected_controller = self._select_controller()

                    if selected_controller == 'navdp':
                        tx = nav.target_map_x * self.resolution / 100.0
                        ty = nav.target_map_y * self.resolution / 100.0
                        ax = current_pose[0]
                        ay = current_pose[1]
                        ah = np.deg2rad(current_pose[2])
                        
                        dx = tx - ax
                        dy = ty - ay
                        x_body = dx * np.cos(ah) + dy * np.sin(ah)
                        y_body = -dx * np.sin(ah) + dy * np.cos(ah)
                        z_body = 0.0
                        
                        goals = np.array([[x_body, y_body, z_body]])
                        rgb_img = obs[0]['rgb']
                        min_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                        max_d = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                        depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                        depth_img = depth_meters
                        # NavDP Frequency Control
                        # Initialize variables if not present
                        if not hasattr(self, 'last_navdp_step'):
                            self.last_navdp_step = -999
                        if not hasattr(self, 'navdp_local_traj'):
                            self.navdp_local_traj = None

                        # Execute NavDP planning every 5 steps or if no trajectory exists
                        if step - self.last_navdp_step >= 5 or self.navdp_local_traj is None or len(self.navdp_local_traj) == 0:
                            
                            if depth_img.ndim == 2:
                                depth_img = depth_img[:, :, np.newaxis]
                            images = rgb_img[np.newaxis, ...]
                            depths = depth_img[np.newaxis, ...]
                            
                            next_point, traj, val, mask = self.navdp_agent.step_pointgoal(goals, images, depths)
                            
                            # Store the new trajectory and update timestamp
                            self.navdp_local_traj = next_point[0] # (N, 3)
                            self.last_navdp_step = step
                            
                            # Update Global Visualization (Optional, but good for debugging)
                            local_traj = self.navdp_local_traj
                            global_traj = []
                            for pt in local_traj:
                                lb_x, lb_y = pt[:2]
                                wx = ax + lb_x * np.cos(ah) - lb_y * np.sin(ah)
                                wy = ay + lb_x * np.sin(ah) + lb_y * np.cos(ah)
                                global_traj.append([wx, wy])
                            self.navdp_traj_global = np.array(global_traj)
                        
                        # Let's use the GLOBAL trajectory method for stability across steps.
                        if self.navdp_traj_global is not None and len(self.navdp_traj_global) > 0:
                            # Find a lookahead point on the global trajectory relative to CURRENT pose
                            target_pt = None
                            lookahead_dist = 0.5
                            
                            # Convert global trajectory back to CURRENT local frame
                            current_local_traj = []
                            for gpt in self.navdp_traj_global:
                                gx, gy = gpt
                                dx = gx - ax
                                dy = gy - ay
                                # Inverse rotate to get local x, y
                                lx = dx * np.cos(ah) + dy * np.sin(ah)
                                ly = -dx * np.sin(ah) + dy * np.cos(ah)
                                current_local_traj.append([lx, ly])
                            
                            # Now find the target point in current local frame
                            for pt in current_local_traj:
                                dist = np.linalg.norm(pt[:2])
                                if dist > lookahead_dist:
                                    target_pt = pt
                                    break
                            
                            if target_pt is None and len(current_local_traj) > 0:
                                target_pt = current_local_traj[-1]
                        else:
                            # Fallback
                            target_pt = None

                        if target_pt is not None:
                            x, y = target_pt[:2]
                            angle = np.arctan2(y, x)
                            if angle > 0.26:
                                action_id = HabitatSimActions.TURN_LEFT
                            elif angle < -0.26:
                                action_id = HabitatSimActions.TURN_RIGHT
                            else:
                                action_id = HabitatSimActions.MOVE_FORWARD
                        else:
                            action_id = HabitatSimActions.MOVE_FORWARD
                        
                        action_list.append(action_id)
                    elif not use_special_controller and self.use_fmm:
                        logger.info(f"DEBUG: Step 3 - Using FMM. use_navdp={self.use_navdp}, stair={self.agent.stair}")
                        # Clear special controller trajectories to avoid stale visualization
                        self.navdp_traj_global = None
                        self.navdp_local_traj = None
                        
                        self._save_goal_star_viz(obs[0], step, current_pose, nav.target_map_x, nav.target_map_y)

                        navigation_action = self.policy._get_action(
                            current_pose, nav.waypoint, full_map[0], self.traversable,
                            self.collision_map, step, self.current_episode_id,
                            self.detected_classes, search_destination
                        )
                        action_list.append(navigation_action)
                    else:
                        logger.warning("Step 3 - FMM is disabled and no other planner took over!")


            # The frame for this step was written at the top of the loop, before
            # the decide above existed: the grounding's grounding box, the planner's chosen
            # action, the projected target and the STOP gate verdict all land
            # here. Rewrite it so the frame shows the step it is labelled with.
            # No-ops on steps where no decision was made.
            if self.visualizer.visualize:
                self.visualizer.redraw_last_frame(
                    target_coords=(nav.target_map_x, nav.target_map_y),
                    todo_list=self.agent._format_todo_for_prompt() if self.agent.use_todo_list else None,
                    plan_output=self.latest_plan,
                    navdp_traj=([(pt[0], pt[1]) for pt in self.navdp_traj_global]
                                if getattr(self, 'navdp_traj_global', None) is not None else None),
                    telemetry=self._vis_telemetry(
                        (getattr(self, '_last_info', None) or {}).get('distance_to_goal')))

            # Execute only when there is a queued action.
            if action_list:
                # =================================================================
                # 3. ACT for step N: execute the chosen action.
                # =================================================================
                self._action = action_list[0]
                action_list.pop(0)
                actions = [{"action": self._action}]

                outputs = self._harness_step(
                    actions, step, tool_name=self._action_tool_name(self._action),
                    reason='main_action', metadata={'action': int(self._action)})
                step += 1

                # =================================================================
                # 4. UPDATE for step N+1: refresh map and pose from the new obs.
                # =================================================================
                obs, _, dones, infos = [list(x) for x in zip(*outputs)]
                # Freshest measured geometry; the per-step visualization reads
                # distance_to_goal off this rather than relying on `infos` still
                # being in scope from an earlier branch.
                self._last_info = infos[0] if infos else None
                if obs:
                    self._last_obs = obs[0]

                if not dones[0]:
                    batch_obs = self._batch_obs(obs)
                    poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
                    self.mapping_module(batch_obs, poses, self.current_step)
                    full_map, full_pose, one_step_full_map = \
                        self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
                    self.mapping_module.one_step_full_map.fill_(0.)
                    self.mapping_module.one_step_local_map.fill_(0.)

                    self.traversable, self.floor, self.frontiers = self._process_map(step, full_map[0])
                    if self.harness_enabled and self.harness is not None:
                        self.harness.clear_fresh_observation(step, 'post_action_map_update')
                    self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])

                    last_pose = current_pose
                    current_pose = full_pose[0]
                    if last_pose is not None and current_pose is not None:
                        displacement = calculate_displacement(last_pose, current_pose, self.resolution)
                        if displacement < 0.2 * 100 / self.resolution:
                            collided += 1
                        else:
                            collided = 0
                        if collided >= 30:
                            fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                                 f"r{self.local_rank}_w{self.world_size}_collision_stuck.txt")
                            with open(fname, "a") as f:
                                f.writelines(
                                    f"id: {str(self.current_episode_id)}; step: {str(step)}; collided: {str(collided)}\n")

                    current_action = self._action
                    if last_pose is not None and current_action is not None and current_action == 1:
                        collision_map = collision_check_fmm(last_pose, current_pose, self.resolution,
                                                            self.mapping_module.map_shape)
                        self.collision_map = np.logical_or(self.collision_map, collision_map)
                    self.traversable[self.collision_map == 1] = 0
                else:
                    self._finish_episode(step, 'env_done_after_action', infos)
                    return
            else:
                # action_list is empty: skip ACT and let the next iteration plan again.
                pass
        self._finish_episode(step, terminal_reason, infos)

    def _exp_name(self):
        """The round's name. Prefer config.EXP_NAME: under EVAL_DATA_ROOT the
        EVAL_CKPT_PATH_DIR basename is the literal "stats", not the run name.
        """
        name = getattr(self.config, 'EXP_NAME', '')
        if name:
            return name
        return os.path.basename(self.config.EVAL_CKPT_PATH_DIR.rstrip('/'))

    def eval(self):
        # Load detailed assignments if available. File is namespaced by exp_name
        # so parallel runs don't read each other's assignments.
        exp_name = self._exp_name()
        log_dir = getattr(self.config, 'RUNNING_LOG_DIR', 'data/logs/running_log')
        assignment_file = os.path.join(
            log_dir, f"worker_{self.local_rank}_assignments_{exp_name}.json")
        if not os.path.exists(assignment_file):
            # Backward compat: fall back to the un-namespaced filename
            assignment_file = os.path.join(
                log_dir, f"worker_{self.local_rank}_assignments.json")
        if os.path.exists(assignment_file):
            with open(assignment_file, 'r') as f:
                assignments = json.load(f)
                # Set of (id, scene_id) tuples for fast lookup
                self.allowed_episodes = set((str(item['id']), item['scene_id']) for item in assignments)
            logger.info(f"Loaded {len(self.allowed_episodes)} specific episodes to process from {assignment_file}")

        # Sampling logic
        if self.config.EVAL.EPISODE_COUNT > -1:
            pass # Sampling is handled in run_mp.py, here we just use the assigned episodes


        self._init_envs()
        self._collect_val_traj()
        self._initialize_policy()
        self.agent.reset()

        # Reset model usage stats
        self.agent.model.reset_stats()

        eps_to_eval = sum(self.envs.number_of_episodes)

        self.state_eps = {}
        t1 = time.time()
        
        logger.info(f"Worker {self.local_rank}: Starting eval loop with {eps_to_eval} episodes")
        
        for i in tqdm(range(eps_to_eval), desc=f"Worker {self.local_rank}"):
            self.rollout()
            self.reset()

        self.envs.close()

        # Print final statistics
        logger.info("=== FINAL MODEL USAGE STATISTICS ===")
        final_stats = self.agent.model.print_usage_stats()

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)

        # Save model usage stats to separate file
        stats_fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                   f"model_usage_stats_{split}_r{self.local_rank}_w{self.world_size}.json")
        with open(stats_fname, "w") as f:
            json.dump(final_stats, f, indent=2)
        logger.info(f"Model usage statistics saved to: {stats_fname}")

        # Save navigation statistics (backtracks, waypoints, stop-gate counters)
        nav_stats = self._nav_stats_dict()
        nav_stats_fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                       f"nav_stats_{split}_r{self.local_rank}_w{self.world_size}.json")
        with open(nav_stats_fname, "w") as f:
            json.dump(nav_stats, f, indent=2)
        logger.info(f"Navigation statistics saved to: {nav_stats_fname}")

        t2 = time.time()
        logger.info(f"time: {t2 - t1}s")
        logger.info("test time: %d", t2 - t1)

    def eval_dynamic(self, ep_queue):
        """Dynamic work-stealing eval: pop episodes from a shared mp.Queue.

        Each worker reuses its trainer/policy/model across episodes but
        rebuilds the habitat env per episode (so EPISODES_ALLOWED can be
        re-targeted to a single id). The shared queue lets fast workers
        grab the next pending episode instead of being stuck with a fixed
        slice.
        """
        import queue as _queue
        # Heavy initialisation done once per worker
        self._initialize_policy()
        self._collect_val_traj()
        self.agent.model.reset_stats()
        self.state_eps = {}
        t1 = time.time()
        ep_count = 0
        skipped = 0
        logger.info(f"Worker {self.local_rank}: dynamic queue eval starting")

        while True:
            try:
                ep_info = ep_queue.get(timeout=2)
            except _queue.Empty:
                logger.info(f"Worker {self.local_rank}: queue empty, exiting after {ep_count} eps")
                break
            ep_id = str(ep_info['id'])
            scene_id = ep_info['scene_id']

            # Re-target the dataset filter to a single ep id and rebuild env.
            self.config.defrost()
            self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = [ep_id]
            self.config.freeze()
            self.allowed_episodes = {(ep_id, scene_id)}
            if hasattr(self, 'envs') and self.envs is not None:
                try:
                    self.envs.close()
                except Exception as e:
                    logger.warning(f"envs.close() failed: {e}")
            # Skip IDs excluded by the split or language filter and continue draining the queue.
            try:
                self._init_envs()
            except NoMatchingEpisodesError as e:
                logger.error(
                    f"Worker {self.local_rank}: skipping ep {ep_id} -- not in this "
                    f"dataset ({e})"
                )
                skipped += 1
                continue
            self.agent.reset()
            try:
                self.rollout()
            except Exception as e:
                logger.error(f"Worker {self.local_rank} ep {ep_id} rollout failed: {e}")
                if _is_fatal_cuda_error(e):
                    raise RuntimeError(
                        f"Fatal CUDA failure while evaluating episode {ep_id}; "
                        "aborting this worker instead of skipping the remaining queue."
                    ) from e
            try:
                self.reset()
            except Exception as e:
                logger.error(f"Worker {self.local_rank} ep {ep_id} reset failed: {e}")
            ep_count += 1

        # Final cleanup + stats persistence (mirrors eval())
        if hasattr(self, 'envs') and self.envs is not None:
            try:
                self.envs.close()
            except Exception:
                pass

        final_stats = self.agent.model.print_usage_stats()
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        stem = f"{split}_r{self.local_rank}_w{self.world_size}"
        with open(os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"stats_ep_ckpt_{stem}.json"), "w") as f:
            json.dump(self.state_eps, f, indent=2)
        with open(os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"model_usage_stats_{stem}.json"), "w") as f:
            json.dump(final_stats, f, indent=2)
        with open(os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"nav_stats_{stem}.json"), "w") as f:
            json.dump(self._nav_stats_dict(), f, indent=2)
        logger.info(f"Worker {self.local_rank}: ran {ep_count} eps in {time.time() - t1:.0f}s"
                    + (f" ({skipped} skipped as not-in-dataset)" if skipped else ""))
