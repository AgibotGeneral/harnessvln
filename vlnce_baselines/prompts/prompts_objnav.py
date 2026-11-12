
NAVIGATION_WITH_BACKTRACK_PROMPT = """Based on the navigation history, current 4-directional views, and the TODO list,
decide the next action to find the target object. If appropriate, suggest incremental updates to the TODO list.

Current TODO List:
{todo_list}

{memory}

{feedback}

Task:
1. FIRST, write `reasoning_todo`: review ALL TODO items (not just the first one) and
   decide what to update. Address every item by the `todo_id` shown in brackets, e.g.
   `todo-3` — NOT by its position in the list, which shifts as items are added.
   Items have four statuses:
   - "pending": not started.
   - "active": being worked on right now. At most ONE item may be active.
   - "completed": done, and you can point at the observation that confirms it.
   - "blocked": cannot proceed; requires a `reason`.
   Sub-goals may be completed out of order or implicitly as you navigate.
   - If yes: in `todo_updates`, mark it `completed` with a concrete `result` describing
     the observation that confirms completion (e.g., "Saw kitchen through archway at
     Waypoint 3").
   - IMPORTANT: If you mark a LATER item `completed`, re-examine EARLIER items —
     they may have been done implicitly while you traversed.
   - If an item no longer matches what you see, use `op:rewrite` to refine it.
   - If a new sub-goal emerges, `op:add` it; omit `after` to append at the end, or
     set `after` to a `todo_id` to insert directly after that item.
   - If an item turns out to be irrelevant / unreachable, `op:remove` it.
   - `completed` WITHOUT a concrete `result` is REJECTED, and the rejection is
     reported back to you on the next turn rather than silently discarded.

2. THEN, write `reasoning_action` and choose the next action based on the UPDATED TODO
   plus what you observe.
   Choose one of these actions:
{action_list}

Response format (JSON — strictly in this order):
{{
    "progress_analysis": "<brief assessment of what you've observed and done so far>",
    "place_summary": "<one line: the room type you are standing in and its 2-3 defining objects>",
    "reasoning_todo": "<reasoning behind the TODO updates>",
    "todo_updates": [  // include any needed updates; use [] if nothing changes
        {{"todo_id": "<id>", "status": "completed", "result": "<REQUIRED observation>", "evidence_ids": ["<the ob- id shown on the view that proves it>"]}},
        {{"todo_id": "<id>", "status": "active"}},
        {{"todo_id": "<id>", "status": "blocked", "reason": "<REQUIRED why it cannot proceed>"}},
        {{"op": "rewrite", "todo_id": "<id>", "content": "<refined description>"}},
        {{"op": "add", "content": "<new sub-goal>", "status": "pending", "after": "<todo_id (optional; omit to append at end)>"}},
        {{"op": "remove", "todo_id": "<id>"}}
    ],
    "reasoning_action": "<reasoning for the chosen action, given the updated TODO + observations>",
    "action": "{action_desc}",
    "stop": True/False,
    "stair": False|"up"|"down"
}}

Guidelines:
- Do NOT try to open the doors.
- Default to False. Set "up"/"down" ONLY when physically on a staircase crossing between floors. Flat ground, ramps, thresholds, a single step, or carpet edges are all False.
- Set stop=true whenever you have visual confirmation of the target within ~1m. The TODO
  list does NOT gate stopping — you may stop regardless of how many pending items remain.
- Focus on finding the target object specified in the task.
{negative_constraints}"""


REPLAN_PROMPT = """You are a navigation agent. You have backtracked to a previous waypoint to have a second chance to choose action.

Instruction: "{instruction}"

Navigation History (up to this waypoint):
(Images provided above)

Current 4-directional views at this waypoint:
(Images provided above)

Previous Action: **navigate to {previous_action}** from here.

Previous Trajectory (Path taken from here):
(Images provided above)

Task:
- Review the Previous Trajectory to understand the outcome of the previous choice.
- Analyze the Current 4-directional views to give a second-chance choice.
- The previous choice can be reconsidered if necessary.

Available Actions:
{action_list}

Response format (JSON):
{{
    "reasoning": "<analysis of the previous path and why the new direction is chosen>",
    "action": "{action_desc}",
    "stop": True/False,
    "stair": False|"up"|"down"
}}

Guidelines:
- Do NOT try to open the doors.
- If you believe you have reached the target object at this waypoint, set "stop": True.
- Default to False. Set "up"/"down" ONLY when physically on a staircase crossing between floors. Flat ground, ramps, thresholds, a single step, or carpet edges are all False.
- Focus on finding the target object specified in the task. 
{negative_constraints}"""

TODO_PLAN_PROMPT = """You are a navigation agent.
Instruction: "{instruction}"

Create a TODO list to complete this instruction.

Response format (JSON ONLY):
{{
    "todos": [
        {{"content": "<step description>", "status": "pending"}},
        {{"content": "<step description>", "status": "pending"}},
        ...
    ]
}}

Return ONLY the JSON object.
"""

NAVIGATION_PROMPT = """Based on the navigation history, current 4-directional views, and the TODO list,
decide the next navigation direction. If appropriate, suggest incremental updates to the TODO list.

Current TODO List:
{todo_list}

{memory}

{feedback}

Task:
1. FIRST, write `reasoning_todo` and `todo_updates`. Address every item by the
   `todo_id` shown in brackets (e.g. `todo-3`), NOT by position. Items have four
   statuses: "pending", "active" (at most one), "completed", "blocked" (needs a
   `reason`). Review ALL items (not just the first one); a sub-goal may be
   completed out of order.
   - Mark an item `completed` if you judge it done (with a concrete `result`).
   - If marking a LATER item `completed`, re-check EARLIER items.
   - Use `op:rewrite` to refine; `op:add` to introduce a new sub-goal (optional
     `after: <todo_id>` to insert after that item; omit to append); `op:remove` to drop.
   - `completed` without a concrete `result` is rejected, and you are told why.

2. THEN, write `reasoning_action` and choose the next action based on the UPDATED TODO
   plus what you observe.
   Choose one of these actions:
{action_list}

Response format (JSON — strictly in this order):
{{
    "progress_analysis": "<brief assessment of search progress>",
    "place_summary": "<one line: the room type you are standing in and its 2-3 defining objects>",
    "reasoning_todo": "<reasoning behind the TODO updates>",
    "todo_updates": [
        {{"todo_id": "<id>", "status": "completed", "result": "<REQUIRED observation>", "evidence_ids": ["<the ob- id shown on the view that proves it>"]}},
        {{"todo_id": "<id>", "status": "active"}},
        {{"todo_id": "<id>", "status": "blocked", "reason": "<REQUIRED why it cannot proceed>"}},
        {{"op": "rewrite", "todo_id": "<id>", "content": "<refined description>"}},
        {{"op": "add", "content": "<new task>", "status": "pending", "after": "<todo_id (optional; omit to append)>"}},
        {{"op": "remove", "todo_id": "<id>"}}
    ],
    "reasoning_action": "<reasoning for the chosen action, given the updated TODO + observations>",
    "action": "{action_desc}",
    "stop": False,
    "stair": False|"up"|"down"
}}

Guidelines:
- Focus on finding the target object specified in the task.
- Default to False. Set "up"/"down" ONLY when physically on a staircase crossing between floors. Flat ground, ramps, thresholds, a single step, or carpet edges are all False.
{negative_constraints}"""


GROUNDING_PROMPT = """Navigation Task: "{instruction}"

Current situation:
- Step: {current_step}
- Image size: {width}x{height} pixels
- Current TODO List:
{todo_list}
- Progress Info: {progress_info}

Your task:
1. Identify the most relevant target object/area for what you should do next. Specify ONLY ONE. And it should not be too close to you.
2. Provide the bounding box of the target.

Response format (JSON):
{{
    "reasoning": "<brief explanation of decision>",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<description of target object>"
}}

Guidelines:
- Target description should be specific and clear.
- If deciding to go up or down stairs, the bbox MUST select the ENTRY of the stairs.
- Provide bounding box of next target."""

# ==== Additional ObjectNav-specific prompts ====


OBJECTNAV_PROMPT = """Task: Find the "{target_object}".

Based on the navigation history and current 4-directional views, decide the next action to find the target.

Review the "Navigation History" provided above to identify available waypoints for backtracking.
Each waypoint in the history (e.g., "Waypoint 0", "Waypoint 1") is a potential candidate.

Analysis:
1. **Target Identification**: Is the "{target_object}" visible in any current view?
2. **Room Type Inference**: Based on objects seen, where are you? Is the target likely here?
3. **Exploration**: If target is not seen, choose a direction that leads to a *new* area or a room likely to contain the target.

Choose one of these actions:
1. navigate to forward - continue straight ahead
2. navigate to left - turn left and go forward
3. navigate to right - turn right and go forward
4. navigate to behind - turn around and go forward
5. backtrack to <waypoint_id> - return to a previous waypoint

Response format (JSON):
{{
    "progress_analysis": "<searching/spotted/exploring>",
    "reasoning": "<explanation of chosen action>",
    "action": "navigate to forward|left|right|behind" or "backtrack to <waypoint_id>",
    "stair": True/False
}}

Guidelines:
- If "{target_object}" is visible, go towards it immediately.
- If not, go towards the most open path or a door leading to a new room.
- Consider backtracking if you have explored the current area sufficiently.
- Use "stair": True ONLY when physically on a staircase crossing between floors; default False.
"""


# NOTE: The first GROUNDING_PROMPT (line 197) uses {todo_list} and matches query_llm's
# call signature. The variant below (with {visited_targets_str}) is kept only as
# GROUNDING_PROMPT_VISITED_TARGETS for reference — we do NOT override GROUNDING_PROMPT here.
GROUNDING_PROMPT_VISITED_TARGETS = """Navigation Task: "{instruction}"

Current situation:
- Step: {current_step}
- Image size: {width}x{height} pixels{visited_targets_str}{progress_info}

Your task:
1. Analyze at what stage the current instruction has been completed and what should be done next
2. Identify the most relevant target object/area for what you should do next. Specify ONLY ONE. And it should not be too close to you.
3. Decide if the robot should STOP (if task is completed or very close to final goal)

Response format (JSON):
{{
    "progress": "<assessment of how close to completing the instruction>",
    "reasoning": "<brief explanation of decision>",
    "action": "NAVIGATE" or "STOP",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<description of target object>"
}}

Guidelines:
- If you see the final destination mentioned in instruction, consider STOP action
- If already very close to the goal object, choose STOP
- If still need to navigate, choose NAVIGATE and provide bounding box of next target
- Target description should be specific and clear
- Consider the instruction completion progress based on visited targets
- Use the progress analysis to inform your decision"""

GROUNDING_PROMPT_OBJECTNAV = """Object Navigation Task: "{instruction}"

Current View Status:
- Step: {current_step}
- Image size: {width}x{height} pixels{visited_targets_str}{progress_info}

Goal: Find and approach the "{target_object}".

Your Task:
1. **Detect**: Check if "{target_object}" is present in the image.
2. **Contextualize**: If not found, identify the current room type and visible exits (doors, hallways).
3. **Act**:
   - If **Found**: Set bbox around the "{target_object}" and NAVIGATE.
   - If **Searching**: Set bbox around the most promising area to explore (e.g., a door, a hallway, or a region likely to contain the target).
   - **STOP**: Only if you are within 1.0 meters of the "{target_object}".

Response format (JSON):
{{
    "progress": "<searching status>",
    "reasoning": "<logic for choice>",
    "action": "NAVIGATE" or "STOP",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<description of the selected target object or area>"
}}
"""

STOP_CHECK_PROMPT = """You are an intelligent navigation agent.
Task: "{instruction}"

You have decided to STOP, believing you have reached the target.
Now, please double-check your decision based on the current 4-directional views.

Current Views:
{current_views}

Requirements:
1. Check if the target object is clearly visible in any of the views.
2. Check that it is the correct object category for the task, and not a
   look-alike (for example a framed picture is not a mirror, a dryer is not a
   refrigerator, a rug is not a carpeted floor).
3. Do NOT judge distance. Distance is measured from the depth sensor, not
   estimated from the image. Your job is identification only.
4. If the target is visible, report which view shows it most clearly and give a
   tight bounding box around the target object in that view.

Decision Rules:
- If the target is clearly visible AND is the correct category: CONFIRM STOP.
- If the target is NOT visible, OR you are unsure it is the right category:
  CONTINUE NAVIGATION.

Bounding box format: [x1, y1, x2, y2] on a normalized 0-1000 scale for both
axes, with (0, 0) at the top-left of the chosen view -- the same convention the
grounding step uses. Box the object itself, tightly: the box anchors the depth
measurement, so a box that spills onto the surrounding wall or floor will
measure the distance to that surface instead of to the target.

Response format (JSON):
{{
    "analysis": "<what is visible, and why it is or is not the target category>",
    "decision": "STOP" or "CONTINUE",
    "target_view": "forward" | "left" | "behind" | "right" | null,
    "bbox_2d": [x1, y1, x2, y2]
}}

Set "target_view" and "bbox_2d" whenever "decision" is "STOP". Use null for
"target_view" and an empty list for "bbox_2d" when the decision is "CONTINUE".
"""

GUIDELINES_GENERATOR = """You are given a natural language route instruction for an indoor navigation task.

Instruction:
"{instruction}"

Your task:
1. Read and fully understand the instruction.
2. Summarize it into stable, high-level navigation guidelines that should be remembered at EVERY step of the episode.

Requirements for the guidelines:
- Be concise (3–7 bullet points).
- Cover:
  - The global navigation goal (final destination / outcome).
  - Key intermediate landmarks or sub-goals to pass by.
  - Important turning patterns or overall direction trends.
  - Safety or motion constraints (e.g., stay close to wall, avoid stairs if not mentioned, etc.).
  - When to STOP (stopping condition) according to the instruction.
- Do NOT refer to step indices, images, or past actions.
- Use generic wording that remains valid throughout the whole episode.

Response format:
Return ONLY the guidelines as a numbered list, one guideline per line, for example: 
1. Exit the current room by turning around and passing through the doorway.  
2. Turn left and exit the next door encountered.  
3. Proceed towards the pool as your destination.  
4. Stay alert for obstacles and maintain a clear path.  
5. Stop when you reach the area next to the pool.  
6. Avoid entering any other doors or rooms unless instructed.  
7. Ensure you are positioned safely next to the pool before stopping. 
... (The guidelines will vary based on the instruction provided, this example is just for illustration of the format.)
"""

GUIDELINES_GENERATOR_OBJECTNAV = """You are an expert indoor navigation agent. Your task is to find a specific object in an unknown environment.

Target Object: "{target_object}"

Your task:
Generate 3-5 strategic guidelines for finding this object efficiently.

Requirements:
1. **Probable Locations**: Identify the room types where "{target_object}" is typically found (e.g., Bed -> Bedroom, Stove -> Kitchen, Sofa -> Living Room).
2. **Visual Cues**: List associated objects that might indicate proximity to the target (e.g., seeing a mirror might mean a sink/bathroom is near).
3. **Exploration Strategy**: Advise on how to explore (e.g., checking multiple rooms quickly vs. searching inside cabinets/corners).
4. **Stopping Condition**: Explicitly state to STOP when the object is clearly visible and within 1.0 meters.

Response format:
Return ONLY the numbered list of guidelines.
"""

# Leading text block passed to the grounding model alongside the current RGB and depth
# images. Kept as a constant so the system prompt lives next to the rest of the
# task prompts. No placeholders.
ROBOT_NAVIGATION_SYSTEM_PROMPT = (
    "You are a robot performing navigation task. "
    "Look at this image and the corresponding depth gray image and help the robot navigate."
)
