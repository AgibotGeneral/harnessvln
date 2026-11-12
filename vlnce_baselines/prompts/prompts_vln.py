
NAVIGATION_WITH_BACKTRACK_PROMPT = """Based on the text instruction, navigation history, current 4-directional views, and the TODO list,
decide the next action. If appropriate, suggest incremental updates to the TODO list.

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
   For each item, ask: "Based on what I've observed so far, is this item done?"
   - If yes: in `todo_updates`, mark it `completed` with a concrete `result` describing
     the observation that confirms completion (e.g., "Exited via left door at Waypoint 1,
     now in hallway").
   - IMPORTANT: If you mark a LATER item `completed`, re-examine EARLIER items —
     they may have been done implicitly while you traversed.
   - Use `op:rewrite` to refine an item based on new observations.
   - Use `op:add` to introduce a new sub-goal: omit `after` to append at the end, or
     set `after` to a `todo_id` to insert directly after that item.
   - Use `op:remove` to drop irrelevant ones.
   - `completed` WITHOUT a concrete `result` is REJECTED, and the rejection is
     reported back to you on the next turn rather than silently discarded.

2. THEN, write `reasoning_action` and choose `action`: based on the UPDATED TODO
   (after applying your `todo_updates`), the instruction, and observations,
   reason about what to do next, then pick one action.
   Choose one of these actions:
{action_list}

Response format (JSON — strictly in this order):
{{
    "progress_analysis": "<brief assessment of what you've observed and done so far>",
    "place_summary": "<one line: the room type you are standing in and its 2-3 defining objects>",
    "reasoning_todo": "<reasoning behind the TODO updates: which items now satisfied, which need rewrite, etc.>",
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
- Set stop=true whenever you have reached the target described in the instruction. The
  TODO list does NOT gate stopping — you may stop regardless of how many pending items remain.
- Focus on following the text instruction.
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
- Focus on following the text instruction. 
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

DESTINATION_PROMPT = """You are a navigation agent.
Instruction: "{instruction}"

The images provided above are the 4-directional views from the starting position.
Your task is to identify the final object/area the robot needs to reach or get close to. Once the robot is close to the object/area, the whole navigation process stops.

Response format:
Return ONLY the object/area.
"""


INITIAL_PLAN_PROMPT = """You are a navigation agent.
Instruction: "{instruction}"

The images provided above are the 4-directional views from the starting position.

Do TWO things in a single response:
1. Create a TODO list to complete this instruction (sub-goals, in order).
2. Identify the FINAL target object/area. The whole navigation process stops once
   the robot is close to it.

Response format (JSON ONLY):
{{
    "todos": [
        {{"content": "<step description>", "status": "pending"}},
        {{"content": "<step description>", "status": "pending"}}
    ],
    "last_object": "<final object/area, short phrase>"
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
   plus the instruction and observations.
   Choose one of these actions:
{action_list}

Response format (JSON — strictly in this order):
{{
    "progress_analysis": "<brief assessment of progress>",
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
- Focus on following the text instruction.
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


STOP_CHECK_PROMPT = """You are an intelligent navigation agent.
Task: "{instruction}"

You have decided to STOP, believing you have reached the target.
Now, please double-check your decision based on the current 4-directional views and the final target.

Final Target:
{target}

Current Views:
{current_views}

Requirements:
1. Check if the target object is clearly visible in any of the views.
2. Estimate if the distance to the target is less than 1 meter.
3. Compare previous views with current views to confirm you have approached the target.

Decision Rules:
- If target is visible AND distance < 1m: CONFIRM STOP.
- If target is NOT visible OR distance >= 1m: CONTINUE NAVIGATION.

Response format (JSON):
{{
    "analysis": "<analysis of visibility and distance, and comparison with previous views>",
    "decision": "STOP" or "CONTINUE"
}}
"""


# Leading text block passed to the grounding model alongside the current RGB and depth
# images. No placeholders — kept as a constant so the system prompt lives next to
# the rest of the navigation prompts instead of being inlined into query_llm().
ROBOT_NAVIGATION_SYSTEM_PROMPT = (
    "You are a robot performing navigation task. "
    "Look at this image and the corresponding depth gray image and help the robot navigate."
)
