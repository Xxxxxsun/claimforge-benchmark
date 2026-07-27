"""Frozen protocol prompts. Bump PROTOCOL_VERSION when changing any text."""

DETECTION_PROTOCOL_VERSION = "mllm_protocol_v3_reasoning_image_coordinates"
LOCALIZATION_PROTOCOL_VERSION = "mllm_protocol_v4_reasoning_pixel_coordinates"
LOCALIZATION_BBOX1000_PROTOCOL_VERSION = "mllm_protocol_v5_reasoning_bbox1000_coordinates"
PROTOCOL_SUITE_VERSION = "mllm_protocol_suite_20260724"
BBOX1000_PROTOCOL_SUITE_VERSION = "mllm_protocol_suite_20260727_bbox1000"
PROTOCOL_VERSIONS = {
    "detection": DETECTION_PROTOCOL_VERSION,
    "localization": LOCALIZATION_PROTOCOL_VERSION,
}
# Backward-compatible alias for code that imports the latest protocol version.
PROTOCOL_VERSION = LOCALIZATION_PROTOCOL_VERSION

SYSTEM_PROMPT = """You are an image-forensics assistant. Your task is to examine a given image and determine whether any object in the image has been digitally modified or manipulated. Pay close attention to subtle inconsistencies in lighting, shadows, textures, edges, perspective, or logical composition. Carefully analyze these visual cues before making a judgment."""

DETECTION_PROMPT = """
Instructions:
1. Provide a **detailed explanation** of your reasoning.
2. Then, based on your analysis, provide a final decision on "edited" or "not_edited".
3. Provide a probability estimate and up to 3 visible-evidence statements.

Important Constraints:
- Your reasoning must come before the result statement.
- Be cautious: minor edits may be hard to detect but should still be flagged if visible.

Return exactly one JSON object and no Markdown:
{
  \"reasoning\": \"<detailed explanation>\",
  \"decision\": \"edited\" | \"not_edited\",
  \"p_ai_edited\": <integer 0-100>,
  \"evidence\": [<at most 3 short visible-evidence statements>]
}
"""

LOCALIZATION_PROMPT = """
Instructions:
1. Provide a **detailed explanation** of your reasoning.
2. Then, based on your analysis, provide a final decision on "localized_edit", "no_localized_edit".
3. If your decision is "localized_edit", provide a list of regions in the format of { "bbox_px": [<x1>, <y1>, <x2>, <y2>], "confidence": <integer 0-100>, "evidence": "<short visible-evidence statement>" }.

Important Constraints:
- Your reasoning must come before the result statement.
- Coordinates are original full-image pixel coordinates, not normalized coordinates and not coordinates from a crop.
- Do not invent a region when there is no visual evidence. Use an empty regions array when no specific region can be supported by visible evidence.
- Return no more than 3 regions, ordered by confidence.

Return exactly one JSON object and no Markdown:
{
  \"reasoning\": \"<detailed explanation>\",
  \"decision\": \"localized_edit\" | \"no_localized_edit\",
  \"p_ai_edited\": <integer 0-100>,
  \"regions\": [{
    \"bbox_px\": [<x1>, <y1>, <x2>, <y2>],
    \"confidence\": <integer 0-100>,
    \"evidence\": \"<short visible-evidence statement>\"
  }]
}

"""

LOCALIZATION_BBOX1000_PROMPT = """
Instructions:
1. Provide a **detailed explanation** of your reasoning.
2. Then, based on your analysis, provide a final decision on "localized_edit", "no_localized_edit".
3. If your decision is "localized_edit", provide a list of regions in the format of { "bbox_1000": [<x1>, <y1>, <x2>, <y2>], "confidence": <integer 0-100>, "evidence": "<short visible-evidence statement>" }.

Important Constraints:
- Your reasoning must come before the result statement.
- Coordinates use a normalized 0-1000 full-image coordinate system, not pixel coordinates and not coordinates from a crop.
- The top-left corner is (0, 0) and the bottom-right corner is (1000, 1000).
- Do not invent a region when there is no visual evidence. Use an empty regions array when no specific region can be supported by visible evidence.
- Return no more than 3 regions, ordered by confidence.

Return exactly one JSON object and no Markdown:
{
  "reasoning": "<detailed explanation>",
  "decision": "localized_edit" | "no_localized_edit",
  "p_ai_edited": <integer 0-100>,
  "regions": [{
    "bbox_1000": [<x1>, <y1>, <x2>, <y2>],
    "confidence": <integer 0-100>,
    "evidence": "<short visible-evidence statement>"
  }]
}

"""

REPAIR_SUFFIX = "\n\nYour previous response was not valid for the required JSON schema. Output only one valid JSON object."
PROMPTS = {"detection": DETECTION_PROMPT, "localization": LOCALIZATION_PROMPT}
