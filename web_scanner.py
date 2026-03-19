#!/usr/bin/env python3
"""
Product SKU Scanner — Web App
Flask server that accepts an image upload, sends it to Claude Vision
for product/SKU identification, and displays results in the browser.
"""

import base64
import json
import os
import uuid
from io import BytesIO

from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import anthropic

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB

ALLOWED_EXT = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff", "tif"}

VISION_PROMPT = """\
You are analyzing a product shelf or display photo. Identify every product position \
and estimate the count of sellable items at each position.

Scan from TOP-LEFT to BOTTOM-RIGHT, row by row. A "row" is a horizontal shelf level.

=== DECISION TREE — Follow these steps IN ORDER for each position ===

STEP 1: IS THERE AN ENCLOSING DISPLAY BOX?
Look at the position. Is the product contained in a cardboard display box, shipper tray, \
or open-top case? These are branded cardboard containers that hold individual items inside.
  → If YES: go to STEP 2.
  → If NO (items on a peg, hook, bare shelf, etc.): go to STEP 3. Set has_display_box = false.

STEP 2: IS THE DISPLAY BOX EMPTY?
Look INSIDE the box opening. Ignore the box exterior (the printed brand graphics on the \
cardboard walls — that is advertising, NOT inventory).
Focus ONLY on what is INSIDE the box opening:
  - Can you see the flat bottom of the box (plain cardboard)?
  - Is the interior mostly hollow/empty space?
  - Are there fewer than 3 distinct individual wrapped packages visible inside?
  - Does the box interior look flat and uniform rather than bumpy with stacked items?

If the box interior appears empty or nearly empty:
  → Set has_display_box = true, box_appears_empty = true
  → Set estimated_count = 0 (or 1-2 if you can see that many individual items inside)
  → Set box_fill_assessment = "empty_box" or "nearly_empty"
  → STOP counting for this position. Do not proceed to Step 3.

If the box clearly contains multiple individual items:
  → Set has_display_box = true, box_appears_empty = false
  → Set box_fill_assessment = "partially_filled" or "well_stocked"
  → Go to STEP 3 to count ONLY the individual items INSIDE the box.

STEP 3: COUNT INDIVIDUAL ITEMS
Now count the sellable items. If there is a display box, count only what is INSIDE it.

Method A — Direct visual count:
  - Count each individual wrapped/packaged item you can see as a distinct unit.
  - For items behind the front row, estimate based on visible depth and shadows.
  - Record as method_a_count.

Method B — Capacity-based estimate:
  - Estimate the space depth (rod length for pegs, shelf depth, or box interior depth) \
    in inches. Record as space_depth_inches.
  - Estimate ONE package's depth in inches. Record as package_depth_inches.
  - Compute capacity = floor(space_depth_inches / package_depth_inches).
  - Estimate fullness (0-100%). If has_display_box and you can see empty space inside, \
    factor that in. Record as method_b_fullness_pct.
  - Compute method_b_count = round(capacity * method_b_fullness_pct / 100).

Choose the final estimated_count:
  - Use the method you have higher confidence in.
  - If similar confidence, choose the LOWER count.
  - When in doubt, round DOWN.

=== END DECISION TREE ===

CRITICAL REMINDERS:
  - A display box's printed exterior graphics are NEVER items. Do not count them.
  - If box_appears_empty is true, estimated_count MUST be 0 (or 1-2 at most).
  - Every numeric field must have a value, never null.

Return ONLY valid JSON (no markdown, no commentary):
{
  "rows": [
    {
      "row_number": 1,
      "positions": [
        {
          "position": 1,
          "product_name": "Brand Name Product Description Size",
          "has_display_box": true,
          "box_appears_empty": false,
          "box_fill_assessment": "well_stocked",
          "display_type": "box",
          "method_a_count": 4,
          "space_depth_inches": 10,
          "package_depth_inches": 1.5,
          "method_b_capacity": 6,
          "method_b_fullness_pct": 50,
          "method_b_count": 3,
          "chosen_method": "B",
          "estimated_count": 3,
          "counting_notes": "Display box present and stocked. 3 items visible inside, ~1.5in each."
        }
      ]
    }
  ],
  "summary": {
    "total_rows": 3,
    "total_distinct_skus": 10,
    "total_items": 45
  }
}

Rules:
- Be specific with product names (brand, variant, size if readable).
- If you cannot read the label, describe visually (e.g., "Red can, unknown brand").
- has_display_box: true/false — is there an enclosing cardboard display container?
- box_appears_empty: true/false — does the box interior look empty? (only when has_display_box=true)
- box_fill_assessment: "empty_box", "nearly_empty", "partially_filled", "well_stocked", or "not_a_box".
- If box_appears_empty is true: estimated_count MUST be 0 (or 1-2 if you literally see items).
- display_type: "peg", "shelf", "bin", "box", "hook", "slot".
- ALL numeric fields must be numbers, never null.
- chosen_method: "A", "B", or "both".
- When in doubt, round DOWN.
"""


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


MAX_IMAGE_BYTES = 4_800_000  # stay under Claude's 5 MB base64 limit


def encode_image(path: str) -> tuple:
    """Read, resize if needed, and base64-encode an image."""
    ext = os.path.splitext(path)[1].lower()
    img = Image.open(path)

    # Determine output format — use JPEG for photos (smaller), PNG for others
    if ext in (".jpg", ".jpeg"):
        out_fmt, media_type = "JPEG", "image/jpeg"
    else:
        out_fmt, media_type = "PNG", "image/png"

    def _encode(image, fmt, quality=85):
        buf = BytesIO()
        if fmt == "JPEG":
            image = image.convert("RGB")  # JPEG can't handle alpha
            image.save(buf, format=fmt, quality=quality, optimize=True)
        else:
            image.save(buf, format=fmt, optimize=True)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    # Try encoding at original size first
    data = _encode(img, out_fmt)
    raw_size = len(base64.b64decode(data))

    # If too large, progressively downscale
    scale = 1.0
    while raw_size > MAX_IMAGE_BYTES and scale > 0.15:
        scale *= 0.75
        new_size = (int(img.width * scale), int(img.height * scale))
        resized = img.resize(new_size, Image.LANCZOS)
        data = _encode(resized, out_fmt, quality=80)
        raw_size = len(base64.b64decode(data))

    return data, media_type


def analyze_image(path: str) -> dict:
    """Send the image to Claude Vision and return parsed JSON results."""
    client = anthropic.Anthropic()
    img_data, media_type = encode_image(path)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16384,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_data,
                        },
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    parsed = json.loads(raw)

    # Log raw response for debugging
    print("=== Claude Response ===")
    print(json.dumps(parsed, indent=2)[:3000])
    print("=== End Response ===")

    # Normalize: ensure all fields exist with defaults
    for row in parsed.get("rows", []):
        for pos in row.get("positions", []):
            pos.setdefault("method_a_count", pos.get("estimated_count"))
            pos.setdefault("display_type", "shelf")
            pos.setdefault("space_depth_inches", pos.get("rod_length_inches"))
            pos.setdefault("package_depth_inches", None)
            pos.setdefault("method_b_capacity", None)
            pos.setdefault("method_b_fullness_pct", None)
            pos.setdefault("method_b_count", None)
            pos.setdefault("chosen_method", "A")
            pos.setdefault("counting_notes", "")
            pos.setdefault("has_display_box", False)
            pos.setdefault("box_appears_empty", False)
            pos.setdefault("box_fill_assessment", "not_a_box")

            # Server-side enforcement: if box is flagged empty, force counts down
            if pos.get("box_appears_empty") is True:
                pos["box_fill_assessment"] = "empty_box"
                pos["method_a_count"] = 0
                pos["method_b_count"] = 0
                pos["estimated_count"] = 0
                pos["method_b_fullness_pct"] = 0

            bfa = pos.get("box_fill_assessment", "")
            if bfa == "empty_box":
                pos["method_a_count"] = 0
                pos["method_b_count"] = 0
                pos["estimated_count"] = 0
                pos["method_b_fullness_pct"] = 0
            elif bfa == "nearly_empty":
                pos["method_a_count"] = min(pos.get("method_a_count", 2), 2)
                pos["method_b_count"] = min(pos.get("method_b_count", 2), 2)
                pos["estimated_count"] = min(pos.get("estimated_count", 2), 2)
                pos["method_b_fullness_pct"] = min(pos.get("method_b_fullness_pct", 15), 15)

    return parsed


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Use: {', '.join(sorted(ALLOWED_EXT))}"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        data = analyze_image(filepath)
        return jsonify({"image_url": f"/uploads/{filename}", "results": data})
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to parse Claude response: {e}"}), 500
    except anthropic.APIError as e:
        return jsonify({"error": f"Claude API error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Product SKU Scanner at http://localhost:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
