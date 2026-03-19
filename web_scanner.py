#!/usr/bin/env python3
"""
Product SKU Scanner — Web App
Flask server that accepts an image upload, sends it to Claude Vision
for product/SKU identification, and displays results in the browser.
"""

import base64
import concurrent.futures
import json
import os
import re
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

MANDATORY: Before deciding, write an interior_description field describing ONLY what you \
physically see through the box opening. Describe the actual physical contents you observe: \
"flat cardboard bottom visible, no items inside", "3 wrapped candy bars lying flat", \
"box interior is hollow with visible cardboard floor", etc. \
Do NOT describe what is printed/graphic on the cardboard walls — those are ads, not items. \
If you catch yourself describing a logo, product image, or brand graphic, STOP — that is \
the box EXTERIOR, not the interior.

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
  - If your interior_description mentions "logo", "graphic", "printed", "picture", or \
    "branding", you are likely describing the box exterior — re-evaluate and look deeper \
    into the box opening for actual physical items.
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
          "interior_description": "3 individually wrapped candy bars visible lying flat inside the box",
          "box_fill_assessment": "well_stocked",
          "bbox_pct": [10, 25, 30, 55],
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
- interior_description: REQUIRED when has_display_box=true. Describe what you physically see \
  inside the box opening. NOT what is printed on the box exterior.
- box_fill_assessment: "empty_box", "nearly_empty", "partially_filled", "well_stocked", or "not_a_box".
- bbox_pct: [left%, top%, right%, bottom%] — approximate bounding box of this position as \
  percentage of full image dimensions (0-100). Does not need to be exact.
- If box_appears_empty is true: estimated_count MUST be 0 (or 1-2 if you literally see items).
- display_type: "peg", "shelf", "bin", "box", "hook", "slot".
- ALL numeric fields must be numbers, never null.
- chosen_method: "A", "B", or "both".
- When in doubt, round DOWN.
"""


VERIFY_BOX_PROMPT = """\
This is a close-up crop of a single display box on a store shelf. \
Your ONLY job is to determine if this box contains actual product items or is empty.

CRITICAL: IGNORE all printed graphics, logos, and product images on the cardboard exterior. \
Those are advertisements printed on the box walls, NOT real inventory items.

Look at the OPENING of the box and describe what you see through the opening:
- Can you see a flat cardboard bottom with no items on it? → EMPTY
- Is the interior hollow with visible cardboard floor? → EMPTY
- Can you see individual wrapped/packaged items stacked or lying inside? → HAS ITEMS
- Can you see depth/shadows suggesting items behind the front? → HAS ITEMS
- Is there bumpy/uneven texture from multiple packages? → HAS ITEMS

Return ONLY valid JSON (no markdown, no commentary):
{
  "interior_description": "Describe what you physically see inside the box opening",
  "is_empty": true,
  "verified_count": 0,
  "confidence": "high"
}

Rules:
- interior_description: Describe ONLY what is physically inside the box, NOT exterior graphics.
- is_empty: true if box is empty or nearly empty (0-2 items), false if it contains items.
- verified_count: Number of individual items you can see inside (0 if empty).
- confidence: "high", "medium", or "low".
"""

MAX_VERIFICATION_CALLS = 6


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


def crop_region(path: str, bbox_pct: list) -> tuple:
    """Crop a region from the image using percentage-based coordinates.

    bbox_pct: [left%, top%, right%, bottom%] — each value 0-100.
    Returns (base64_data, media_type) of the cropped region.
    """
    img = Image.open(path)
    w, h = img.size

    # Convert percentages to pixels with 5% padding for imprecision
    pad = 5
    left = max(0, int(w * (bbox_pct[0] - pad) / 100))
    top = max(0, int(h * (bbox_pct[1] - pad) / 100))
    right = min(w, int(w * (bbox_pct[2] + pad) / 100))
    bottom = min(h, int(h * (bbox_pct[3] + pad) / 100))

    # Ensure minimum crop size
    if right - left < 50 or bottom - top < 50:
        return None, None

    cropped = img.crop((left, top, right, bottom))

    # Encode cropped region
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        out_fmt, media_type = "JPEG", "image/jpeg"
    else:
        out_fmt, media_type = "PNG", "image/png"

    buf = BytesIO()
    if out_fmt == "JPEG":
        cropped = cropped.convert("RGB")
        cropped.save(buf, format=out_fmt, quality=85, optimize=True)
    else:
        cropped.save(buf, format=out_fmt, optimize=True)

    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return data, media_type


def verify_display_box(client: anthropic.Anthropic, path: str, bbox_pct: list, product_name: str) -> dict:
    """Send a cropped display box image to Claude for focused empty/stocked verification."""
    try:
        img_data, media_type = crop_region(path, bbox_pct)
        if img_data is None:
            return {"error": "crop_too_small", "is_empty": False, "confidence": "low"}

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
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
                        {"type": "text", "text": f"Product: {product_name}\n\n{VERIFY_BOX_PROMPT}"},
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")]
        return json.loads(raw)

    except Exception as e:
        print(f"Verification failed for {product_name}: {e}")
        return {"error": str(e), "is_empty": False, "confidence": "low"}


# Keywords that suggest the interior_description is about exterior graphics, not actual contents
EXTERIOR_KEYWORDS = re.compile(
    r"\b(logo|graphic|printed|picture|branding|brand image|product image|advertisement)\b",
    re.IGNORECASE,
)


def needs_verification(pos: dict) -> bool:
    """Check if a display box position needs a second-pass verification."""
    if not pos.get("has_display_box", False):
        return False
    if pos.get("box_appears_empty", False):
        return False  # Already flagged empty, no need to verify
    if pos.get("estimated_count", 0) <= 2:
        return False  # Low count already, not suspicious

    # Check if bbox_pct is available and valid
    bbox = pos.get("bbox_pct")
    if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
        return False

    # Check if interior_description is suspicious
    desc = pos.get("interior_description", "")
    if not desc or len(desc) < 15:
        return True  # Missing or too vague — suspicious
    if EXTERIOR_KEYWORDS.search(desc):
        return True  # Describes exterior graphics — suspicious

    return False


def enforce_box_rules(pos: dict) -> None:
    """Server-side enforcement of box emptiness rules on a position."""
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

    # ── Pass 1 normalization: ensure all fields exist with defaults ──
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
            pos.setdefault("interior_description", "")
            pos.setdefault("bbox_pct", None)
            pos.setdefault("verification_status", None)

            # First pass enforcement
            enforce_box_rules(pos)

    # ── Pass 2: Verify suspicious display boxes with cropped images ──
    suspicious = []
    for row in parsed.get("rows", []):
        for pos in row.get("positions", []):
            if needs_verification(pos):
                suspicious.append(pos)

    # Cap verification calls
    suspicious = sorted(suspicious, key=lambda p: p.get("estimated_count", 0), reverse=True)
    suspicious = suspicious[:MAX_VERIFICATION_CALLS]

    if suspicious:
        print(f"=== Verification Pass: {len(suspicious)} display box(es) to verify ===")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_pos = {}
                for pos in suspicious:
                    future = executor.submit(
                        verify_display_box,
                        client,
                        path,
                        pos["bbox_pct"],
                        pos.get("product_name", "Unknown"),
                    )
                    future_to_pos[future] = pos

                for future in concurrent.futures.as_completed(future_to_pos, timeout=60):
                    pos = future_to_pos[future]
                    result = future.result()
                    print(f"  Verified '{pos.get('product_name')}': {json.dumps(result)}")

                    if result.get("error"):
                        pos["verification_status"] = "error"
                        continue

                    confidence = result.get("confidence", "low")
                    if result.get("is_empty") and confidence in ("high", "medium"):
                        # Override: box is actually empty
                        pos["original_estimated_count"] = pos.get("estimated_count", 0)
                        pos["box_appears_empty"] = True
                        pos["box_fill_assessment"] = "empty_box"
                        verified_count = result.get("verified_count", 0)
                        pos["estimated_count"] = min(verified_count, 2)
                        pos["verification_status"] = "overridden_by_crop_check"
                        pos["interior_description"] = result.get(
                            "interior_description", pos.get("interior_description", "")
                        )
                        # Re-enforce box rules after override
                        enforce_box_rules(pos)
                    else:
                        pos["verification_status"] = "confirmed_stocked"

        except concurrent.futures.TimeoutError:
            print("=== Verification timed out — using pass 1 results for remaining ===")
            for future, pos in future_to_pos.items():
                if not future.done():
                    pos["verification_status"] = "timeout"
        except Exception as e:
            print(f"=== Verification error: {e} ===")

    # Recalculate summary totals after verification overrides
    total_items = 0
    total_skus = 0
    total_rows = 0
    for row in parsed.get("rows", []):
        total_rows += 1
        for pos in row.get("positions", []):
            total_skus += 1
            total_items += pos.get("estimated_count", 0)

    parsed["summary"] = {
        "total_rows": total_rows,
        "total_distinct_skus": total_skus,
        "total_items": total_items,
    }

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
