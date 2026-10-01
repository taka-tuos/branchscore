#!/usr/bin/env python3
"""Render fixed Step 4 evidence with ImageMagick; Python only orchestrates it."""

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "phase4-step4-vision.jsonl"
CONVERT = shutil.which("magick") or shutil.which("convert")
MONTAGE = [CONVERT, "montage"] if CONVERT and Path(CONVERT).name == "magick" else [shutil.which("montage")]
SIZE = (960, 720)
PALETTE = {"red": "#dc3545", "blue": "#2563eb", "green": "#16a34a"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Canvas:
    def __init__(self, title, subtitle, relayout=False):
        self.commands = []
        self.rect(0, 0, 959, 719, "#f1f5f9")
        self.rect(0, 0, 959, 126, "#14253d")
        self.rect(32, 28, 39, 96, "#2dd4bf" if relayout else "#60a5fa")
        self.text(58, 64, title, 32, "#ffffff", bold=True)
        self.text(59, 97, subtitle, 19, "#cbd5e1")
        self.text(34, 699, "DECISION LAB   /   visual evidence", 16, "#64748b")
        self.text(805, 699, "SNAPSHOT", 16, "#64748b")

    def rect(self, x1, y1, x2, y2, fill, stroke="none", radius=0):
        primitive = f"roundrectangle {x1},{y1} {x2},{y2} {radius},{radius}" if radius else f"rectangle {x1},{y1} {x2},{y2}"
        self.commands.append(f"fill '{fill}' stroke '{stroke}' stroke-width 1 {primitive}")

    def text(self, x, y, text, size=24, color="#172b44", bold=False, mono=False):
        font = "DejaVu-Sans-Mono" if mono else "DejaVu-Sans-Bold" if bold else "DejaVu-Sans"
        # All image text is fixed ASCII. Quoting goes into MVG, never a shell.
        self.commands.append(f"font '{font}' font-size {size} fill '{color}' stroke 'none' text {x},{y} {json.dumps(text)}")

    def shape(self, x, y, kind, color):
        fill = PALETTE[color]
        prefix = f"fill '{fill}' stroke '#172b44' stroke-width 1.5 "
        if kind == "square":
            draw = f"rectangle {x-17},{y-17} {x+17},{y+17}"
        elif kind == "circle":
            draw = f"circle {x},{y} {x+18},{y}"
        else:
            draw = f"polygon {x},{y-20} {x-20},{y+17} {x+20},{y+17}"
        self.commands.append(prefix + draw)

    def save(self, path):
        subprocess.run([CONVERT, "-size", "960x720", "xc:white", "-draw", "\n".join(self.commands),
                        "-colorspace", "sRGB", "-depth", "8", "-strip",
                        "-define", "png:color-type=2", str(path)], check=True)


def objects(region, red_squares, other):
    kinds = [("red", "square")] * red_squares + other
    return [{"id": f"{region}-{i}", "region": region, "slot": i, "color": color, "shape": shape}
            for i, (color, shape) in enumerate(kinds)]


BOARD = sum([
    objects("north", 3, [("red", "circle"), ("blue", "square"), ("green", "triangle")]),
    objects("east", 4, [("red", "circle"), ("blue", "square")]),
    objects("south", 2, [("red", "circle"), ("red", "circle"), ("blue", "square"), ("green", "triangle")]),
    objects("west", 1, [("red", "circle"), ("blue", "square"), ("blue", "square"), ("green", "triangle"), ("green", "triangle")]),
], [])
JOBS = [
    {"id": "orion", "state": "FAILED", "response": "ACK", "severity": "HIGH"},
    {"id": "lyra", "state": "FAILED", "response": "OPEN", "severity": "HIGH"},
    {"id": "vega", "state": "FAILED", "response": "OPEN", "severity": "LOW"},
    {"id": "nova", "state": "OK", "response": "OPEN", "severity": "HIGH"},
    {"id": "atlas", "state": "PAUSED", "response": "OPEN", "severity": "MEDIUM"},
]
CANDIDATES = [
    {"id": "aster", "status": "READY", "quality": 85, "cost": 120},
    {"id": "birch", "status": "READY", "quality": 84, "cost": 110},
    {"id": "cedar", "status": "HOLD", "quality": 92, "cost": 95},
    {"id": "dahlia", "status": "READY", "quality": 79, "cost": 100},
    {"id": "elm", "status": "READY", "quality": 88, "cost": 130},
]


def board(facts, relayout=False):
    c = Canvas("VISUAL INVENTORY", "Four regions / mixed colors and shapes", relayout)
    order = ["south", "north", "west", "east"] if relayout else ["north", "east", "south", "west"]
    for i, name in enumerate(order):
        x, y = 32 + (i % 2) * 464, 148 + (i // 2) * 260
        c.rect(x+3, y+4, x+435, y+246, "#e2e8f0", radius=14)
        c.rect(x, y, x+432, y+242, "#ffffff", "#cbd5e1", 14)
        c.text(x+22, y+35, name.upper(), 24, bold=True)
        c.rect(x+18, y+50, x+414, y+51, "#e2e8f0")
        for obj in (o for o in facts if o["region"] == name):
            slot = obj["slot"]
            c.shape(x+78+(slot % 3)*136, y+82+(slot // 3)*53, obj["shape"], obj["color"])
    return c


def grid(title, subtitle, labels, relayout):
    c = Canvas(title, subtitle, relayout)
    c.rect(32, 148, 928, 656, "#ffffff", "#cbd5e1", 14)
    c.rect(49, 166, 910, 205, "#edf2f7", radius=5)
    for x, label in labels:
        c.text(x, 192, label, 18, "#475569", bold=True)
    return c


def pill(c, x, y, width, label, fill, ink, size=21):
    c.rect(x, y, x+width, y+43, fill, radius=7)
    c.text(x+13, y+29, label, size, ink, bold=True)


def status(facts, relayout=False):
    c = grid("OPERATIONS MONITOR", "Job state / response / severity", [(72, "JOB"), (281, "STATE"), (493, "RESPONSE"), (749, "SEVERITY")], relayout)
    order = ["vega", "atlas", "orion", "nova", "lyra"] if relayout else ["orion", "lyra", "vega", "nova", "atlas"]
    by_id = {r["id"]: r for r in facts}
    for i, name in enumerate(order):
        r, y = by_id[name], 222+i*83
        if i % 2 == 0:
            c.rect(50, y-5, 910, y+63, "#f8fafc", radius=4)
        c.text(72, y+34, name.upper(), 27, bold=True)
        fill, ink = {"FAILED": ("#fee2e2", "#b91c1c"), "OK": ("#dcfce7", "#166534"), "PAUSED": ("#e2e8f0", "#475569")}[r["state"]]
        pill(c, 275, y+5, 145, r["state"], fill, ink)
        fill, ink = ("#fef3c7", "#92400e") if r["response"] == "OPEN" else ("#dbeafe", "#1d4ed8")
        pill(c, 486, y+5, 173, r["response"], fill, ink)
        fill, ink = {"HIGH": ("#fee2e2", "#b91c1c"), "MEDIUM": ("#fef3c7", "#92400e"), "LOW": ("#dbeafe", "#1d4ed8")}[r["severity"]]
        pill(c, 741, y+5, 154, r["severity"], fill, ink, 20)
        c.rect(65, y+72, 895, y+73, "#e2e8f0")
    return c


def table(facts, relayout=False):
    c = grid("CANDIDATE COMPARISON", "Availability / quality points / cost units", [(72, "CANDIDATE"), (292, "STATUS"), (510, "QUALITY"), (758, "COST")], relayout)
    order = ["elm", "dahlia", "birch", "cedar", "aster"] if relayout else ["aster", "birch", "cedar", "dahlia", "elm"]
    by_id = {r["id"]: r for r in facts}
    for i, name in enumerate(order):
        r, y = by_id[name], 222+i*83
        if i % 2 == 0:
            c.rect(50, y-5, 910, y+63, "#f8fafc", radius=4)
        c.text(72, y+34, name.upper(), 27, bold=True)
        fill, ink = ("#dcfce7", "#166534") if r["status"] == "READY" else ("#fef3c7", "#92400e")
        pill(c, 284, y+5, 147, r["status"], fill, ink)
        c.text(513, y+34, str(r["quality"]), 29, mono=True)
        c.rect(572, y+20, 700, y+31, "#e2e8f0", radius=4)
        c.rect(572, y+20, 572+round(128*r["quality"]/100), y+31, "#0d9488" if relayout else "#2563eb", radius=4)
        c.text(756, y+34, str(r["cost"]), 29, mono=True)
        c.rect(65, y+72, 895, y+73, "#e2e8f0")
    return c


def winner(family, facts):
    if family == "board":
        counts = {name: sum(o["region"] == name and o["color"] == "red" and o["shape"] == "square" for o in facts)
                  for name in ("north", "east", "south", "west")}
        names = [name for name, count in counts.items() if count == max(counts.values())]
        assert len(names) == 1
        return names[0], "Red-square counts by printed region: " + ", ".join(f"{n}={v}" for n, v in counts.items()) + ". The maximum is unique."
    if family == "status":
        eligible = [r for r in facts if r["state"] == "FAILED" and r["response"] == "OPEN"]
        ranks = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        names = [r["id"] for r in eligible if ranks[r["severity"]] == max(ranks[s["severity"]] for s in eligible)]
        assert len(names) == 1
        return names[0], "Eligible FAILED/OPEN jobs: " + ", ".join(f"{r['id']} ({r['severity']})" for r in eligible) + ". Select the uniquely highest severity."
    eligible = [r for r in facts if r["status"] == "READY" and r["quality"] >= 80]
    names = [r["id"] for r in eligible if r["cost"] == min(s["cost"] for s in eligible)]
    assert len(names) == 1
    return names[0], "Eligible READY, quality>=80 candidates: " + ", ".join(f"{r['id']} (cost {r['cost']})" for r in eligible) + ". Select the uniquely lowest cost."


def options(names, verb):
    return [{"id": n, "description": f"{verb} {n.upper()}."} for n in names]


def main():
    if not CONVERT or not MONTAGE[0]:
        raise SystemExit("ImageMagick convert/montage (or magick) is required")
    renderers = {"board": board, "status": status, "table": table}
    originals = {"board": BOARD, "status": JOBS, "table": CANDIDATES}
    moved = copy.deepcopy(BOARD)
    item = next(o for o in moved if o["id"] == "east-3")
    item.update(region="north", slot=6)
    acknowledged = copy.deepcopy(JOBS)
    next(r for r in acknowledged if r["id"] == "lyra")["response"] = "ACK"
    below_threshold = copy.deepcopy(CANDIDATES)
    next(r for r in below_threshold if r["id"] == "birch")["quality"] = 78
    changed = {"board": moved, "status": acknowledged, "table": below_threshold}
    changed_fact = {"board": "Move the single red square east-3 from EAST to NORTH; preserve all other objects.",
                    "status": "Change only LYRA's RESPONSE from OPEN to ACK.",
                    "table": "Change only BIRCH's QUALITY from 84 to 78 (including its quality bar)."}
    states = {"board": "Use only the supplied image as evidence. Count red squares inside each card. Ignore circles, triangles, and squares of other colors. Identify each region by its printed name, not its screen position.",
              "status": "Use only the supplied image as evidence. A job needs intervention only if STATE is FAILED and RESPONSE is OPEN. Among eligible jobs choose the highest SEVERITY: HIGH before MEDIUM before LOW. If none are eligible, choose no intervention.",
              "table": "Use only the supplied image as evidence. A candidate is eligible only if STATUS is READY and QUALITY is at least 80. Among eligible candidates choose the one with the lowest COST."}
    questions = {"board": "Which region contains the most red squares?", "status": "Which job should receive intervention first?", "table": "Which candidate should be selected?"}
    choices = {"board": options(["west", "east", "north", "south"], "Select region"),
               "status": options(["atlas", "vega", "orion", "lyra", "nova"], "Intervene on job") + [{"id": "none", "description": "Choose no intervention."}],
               "table": options(["birch", "cedar", "elm", "dahlia", "aster"], "Select candidate")}
    extras = {"board": [{"id": "tied", "description": "All four regions are tied."}, {"id": "empty", "description": "No region contains any red squares."}],
              "status": [{"id": "all", "description": "Intervene on every job equally."}, {"id": "defer", "description": "Defer choosing an eligible job until another snapshot."}],
              "table": [{"id": "all", "description": "Select every candidate equally."}, {"id": "defer", "description": "Defer choosing an eligible candidate."}]}
    expected = {"board": ["east", "north", "east"], "status": ["lyra", "vega", "lyra"], "table": ["birch", "aster", "birch"]}
    rows, images, montage_args = [], [], []
    for family in ("board", "status", "table"):
        for i, variation in enumerate(("base", "changed", "relayout")):
            facts = copy.deepcopy(changed[family] if variation == "changed" else originals[family])
            filename = f"{family}-{variation}.png"
            path = HERE / filename
            renderers[family](facts, variation == "relayout").save(path)
            answer, reason = winner(family, facts)
            assert answer == expected[family][i]
            image_id = f"vision-{family}-{variation}"
            images.append({"id": image_id, "file": filename, "dimensions": list(SIZE), "sha256": sha(path),
                           "family": family, "variation": variation, "facts": facts,
                           "expected_selected_id": answer, "expected_reason": reason,
                           "compared_with": f"vision-{family}-base" if variation != "base" else None,
                           "relation": "selection_changes" if variation == "changed" else "selection_preserved" if variation == "relayout" else "base",
                           "change": changed_fact[family] if variation == "changed" else "Reorder cards/rows and change decorative accent; preserve evidence values." if variation == "relayout" else None})
            for order in ("base", "reversed", "distractors"):
                ordered = copy.deepcopy(choices[family])
                if order == "reversed":
                    ordered.reverse()
                elif order == "distractors":
                    ordered = [copy.deepcopy(extras[family][0])] + ordered + [copy.deepcopy(extras[family][1])]
                rows.append({"id": image_id + ("" if order == "base" else "-" + order), "fixture_group": image_id,
                             "visual_family": family, "visual_variant": variation, "variant": order,
                             "verification_role": "vision_semantic", "state": states[family], "question": questions[family],
                             "image": "phase4-step4-vision/" + filename, "options": ordered,
                             "expected_selected_id": answer, "expected_reason": reason})
            montage_args += ["-label", family.upper() + " / " + variation.upper(), str(path)]
    FIXTURE.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows))
    subprocess.run(MONTAGE + montage_args + ["-font", "DejaVu-Sans", "-pointsize", "18", "-fill", "#172b44",
                   "-background", "#f1f5f9", "-tile", "3x3", "-geometry", "320x240+12+12", "-depth", "8",
                   "-strip", str(HERE / "contact-sheet.png")], check=True)
    manifest = {"schema_version": 1, "created_date_jst": "2026-10-01", "generator": "generate.py",
                "imagemagick_version": subprocess.check_output([CONVERT, "-version"], text=True).splitlines()[0],
                "fonts": ["DejaVu-Sans", "DejaVu-Sans-Bold", "DejaVu-Sans-Mono"],
                "fixture_file": "../phase4-step4-vision.jsonl", "fixture_sha256": sha(FIXTURE),
                "image_count": len(images), "request_rows": len(rows), "independent_families": 3,
                "distinct_evidence_scenarios": 6, "invariance_derivatives": 3,
                "images": images, "preview_file": "contact-sheet.png", "preview_sha256": sha(HERE / "contact-sheet.png")}
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Rendered {len(images)} evidence images and {len(rows)} request rows with ImageMagick.")


if __name__ == "__main__":
    main()
