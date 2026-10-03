"""Post-gate JSON and Markdown inspection with bounded, allowlisted output."""

import html
import json
from pathlib import Path

from .data import FIELDS, ID


def _md_cell(text):
    return (
        html.escape(str(text))
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("`", "&#96;")
    )


class CandidateInspector:
    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)

    @staticmethod
    def candidate_to_row(c):
        return {
            "id": c.id,
            "retrieval_score": c.score,
            "rerank_score": c.rerank_score,
            "document": {k: v for k, v in c.attributes.items() if k in FIELDS},
        }

    def render_markdown(self, config, candidates, top_n=10):
        lines = [
            f"# {_md_cell(config.query_id)}",
            "",
            "Stage: final-post-gate",
            "",
            _md_cell(config.nl_description),
            "",
            "## Hard criteria",
        ]
        lines.extend(f"- {_md_cell(c)}" for c in config.hard_criteria)
        lines.extend(["", "## Soft criteria"])
        lines.extend(f"- {_md_cell(c)}" for c in config.soft_criteria)
        lines.extend(
            [
                "",
                "| # | ID | Title | Kind | Retrieval | Rerank | Excerpt |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for index, c in enumerate(candidates[:top_n], 1):
            a = c.attributes
            cells = [
                index,
                c.id,
                a.get("title", ""),
                a.get("kind", ""),
                c.score,
                c.rerank_score,
                " ".join(a.get("body", "").split())[:200],
            ]
            lines.append("| " + " | ".join(_md_cell(x) for x in cells) + " |")
        for c in candidates[:top_n]:
            lines.extend(
                [
                    "",
                    f"## {_md_cell(c.id)}",
                    "",
                    "<pre>" + html.escape(c.attributes.get("body", "")) + "</pre>",
                ]
            )
        return "\n".join(lines) + "\n"

    def write(self, config, candidates, top_n=10):
        if not isinstance(config.query_id, str) or not ID.fullmatch(config.query_id):
            raise ValueError("Unsafe query identifier")
        if type(top_n) is not int or not 1 <= top_n <= 100:
            raise ValueError("Invalid inspection depth")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{config.query_id}.json"
        payload = {
            "version": 1,
            "query_id": config.query_id,
            "stage": "final-post-gate",
            "candidates": [self.candidate_to_row(c) for c in candidates[:top_n]],
        }
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        md_path = path.with_suffix(".md")
        md_path.write_text(self.render_markdown(config, candidates, top_n), encoding="utf-8")
        return md_path, path
