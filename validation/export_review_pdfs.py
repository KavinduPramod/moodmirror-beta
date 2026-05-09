"""Export review_sample JSON files to clean, readable PDFs."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CaseTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            spaceBefore=12,
            spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#444444"),
            spaceAfter=4,
        ),
    }


def _build_summary_table(summary: dict, styles: dict):
    rows = [[Paragraph("<b>Metric</b>", styles["body"]), Paragraph("<b>Value</b>", styles["body"])]]
    for key, value in summary.items():
        label = key.replace("_", " ").capitalize()
        rows.append([Paragraph(_clean_text(label), styles["body"]), Paragraph(_clean_text(value), styles["body"])])

    table = Table(rows, colWidths=[65 * mm, 105 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f3f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7c7c7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_case_pdf(
    input_json: Path,
    output_pdf: Path,
    include_automated: bool = True,
    user_only: bool = False,
) -> None:
    with input_json.open("r", encoding="utf-8") as f:
        case = json.load(f)

    styles = _build_styles()
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"MoodMirror Review {case.get('case_id', input_json.stem)}",
        author="MoodMirror Validation",
    )

    story = []
    story.append(Paragraph(_clean_text(case.get("case_id", input_json.stem)), styles["title"]))
    if not user_only:
        story.append(Paragraph(_clean_text(case.get("_reviewer_instructions", "")), styles["small"]))

    summary = case.get("behavioral_summary", {})
    story.append(Paragraph("Behavioral Summary", styles["h2"]))
    story.append(_build_summary_table(summary, styles))

    posts = case.get("sample_posts", [])
    story.append(Paragraph("Sample Posts", styles["h2"]))
    for idx, post in enumerate(posts, start=1):
        date = _clean_text(post.get("date", ""))
        subreddit = _clean_text(post.get("subreddit", ""))
        ptype = _clean_text(post.get("type", ""))
        upvotes = _clean_text(post.get("upvotes", ""))
        text = _clean_text(post.get("text", ""))

        story.append(Paragraph(f"<b>Post {idx}</b>  |  {date}  |  r/{subreddit}  |  {ptype}  |  ↑ {upvotes}", styles["small"]))
        story.append(Paragraph(text.replace("\n", "<br/>"), styles["body"]))
        story.append(Spacer(1, 5))
        story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#d9d9d9"), spaceBefore=2, spaceAfter=5))

    if not user_only:
        classification = case.get("your_classification", {})
        story.append(Paragraph("Expert Classification", styles["h2"]))
        story.append(Paragraph(f"Label: {_clean_text(classification.get('expert_label', ''))}", styles["body"]))
        story.append(Paragraph(f"Confidence: {_clean_text(classification.get('confidence', ''))}", styles["body"]))
        story.append(Paragraph(f"Notes: {_clean_text(classification.get('notes', '')) or '______________________________'}", styles["body"]))

    if include_automated and not user_only:
        automated = case.get("automated_result", {})
        story.append(Paragraph("Automated Result", styles["h2"]))
        story.append(Paragraph(f"Automated label: {_clean_text(automated.get('automated_label', ''))}", styles["body"]))
        story.append(Paragraph(f"Confidence: {_clean_text(automated.get('label_confidence', ''))}", styles["body"]))
        story.append(Paragraph(f"Evidence: {_clean_text(automated.get('label_evidence', ''))}", styles["body"]))

    doc.build(story)


def export_folder(
    input_dir: Path,
    output_dir: Path,
    include_automated: bool,
    user_only: bool,
    pattern: str = "user_*.json",
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(input_dir / pattern)))

    created = 0
    for file_path in files:
        file_name = Path(file_path).name
        if file_name.startswith("_"):
            continue
        input_json = Path(file_path)
        output_pdf = output_dir / f"{input_json.stem}.pdf"
        build_case_pdf(
            input_json,
            output_pdf,
            include_automated=include_automated,
            user_only=user_only,
        )
        created += 1
        print(f"Created: {output_pdf}")

    return created


def parse_args():
    parser = argparse.ArgumentParser(description="Export MoodMirror review JSON files to individual PDFs.")
    parser.add_argument(
        "--input-dir",
        default=str(Path(__file__).parent / "review_sample"),
        help="Folder containing user_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "review_sample_pdf"),
        help="Folder to write PDF files.",
    )
    parser.add_argument(
        "--hide-automated",
        action="store_true",
        help="Do not include automated prediction section in the PDFs.",
    )
    parser.add_argument(
        "--user-only",
        action="store_true",
        help="Include only user data sections (case id, behavioral summary, sample posts).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    created = export_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        include_automated=not args.hide_automated,
        user_only=args.user_only,
    )

    if created == 0:
        print("No matching review files found.")
    else:
        print(f"Done. Generated {created} PDF file(s) in: {output_dir}")


if __name__ == "__main__":
    main()