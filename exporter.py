"""Export the agent's report to markdown or PDF."""

import html
import re

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_ORDERED_RE = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def _inline(text: str) -> str:
    """Markdown inline formatting -> ReportLab's mini-HTML, XML-escaped first."""
    out = html.escape(text, quote=False)
    out = _CODE_RE.sub(r'<font face="Courier">\1</font>', out)
    out = _BOLD_RE.sub(r"<b>\1</b>", out)
    out = _ITALIC_RE.sub(r"<i>\1</i>", out)
    out = _LINK_RE.sub(r'<link href="\2" color="blue">\1</link>', out)
    return out


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _build_table(rows: list[list[str]], styles) -> Table:
    cell_style = ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=8.5, leading=11, alignment=TA_LEFT)
    header_style = ParagraphStyle("CellHeader", parent=cell_style, textColor=colors.white, fontName="Helvetica-Bold")

    width = max(len(row) for row in rows)
    data = [
        [Paragraph(_inline(cell), header_style if index == 0 else cell_style) for cell in row + [""] * (width - len(row))]
        for index, row in enumerate(rows)
    ]

    table = Table(data, colWidths=[(A4[0] - 4 * cm) / width] * width, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f4f7f")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b8c4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def export_to_pdf(report_text: str, output_path: str, title: str = "Deep Research AI Report") -> str:
    """Render a markdown report to a PDF, preserving headings, lists, and tables."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        title=title,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    heading_styles = {
        1: ParagraphStyle("H1", parent=styles["Heading1"], spaceBefore=14, spaceAfter=8),
        2: ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6),
        3: ParagraphStyle("H3", parent=styles["Heading3"], spaceBefore=10, spaceAfter=4),
    }
    body = ParagraphStyle("Body", parent=styles["BodyText"], leading=14, spaceAfter=6)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=3)

    story = [Paragraph(html.escape(title), styles["Title"]), Spacer(1, 14)]
    table_rows: list[list[str]] = []
    in_code_block = False

    def flush_table() -> None:
        if table_rows:
            story.append(Spacer(1, 6))
            story.append(KeepTogether(_build_table(list(table_rows), styles)))
            story.append(Spacer(1, 10))
            table_rows.clear()

    for raw_line in report_text.splitlines():
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            flush_table()
            in_code_block = not in_code_block
            continue

        if in_code_block:
            story.append(Paragraph(f'<font face="Courier" size="8">{html.escape(line) or "&nbsp;"}</font>', body))
            continue

        if not line.strip():
            flush_table()
            continue

        if _TABLE_SEPARATOR_RE.match(line):
            continue

        if "|" in line and line.strip().startswith("|"):
            table_rows.append(_split_row(line))
            continue

        flush_table()

        heading = _HEADING_RE.match(line)
        if heading:
            level = min(len(heading.group(1)), 3)
            story.append(Paragraph(_inline(heading.group(2)), heading_styles[level]))
            continue

        bullet_match = _BULLET_RE.match(line)
        if bullet_match:
            story.append(Paragraph(_inline(bullet_match.group(1)), bullet, bulletText="\u2022"))
            continue

        ordered_match = _ORDERED_RE.match(line)
        if ordered_match:
            story.append(Paragraph(_inline(ordered_match.group(2)), bullet, bulletText=f"{ordered_match.group(1)}."))
            continue

        story.append(Paragraph(_inline(line), body))

    flush_table()
    doc.build(story or [Paragraph("Empty report.", body)])
    return output_path
