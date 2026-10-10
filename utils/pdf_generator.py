"""
AgriIntel — Farmer Financial Health Card PDF Generator
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String, Circle, Wedge
from reportlab.graphics import renderPDF
from io import BytesIO
import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

# ── Unicode font (supports the rupee sign, check mark and arrows).
# Built-in Helvetica has no rupee glyph and renders it as a black box.
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
pdfmetrics.registerFont(TTFont("AgriFont", os.path.join(_FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("AgriFont-Bold", os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")))
addMapping("AgriFont", 0, 0, "AgriFont")
addMapping("AgriFont", 1, 0, "AgriFont-Bold")
addMapping("AgriFont", 0, 1, "AgriFont")
addMapping("AgriFont", 1, 1, "AgriFont-Bold")
from datetime import datetime
import math

# ── Colors
NAVY   = HexColor("#0D1B2A")
CYAN   = HexColor("#00A8CC")
GOLD   = HexColor("#D4A843")
GREEN  = HexColor("#1A7A4A")
RED    = HexColor("#C0392B")
LGRAY  = HexColor("#8C9E9E")
CREAM  = HexColor("#F4F6F9")
WHITE  = white

def score_color(score: int):
    if score >= 70: return GREEN
    if score >= 50: return GOLD
    return RED

def decision_color(decision: str):
    d = decision.upper()
    if d == "APPROVE": return GREEN
    if d == "REVIEW":  return GOLD
    return RED

def draw_gauge(score: int, size: float = 80) -> Drawing:
    """Draw a semicircular gauge for AgriScore."""
    d = Drawing(size, size * 0.6)
    cx, cy = size / 2, size * 0.5

    # Background arc segments
    segments = [(0, 60, "#C0392B"), (60, 120, "#D4A843"), (120, 180, "#1A7A4A")]
    for start, end, color in segments:
        for angle in range(start, end, 2):
            rad = math.radians(180 - angle)
            x1 = cx + (size * 0.38) * math.cos(rad)
            y1 = cy + (size * 0.38) * math.sin(rad)
            x2 = cx + (size * 0.48) * math.cos(rad)
            y2 = cy + (size * 0.48) * math.sin(rad)
            from reportlab.graphics.shapes import Line
            line = Line(x1, y1, x2, y2)
            line.strokeColor = HexColor(color)
            line.strokeWidth = 2
            d.add(line)

    # Needle
    needle_angle = 180 - (score / 100 * 180)
    rad = math.radians(needle_angle)
    nx = cx + (size * 0.35) * math.cos(rad)
    ny = cy + (size * 0.35) * math.sin(rad)
    from reportlab.graphics.shapes import Line as GLine
    needle = GLine(cx, cy, nx, ny)
    needle.strokeColor = NAVY
    needle.strokeWidth = 2
    d.add(needle)

    # Center dot
    dot = Circle(cx, cy, 4)
    dot.fillColor = NAVY
    dot.strokeColor = NAVY
    d.add(dot)

    # Score text
    score_text = String(cx, cy - 18, str(score),
                        fontName="AgriFont-Bold", fontSize=16,
                        fillColor=NAVY, textAnchor="middle")
    d.add(score_text)
    label_text = String(cx, cy - 28, "/100",
                        fontName="AgriFont", fontSize=8,
                        fillColor=LGRAY, textAnchor="middle")
    d.add(label_text)
    return d


def generate_health_card(farmer_data: dict) -> bytes:
    """
    Generate a PDF health card for a farmer.
    farmer_data keys: name, phone, district, state, crop_type,
                      land_size, irrigation_type, ownership,
                      predicted_income, agri_score, decision,
                      recommended_loan, repayment_probability,
                      income_vs_district, crop_switch_suggestion,
                      matched_schemes, generated_by, branch_name
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm
    )

    styles = getSampleStyleSheet()
    elements = []

    # ── HEADER
    header_data = [[
        Paragraph(
            f'<font color="#00A8CC" size="20"><b>AgriIntel</b></font><br/>'
            f'<font color="#AAAAAA" size="9">Farmer Financial Health Card</font>',
            ParagraphStyle("h", leading=26, alignment=TA_LEFT, fontName="AgriFont")
        ),
        Paragraph(
            f'<font color="#AAAAAA" size="8">Generated: {datetime.now().strftime("%d %b %Y")}<br/>'
            f'Branch: {farmer_data.get("branch_name","—")}<br/>'
            f'Officer: {farmer_data.get("generated_by","—")}</font>',
            ParagraphStyle("hr", leading=12, alignment=TA_RIGHT, fontName="AgriFont")
        )
    ]]
    header_table = Table(header_data, colWidths=[90*mm, 90*mm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), NAVY),
        ("PADDING",     (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [NAVY]),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 6*mm))

    # ── FARMER DETAILS
    elements.append(Paragraph(
        '<font color="#0D1B2A" size="11"><b>FARMER DETAILS</b></font>',
        ParagraphStyle("sh", fontName="AgriFont-Bold",
                       borderPad=4, borderWidth=0,
                       borderColor=CYAN, leading=16)
    ))
    elements.append(HRFlowable(width="100%", thickness=2, color=CYAN))
    elements.append(Spacer(1, 3*mm))

    farmer_details = [
        ["Farmer Name", farmer_data.get("name","—"),
         "Phone",       farmer_data.get("phone","—")],
        ["District",    farmer_data.get("district","—"),
         "State",       farmer_data.get("state","—")],
        ["Crop Type",   farmer_data.get("crop_type","—"),
         "Land Size",   f'{farmer_data.get("land_size","—")} acres'],
        ["Irrigation",  farmer_data.get("irrigation_type","—"),
         "Ownership",   farmer_data.get("ownership","—")],
    ]
    detail_table = Table(farmer_details, colWidths=[35*mm, 55*mm, 35*mm, 55*mm])
    detail_table.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (-1,-1), "AgriFont"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("FONTNAME",    (0,0), (0,-1), "AgriFont-Bold"),
        ("FONTNAME",    (2,0), (2,-1), "AgriFont-Bold"),
        ("TEXTCOLOR",   (0,0), (0,-1), NAVY),
        ("TEXTCOLOR",   (2,0), (2,-1), NAVY),
        ("BACKGROUND",  (0,0), (-1,-1), CREAM),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [CREAM, WHITE]),
        ("GRID",        (0,0), (-1,-1), 0.5, HexColor("#CCCCCC")),
        ("PADDING",     (0,0), (-1,-1), 6),
    ]))
    elements.append(detail_table)
    elements.append(Spacer(1, 6*mm))

    # ── CREDIT ASSESSMENT
    elements.append(Paragraph(
        '<font color="#0D1B2A" size="11"><b>CREDIT ASSESSMENT</b></font>',
        ParagraphStyle("sh2", fontName="AgriFont-Bold", leading=16)
    ))
    elements.append(HRFlowable(width="100%", thickness=2, color=GOLD))
    elements.append(Spacer(1, 3*mm))

    score = farmer_data.get("agri_score", 0)
    decision = farmer_data.get("decision", "REVIEW")
    income = farmer_data.get("predicted_income", 0)
    loan = farmer_data.get("recommended_loan", 0)
    repay = farmer_data.get("repayment_probability", 0)
    vs_district = farmer_data.get("income_vs_district", 0)

    assessment_data = [
        [
            Paragraph(f'<font size="9" color="#AEBBC9">AgriScore</font><br/>'
                      f'<font size="24" color="{score_color(score).hexval()}"><b>{score}</b></font>'
                      f'<font size="10" color="#AEBBC9">/100</font>',
                      ParagraphStyle("sc", leading=30, alignment=TA_CENTER, fontName="AgriFont")),
            Paragraph(f'<font size="9" color="#AEBBC9">Predicted Annual Income</font><br/>'
                      f'<font size="18" color="#FFFFFF"><b>₹{income:,.0f}</b></font><br/>'
                      f'<font size="8" color="#AEBBC9">Range: ₹{income*0.85:,.0f} — ₹{income*1.15:,.0f}</font>',
                      ParagraphStyle("inc", leading=24, alignment=TA_CENTER, fontName="AgriFont")),
            Paragraph(f'<font size="9" color="#AEBBC9">Decision</font><br/>'
                      f'<font size="18" color="{decision_color(decision).hexval()}"><b>{decision}</b></font>',
                      ParagraphStyle("dec", leading=24, alignment=TA_CENTER, fontName="AgriFont")),
        ],
        [
            Paragraph(f'<font size="9" color="#AEBBC9">Recommended Loan</font><br/>'
                      f'<font size="16" color="#D4A843"><b>₹{loan:,.0f}</b></font>',
                      ParagraphStyle("loan", leading=22, alignment=TA_CENTER, fontName="AgriFont")),
            Paragraph(f'<font size="9" color="#AEBBC9">Repayment Score (est.)</font><br/>'
                      f'<font size="16" color="#1A7A4A"><b>{repay:.0f}%</b></font>',
                      ParagraphStyle("rep", leading=22, alignment=TA_CENTER, fontName="AgriFont")),
            Paragraph(f'<font size="9" color="#AEBBC9">vs District Average</font><br/>'
                      f'<font size="16" color="{"#1A7A4A" if vs_district >= 0 else "#C0392B"}"><b>'
                      f'{"+" if vs_district >= 0 else ""}{vs_district:.1f}%</b></font>',
                      ParagraphStyle("dist", leading=22, alignment=TA_CENTER, fontName="AgriFont")),
        ]
    ]
    assess_table = Table(assessment_data, colWidths=[60*mm, 60*mm, 60*mm])
    assess_table.setStyle(TableStyle([
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND",  (0,0), (-1,-1), NAVY),
        ("GRID",        (0,0), (-1,-1), 0.5, HexColor("#1B4F8A")),
        ("PADDING",     (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [NAVY, HexColor("#111C2B")]),
    ]))
    elements.append(assess_table)
    elements.append(Spacer(1, 6*mm))

    # ── RECOMMENDATIONS
    elements.append(Paragraph(
        '<font color="#0D1B2A" size="11"><b>RECOMMENDATIONS</b></font>',
        ParagraphStyle("sh3", fontName="AgriFont-Bold", leading=16)
    ))
    elements.append(HRFlowable(width="100%", thickness=2, color=GREEN))
    elements.append(Spacer(1, 3*mm))

    recs = list(farmer_data.get("recommendations", []))
    if not recs:
        recs = ["Maintain current farming practices to preserve income stability."]

    crop_switch = farmer_data.get("crop_switch_suggestion")
    if crop_switch:
        recs.insert(0, f"Consider switching to {crop_switch['crop']} — "
                       f"indicative income uplift of ~₹{crop_switch['uplift']:,.0f}/year "
                       f"(validate with local prices and risk)")

    rec_data = [[
        Paragraph(f"{'✓' if i < 2 else '→'}  {rec}",
                  ParagraphStyle(f"rec{i}", fontName="AgriFont", fontSize=9,
                                 leftIndent=5, leading=14))
    ] for i, rec in enumerate(recs[:5])]

    rec_table = Table(rec_data, colWidths=[180*mm])
    rec_table.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [CREAM, WHITE]),
        ("PADDING",        (0,0), (-1,-1), 6),
        ("GRID",           (0,0), (-1,-1), 0.3, HexColor("#CCCCCC")),
    ]))
    elements.append(rec_table)
    elements.append(Spacer(1, 6*mm))

    # ── ELIGIBLE GOVERNMENT SCHEMES
    schemes = farmer_data.get("matched_schemes", [])
    if schemes:
        elements.append(Paragraph(
            '<font color="#0D1B2A" size="11"><b>ELIGIBLE GOVERNMENT SCHEMES</b></font>',
            ParagraphStyle("sh4", fontName="AgriFont-Bold", leading=16)
        ))
        elements.append(HRFlowable(width="100%", thickness=2, color=CYAN))
        elements.append(Spacer(1, 3*mm))

        scheme_data = [["Scheme Name", "Benefit", "How to Apply"]]
        for s in schemes[:4]:
            scheme_data.append([s["name"], s["benefit"], s["apply"]])

        scheme_table = Table(scheme_data, colWidths=[60*mm, 70*mm, 50*mm])
        scheme_table.setStyle(TableStyle([
            ("FONTNAME",    (0,0), (-1,-1), "AgriFont"),
            ("BACKGROUND",  (0,0), (-1,0), NAVY),
            ("TEXTCOLOR",   (0,0), (-1,0), CYAN),
            ("FONTNAME",    (0,0), (-1,0), "AgriFont-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [CREAM, WHITE]),
            ("GRID",        (0,0), (-1,-1), 0.5, HexColor("#CCCCCC")),
            ("PADDING",     (0,0), (-1,-1), 5),
        ]))
        elements.append(scheme_table)
        elements.append(Spacer(1, 4*mm))

    # ── FOOTER
    elements.append(HRFlowable(width="100%", thickness=1, color=LGRAY))
    elements.append(Paragraph(
        '<font size="7" color="#888888">This report is generated by AgriIntel AI System. '
        'Income predictions are estimates based on ML models and should be used as a '
        'decision-support tool only. Final credit decisions rest with the bank officer. '
        'AgriIntel • Student project, IIT Guwahati • Confidential</font>',
        ParagraphStyle("footer", alignment=TA_CENTER, fontName="AgriFont",
                       fontSize=7, leading=10)
    ))

    doc.build(elements)
    return buffer.getvalue()
