# ================================================================
# RETINASCAN AI — PDF Report Generator
# ================================================================
# Generates professional medical-AI research reports from analysis
# results using reportlab.
# ================================================================

import io
import base64
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF


# ================================================================
# COLOR PALETTE
# ================================================================
NAVY = colors.HexColor("#0C1D30")
NAVY_LIGHT = colors.HexColor("#1B3A5C")
TEAL = colors.HexColor("#0D9488")
TEAL_LIGHT = colors.HexColor("#CCFBF1")
GRAY_50 = colors.HexColor("#F9FAFB")
GRAY_100 = colors.HexColor("#F2F4F7")
GRAY_200 = colors.HexColor("#EAECF0")
GRAY_400 = colors.HexColor("#98A2B3")
GRAY_500 = colors.HexColor("#667085")
GRAY_600 = colors.HexColor("#475467")
GRAY_700 = colors.HexColor("#344054")
GRAY_800 = colors.HexColor("#1D2939")
GRAY_900 = colors.HexColor("#101828")
WHITE = colors.white

# Class colors
CLASS_COLORS = {
    "CNV": colors.HexColor("#F59E0B"),
    "DME": colors.HexColor("#EF4444"),
    "DRUSEN": colors.HexColor("#6366F1"),
    "NORMAL": colors.HexColor("#22C55E"),
}


# ================================================================
# STYLES
# ================================================================
def get_report_styles():
    """Create custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='ReportTitle',
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=28,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name='ReportSubtitle',
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        textColor=TEAL,
        alignment=TA_CENTER,
        spaceAfter=20,
    ))

    styles.add(ParagraphStyle(
        name='SectionHeading',
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=18,
        textColor=NAVY,
        spaceBefore=16,
        spaceAfter=8,
        borderWidth=0,
        borderPadding=0,
    ))

    styles.add(ParagraphStyle(
        name='SubHeading',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=GRAY_700,
        spaceBefore=8,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name='BodyText2',
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=GRAY_700,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name='SmallText',
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=GRAY_500,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name='DisclaimerText',
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=GRAY_500,
        alignment=TA_JUSTIFY,
        spaceBefore=8,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name='PredictionClass',
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=26,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=2,
    ))

    styles.add(ParagraphStyle(
        name='ConfidenceValue',
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=TEAL,
        alignment=TA_CENTER,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name='FooterText',
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=GRAY_400,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name='CaptionText',
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=GRAY_500,
        alignment=TA_CENTER,
        spaceAfter=4,
    ))

    return styles


# ================================================================
# HELPER: base64 → ReportLab Image
# ================================================================
def b64_to_rl_image(b64_string, width=None, height=None):
    """Convert a base64 PNG string to a ReportLab Image object."""
    if not b64_string:
        return None
    try:
        img_data = base64.b64decode(b64_string)
        img_buffer = io.BytesIO(img_data)
        if width and height:
            return RLImage(img_buffer, width=width, height=height)
        elif width:
            return RLImage(img_buffer, width=width)
        else:
            return RLImage(img_buffer)
    except Exception:
        return None


# ================================================================
# HEADER / FOOTER
# ================================================================
def header_footer(canvas, doc):
    """Draw header and footer on each page."""
    canvas.saveState()
    width, height = A4

    # Header line
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(1.5)
    canvas.line(20 * mm, height - 18 * mm, width - 20 * mm, height - 18 * mm)

    # Header text
    canvas.setFont('Helvetica-Bold', 8)
    canvas.setFillColor(NAVY)
    canvas.drawString(20 * mm, height - 16 * mm, "RETINASCAN AI")
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(GRAY_500)
    canvas.drawRightString(width - 20 * mm, height - 16 * mm, "Explainable AI for Retinal OCT Analysis")

    # Footer line
    canvas.setStrokeColor(GRAY_200)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 16 * mm, width - 20 * mm, 16 * mm)

    # Footer text
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(GRAY_400)
    canvas.drawString(20 * mm, 11 * mm, "RetinaScan AI · Research / Decision Support Application")
    canvas.drawRightString(width - 20 * mm, 11 * mm, f"Page {doc.page}")

    canvas.restoreState()


# ================================================================
# SECTION DIVIDER
# ================================================================
def section_divider():
    """Create a horizontal rule for section separation."""
    return HRFlowable(
        width="100%",
        thickness=0.5,
        color=GRAY_200,
        spaceBefore=6,
        spaceAfter=6,
    )


# ================================================================
# MAIN REPORT GENERATION
# ================================================================
def generate_report_pdf(record):
    """
    Generate a professional PDF report from an analysis record.

    Args:
        record: dict with analysis results (from history or live analysis)

    Returns:
        bytes: PDF file content
    """
    buffer = io.BytesIO()
    styles = get_report_styles()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=24 * mm,
        bottomMargin=22 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        title=f"RetinaScan AI Report - {record.get('scan_id', 'Unknown')}",
        author="RetinaScan AI",
    )

    story = []

    # ── TITLE BLOCK ──
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("RETINASCAN AI", styles['ReportTitle']))
    story.append(Paragraph("Explainable AI for Retinal OCT Analysis", styles['ReportSubtitle']))
    story.append(section_divider())

    # ── REPORT INFORMATION ──
    story.append(Paragraph("REPORT INFORMATION", styles['SectionHeading']))

    scan_id = record.get('scan_id', 'N/A')
    timestamp = record.get('timestamp', datetime.now().isoformat())
    try:
        dt = datetime.fromisoformat(timestamp)
        formatted_date = dt.strftime("%B %d, %Y at %H:%M:%S")
    except Exception:
        formatted_date = timestamp[:19].replace('T', ' ')

    report_info = [
        ['Report ID', scan_id],
        ['Date', formatted_date],
        ['Analysis Status', 'Completed' if record.get('valid') else 'Rejected'],
        ['Model', 'ConvNeXt-Tiny (28.5M parameters)'],
        ['Dataset', 'OCT2017'],
        ['Grad-CAM Layer', 'model.features[-1]'],
    ]

    info_table = Table(report_info, colWidths=[45 * mm, 115 * mm])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), GRAY_600),
        ('TEXTCOLOR', (1, 0), (1, -1), GRAY_800),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 0), (-1, -2), 0.3, GRAY_200),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 4 * mm))

    # ── INPUT VALIDATION ──
    story.append(section_divider())
    story.append(Paragraph("INPUT VALIDATION", styles['SectionHeading']))

    input_validation = record.get('input_validation', {})
    validation_status = input_validation.get('status', 'valid' if record.get('valid') else 'rejected')

    val_data = [
        ['Validation Status', validation_status.upper()],
    ]

    feature_sim = input_validation.get('feature_similarity')
    structural_score = input_validation.get('structural_score')
    combined_score = input_validation.get('combined_score')

    if feature_sim is not None:
        val_data.append(['Feature Similarity', f"{feature_sim:.4f}"])
    if structural_score is not None:
        val_data.append(['Structural OCT Score', f"{structural_score:.4f}"])
    if combined_score is not None:
        val_data.append(['Combined Domain Score', f"{combined_score:.4f}"])

    val_table = Table(val_data, colWidths=[45 * mm, 115 * mm])
    val_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), GRAY_600),
        ('TEXTCOLOR', (1, 0), (1, 0),
         colors.HexColor("#16A34A") if validation_status == 'valid' else colors.HexColor("#DC2626")),
        ('TEXTCOLOR', (1, 1), (1, -1), GRAY_800),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 0), (-1, -2), 0.3, GRAY_200),
    ]))
    story.append(val_table)

    # If rejected, show rejection info and stop
    if not record.get('valid'):
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("ANALYSIS NOT PERFORMED", styles['SectionHeading']))
        reason = input_validation.get('reason', 'Uploaded image did not pass retinal OCT input validation.')
        story.append(Paragraph(
            f"The analysis was not completed because the uploaded image did not pass "
            f"the input validation pipeline.",
            styles['BodyText2']
        ))
        story.append(Paragraph(f"<b>Reason:</b> {reason}", styles['BodyText2']))
        story.append(Spacer(1, 8 * mm))
        story.append(section_divider())
        story.append(Paragraph("DISCLAIMER", styles['SectionHeading']))
        story.append(Paragraph(
            "This system is intended for research and decision-support purposes and "
            "is not a substitute for professional clinical diagnosis.",
            styles['DisclaimerText']
        ))
        doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
        return buffer.getvalue()

    # ── MODEL PREDICTION ──
    story.append(Spacer(1, 2 * mm))
    story.append(section_divider())
    story.append(Paragraph("MODEL PREDICTION", styles['SectionHeading']))

    pred_class = record.get('prediction', 'N/A')
    pred_full = record.get('prediction_full', '')
    confidence = record.get('confidence', 0)

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(f"{pred_class}", styles['PredictionClass']))
    story.append(Paragraph(f"{pred_full}", styles['CaptionText']))
    story.append(Paragraph(f"Confidence: {confidence}%", styles['ConfidenceValue']))
    story.append(Spacer(1, 4 * mm))

    # Class probabilities table
    story.append(Paragraph("Class Probabilities", styles['SubHeading']))

    probabilities = record.get('probabilities', {})
    prob_data = [['Class', 'Probability']]
    for cls in ['CNV', 'DME', 'DRUSEN', 'NORMAL']:
        prob_val = probabilities.get(cls, 0)
        prob_data.append([cls, f"{prob_val}%"])

    prob_table = Table(prob_data, colWidths=[80 * mm, 80 * mm])
    prob_table.setStyle(TableStyle([
        # Header row
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        # Data rows
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TEXTCOLOR', (0, 1), (0, -1), GRAY_700),
        ('TEXTCOLOR', (1, 1), (1, -1), GRAY_800),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ('LINEBELOW', (0, 1), (-1, -2), 0.3, GRAY_200),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        # Alternating rows
        ('BACKGROUND', (0, 2), (-1, 2), GRAY_50),
        ('BACKGROUND', (0, 4), (-1, 4), GRAY_50),
        # Border
        ('BOX', (0, 0), (-1, -1), 0.5, GRAY_200),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(
        f"The model predicted <b>{pred_class}</b> ({pred_full}) with <b>{confidence}%</b> confidence.",
        styles['BodyText2']
    ))

    clinical_desc = record.get('clinical_description', '')
    if clinical_desc:
        story.append(Paragraph(f"<b>Clinical Context:</b> {clinical_desc}", styles['BodyText2']))

    # ── GRAD-CAM EXPLAINABILITY ──
    story.append(section_divider())
    story.append(Paragraph("GRAD-CAM EXPLAINABILITY", styles['SectionHeading']))

    # Try to include images
    img_width = 50 * mm
    img_height = 50 * mm

    gradcam_images = []
    labels = []

    original_b64 = record.get('original_b64')
    heatmap_b64 = record.get('heatmap_b64')
    overlay_b64 = record.get('overlay_b64')

    if original_b64:
        img = b64_to_rl_image(original_b64, width=img_width, height=img_height)
        if img:
            gradcam_images.append(img)
            labels.append('Original OCT')

    if heatmap_b64:
        img = b64_to_rl_image(heatmap_b64, width=img_width, height=img_height)
        if img:
            gradcam_images.append(img)
            labels.append('Grad-CAM Heatmap')

    if overlay_b64:
        img = b64_to_rl_image(overlay_b64, width=img_width, height=img_height)
        if img:
            gradcam_images.append(img)
            labels.append('Grad-CAM Overlay')

    if gradcam_images:
        # Create a table with images side by side
        img_table = Table(
            [gradcam_images, labels],
            colWidths=[55 * mm] * len(gradcam_images),
        )
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, 1), 8),
            ('TEXTCOLOR', (0, 1), (-1, 1), GRAY_500),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(img_table)
    else:
        story.append(Paragraph(
            "Grad-CAM images are not available for this analysis.",
            styles['SmallText']
        ))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Grad-CAM visualization highlights image regions that contributed to the model prediction. "
        "It should not be interpreted as a definitive anatomical or clinical diagnosis.",
        styles['DisclaimerText']
    ))

    # ── MODEL INTERPRETATION ──
    attention_region = record.get('attention_region', '')
    interpretation = record.get('interpretation', '')

    if attention_region or interpretation:
        story.append(section_divider())
        story.append(Paragraph("MODEL INTERPRETATION", styles['SectionHeading']))

        if attention_region:
            story.append(Paragraph(
                f"<b>Attention Region:</b> {attention_region}",
                styles['BodyText2']
            ))

        if interpretation:
            story.append(Paragraph(
                f"<b>Interpretation:</b> {interpretation}",
                styles['BodyText2']
            ))

        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph(
            f"The model predicted <b>{pred_class}</b> with <b>{confidence}%</b> confidence. "
            f"This is a model output and should be reviewed by a qualified clinician.",
            styles['BodyText2']
        ))

    # ── DISCLAIMER ──
    story.append(Spacer(1, 6 * mm))
    story.append(section_divider())
    story.append(Paragraph("DISCLAIMER", styles['SectionHeading']))
    story.append(Paragraph(
        "This system is intended for research and decision-support purposes and "
        "is not a substitute for professional clinical diagnosis. The predictions, "
        "confidence values, and Grad-CAM visualizations shown in this report are "
        "model outputs generated by a deep learning system (ConvNeXt-Tiny trained "
        "on the OCT2017 dataset). They should not be used as the sole basis for "
        "clinical decisions.",
        styles['DisclaimerText']
    ))
    story.append(Paragraph(
        "Grad-CAM highlights regions of model attention and does not constitute "
        "anatomical annotation or pathological assessment. All results should be "
        "reviewed and validated by a qualified ophthalmologist or clinical specialist.",
        styles['DisclaimerText']
    ))

    # ── REPORT FOOTER ──
    story.append(Spacer(1, 10 * mm))
    story.append(section_divider())
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("RETINASCAN AI", styles['SectionHeading']))
    story.append(Paragraph(
        "Explainable AI for Retinal OCT Analysis",
        styles['CaptionText']
    ))
    story.append(Paragraph(
        "Research / Decision Support Application",
        styles['CaptionText']
    ))

    # Build PDF
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return buffer.getvalue()
