from pathlib import Path
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Table


def generate_pdf_report(
    save_path,
    original_image_path,
    gradcam_image_path,
    body_part,
    body_conf,
    fracture_result,
    fracture_conf,
):

    # Create PDF document
    doc = SimpleDocTemplate(
        save_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    elements = []

    # Title
    title = Paragraph(
        "<b>Bone Fracture Detection Report</b>",
        styles["Title"]
    )

    elements.append(title)
    elements.append(Spacer(1, 0.3 * inch))

    # Timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    timestamp = Paragraph(
        f"<b>Generated:</b> {current_time}",
        styles["BodyText"]
    )

    elements.append(timestamp)
    elements.append(Spacer(1, 0.2 * inch))

    # Prediction Details
    details = Paragraph(
        f"""
        <b>Predicted Body Part:</b> {body_part} ({body_conf:.2f}%)<br/>
        <b>Fracture Status:</b> {fracture_result.capitalize()} ({fracture_conf:.2f}%)
        """,
        styles["BodyText"]
    )

    elements.append(details)
    elements.append(Spacer(1, 0.12 * inch))

    # Images Section
    elements.append(
        Paragraph("<b>X-ray Analysis</b>", styles["Heading4"])
    )

    original_img = Image(original_image_path)
    original_img.drawHeight = 2.4 * inch
    original_img.drawWidth = 2.4 * inch

    gradcam_img = Image(gradcam_image_path)
    gradcam_img.drawHeight = 2.4 * inch
    gradcam_img.drawWidth = 2.4 * inch

    image_table = Table([
        [original_img, gradcam_img]
    ])

    elements.append(image_table)
    elements.append(Spacer(1, 0.15 * inch))

    # Why Predicted Section
    why_predicted = Paragraph(
        f"""
        <b>Why the Model Predicted This Result:</b><br/><br/>
        
        The Grad-CAM visualization highlights the important regions
        of the X-ray image that influenced the CNN model's decision.
        
        Warmer regions (red/yellow) indicate areas where the model
        focused more strongly while detecting abnormalities.
        
        In this case, the highlighted regions suggest features
        associated with <b>{fracture_result}</b> findings in the
        {body_part.lower()} X-ray image.
        """,
        styles["BodyText"]
    )

    elements.append(why_predicted)
    elements.append(Spacer(1, 0.15 * inch))

    # Disclaimer
    disclaimer = Paragraph(
        """
        <b>Disclaimer:</b><br/>
        This system is intended for educational and research purposes only.
        It should not replace professional medical diagnosis.
        """,
        styles["BodyText"]
    )

    elements.append(disclaimer)

    # Build PDF
    doc.build(elements)

    return save_path