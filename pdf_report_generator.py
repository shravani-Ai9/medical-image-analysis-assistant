
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as ReportImage
)
from reportlab.lib.utils import ImageReader
from io import BytesIO
from datetime import datetime


def create_pdf_report(predicted_class, probabilities, class_names,
                      xray_image=None, gradcam_image=None):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        fontSize=19,
        spaceAfter=8
    ))

    story = []

    story.append(Paragraph(
        "AI Chest X-ray Analysis Report", styles["ReportTitle"]
    ))
    story.append(Paragraph(
        "Educational AI prototype — not a medical diagnosis",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Model Prediction", styles["Heading2"]))
    story.append(Paragraph(
        f"Predicted dataset class: <b>{predicted_class}</b>",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Model Output Scores", styles["Heading2"]))

    score_rows = [["Class", "Model output score"]]
    for name, score in sorted(
        zip(class_names, probabilities),
        key=lambda item: item[1],
        reverse=True
    ):
        score_rows.append([name, f"{float(score) * 100:.2f}%"])

    score_table = Table(score_rows, colWidths=[7 * cm, 7 * cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F2F6FA")),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 12))

    def add_image_section(title, image_data):
        if image_data is not None:
            story.append(Paragraph(title, styles["Heading2"]))
            try:
                if isinstance(image_data, bytes):
                    image_data = BytesIO(image_data)

                img = ReportImage(
                    image_data, width=7 * cm, height=7 * cm,
                    kind="proportional"
                )
                story.append(img)
                story.append(Spacer(1, 8))
            except Exception:
                story.append(Paragraph(
                    "Image could not be added to this PDF.",
                    styles["BodyText"]
                ))

    add_image_section("3. Uploaded X-ray Sample", xray_image)
    add_image_section("4. Grad-CAM Visualization", gradcam_image)

    story.append(Paragraph("5. Explainability", styles["Heading2"]))
    story.append(Paragraph(
        "Grad-CAM highlights image regions that influenced the model output. "
        "These regions are not confirmed medical findings and do not establish "
        "the presence or absence of disease.",
        styles["BodyText"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
