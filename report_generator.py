def generate_report(predicted_class, probabilities):
    """
    Create an educational summary of the model output.
    This is not a medical diagnosis.
    """
    scores = {
        name: float(score) * 100
        for name, score in zip(
            ["BACTERIAL", "NORMAL", "VIRAL"],
            probabilities
        )
    }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    report = f"""
# AI Chest X-ray Analysis Report

## 1. Model Prediction
Predicted class: **{predicted_class}**

## 2. Model Output Scores
"""
    for name, score in ranked:
        report += f"- {name}: {score:.2f}%\n"

    report += f"""
## 3. Explainability
A Grad-CAM heatmap is provided to visualize image regions that influenced
the model's prediction. These highlighted regions do not establish a
medical finding or explain a disease by themselves.

## 4. Limitations
This educational AI prototype classifies images into three dataset
categories. Its prediction and scores are not a confirmed medical result.
It has not been established as safe for clinical decision-making.

## 5. Recommended Use
Use this report for educational demonstration and model evaluation only.
Consult a qualified healthcare professional for medical interpretation.
"""
    return report
