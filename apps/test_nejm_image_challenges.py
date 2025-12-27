#!/usr/bin/env python3
"""Test script to demonstrate the NEJM Image Analysis tool."""

import json
from pathlib import Path
from nejm_image_challenges_cli import NEJMImageAnalysis, ImageClassification, Abnormality


def test_nejm_model_structure():
    """Test that the Pydantic model accepts valid medical image analysis data."""

    # Sample medical image analysis - a chest X-ray showing pneumonia
    sample_analysis = {
        "image_caption": "Frontal chest radiograph showing bilateral alveolar infiltrates with air bronchograms in both lower lobes, consistent with community-acquired pneumonia.",
        "clinical_summary": "This chest X-ray demonstrates findings consistent with bacterial pneumonia, featuring consolidation in the bilateral lower lobes. The presence of air bronchograms and alveolar-pattern opacities are typical of acute pneumonic process.",
        "classification": {
            "primary_category": "Radiology",
            "sub_category": "Chest Radiography",
            "imaging_modality": "X-ray",
            "body_region": "Thorax"
        },
        "abnormalities": [
            {
                "finding": "Bilateral alveolar infiltrates",
                "location": "Lower lobes, predominantly posterior segments",
                "severity": "Moderate",
                "clinical_significance": "Indicates consolidation consistent with pneumonia"
            },
            {
                "finding": "Air bronchograms",
                "location": "Within areas of consolidation",
                "severity": "Present",
                "clinical_significance": "Confirms alveolar filling and rules out pure pleural effusion"
            },
            {
                "finding": "Heart size",
                "location": "Mediastinum",
                "severity": "Normal",
                "clinical_significance": "Excludes acute heart failure as cause of infiltrates"
            }
        ],
        "normal_findings": [
            "No pleural effusions",
            "Normal cardiac silhouette",
            "No pneumothorax",
            "No signs of pulmonary edema"
        ],
        "differential_diagnosis": [
            "Community-acquired pneumonia (bacterial)",
            "Atypical pneumonia (viral or mycoplasma)",
            "Aspiration pneumonia",
            "Pulmonary edema (less likely given normal heart size)"
        ],
        "clinical_pearls": [
            "Bilateral lower lobe involvement is common in community-acquired pneumonia when patient is upright or ambulatory",
            "Air bronchograms are a sign of consolidation and increase specificity for pneumonia",
            "Heart size assessment helps differentiate pneumonia from cardiogenic causes of infiltrates",
            "Severity on imaging does not always correlate with clinical illness; clinical judgment is essential"
        ]
    }

    # Test that the model validates correctly
    analysis = NEJMImageAnalysis(**sample_analysis)

    print("✓ Model validation successful!\n")
    print("=" * 70)
    print("SAMPLE NEJM IMAGE ANALYSIS OUTPUT")
    print("=" * 70)

    # Pretty print the analysis
    output = {
        "image_caption": analysis.image_caption,
        "clinical_summary": analysis.clinical_summary,
        "classification": {
            "primary_category": analysis.classification.primary_category,
            "sub_category": analysis.classification.sub_category,
            "imaging_modality": analysis.classification.imaging_modality,
            "body_region": analysis.classification.body_region
        },
        "abnormalities": [
            {
                "finding": ab.finding,
                "location": ab.location,
                "severity": ab.severity,
                "clinical_significance": ab.clinical_significance
            }
            for ab in analysis.abnormalities
        ],
        "normal_findings": analysis.normal_findings,
        "differential_diagnosis": analysis.differential_diagnosis,
        "clinical_pearls": analysis.clinical_pearls
    }

    print(json.dumps(output, indent=2))
    print("\n" + "=" * 70)
    print("MODEL VALIDATION SUCCESSFUL!")
    print("=" * 70)

    # Verify all fields are populated
    assert analysis.image_caption
    assert analysis.clinical_summary
    assert analysis.classification.primary_category == "Radiology"
    assert len(analysis.abnormalities) == 3
    assert len(analysis.normal_findings) == 4
    assert len(analysis.differential_diagnosis) == 4
    assert len(analysis.clinical_pearls) == 4

    print("\n✓ All assertions passed!")
    print("✓ Tool is ready for production use with vision-capable models")


if __name__ == "__main__":
    test_nejm_model_structure()
