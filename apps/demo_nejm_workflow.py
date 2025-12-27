#!/usr/bin/env python3
"""Demonstration of NEJM Image Challenges CLI workflow."""

import json
from datetime import datetime
from pathlib import Path
from nejm_image_challenges_cli import NEJMImageAnalysis, ImageClassification, Abnormality

def demo_workflow():
    """Demonstrate the complete workflow with sample data."""

    print("=" * 80)
    print("NEJM IMAGE CHALLENGES CLI - WORKFLOW DEMONSTRATION")
    print("=" * 80)
    print()

    # Show command examples
    print("COMMAND EXAMPLES:")
    print("-" * 80)
    examples = [
        ("Single image", "python nejm_image_challenges_cli.py -i chest_xray.jpg mistral"),
        ("Directory scan", "python nejm_image_challenges_cli.py -d /path/to/scans/ claude"),
        ("Image range", "python nejm_image_challenges_cli.py -r scan_001.jpg:scan_010.jpg gpt-4"),
    ]

    for desc, cmd in examples:
        print(f"{desc:20} → {cmd}")

    print()
    print("OUTPUT BEHAVIOR:")
    print("-" * 80)
    print("• Filename: nejm_image_captions_{model_name}.json")
    print("• Location: image_challenges/ directory")
    print("• Writing: Incremental (one image result = immediate write)")
    print("• Appending: Results append to existing file if run again")
    print()

    # Create sample output structure
    print("SAMPLE OUTPUT STRUCTURE:")
    print("-" * 80)

    sample_output = {
        "model": "mistral",
        "analyses": [
            {
                "id": 1,
                "image_caption": "Frontal chest radiograph showing bilateral alveolar infiltrates...",
                "clinical_summary": "This chest X-ray demonstrates findings consistent with bacterial pneumonia...",
                "classification": {
                    "primary_category": "Radiology",
                    "sub_category": "Chest Radiography",
                    "imaging_modality": "X-ray",
                    "body_region": "Thorax"
                },
                "abnormalities": [
                    {
                        "finding": "Bilateral alveolar infiltrates",
                        "location": "Lower lobes",
                        "severity": "Moderate",
                        "clinical_significance": "Indicates consolidation"
                    }
                ],
                "normal_findings": ["No pneumothorax", "Normal cardiac silhouette"],
                "differential_diagnosis": ["CAP", "Atypical pneumonia"],
                "clinical_pearls": ["Bilateral involvement is common in upright patients"]
            }
        ]
    }

    print(json.dumps(sample_output, indent=2)[:800] + "\n    ... (truncated)")

    print()
    print("INCREMENTAL WRITE EXAMPLE:")
    print("-" * 80)
    print("""
When running: python nejm_image_challenges_cli.py -d /path/to/scans/ mistral

Progress:
  Found 5 image(s) to analyze
  Using model: mistral
    Analyzing: /path/to/scans/scan_001.jpg
    ✓ Saved to image_challenges/nejm_image_captions_mistral.json
    Analyzing: /path/to/scans/scan_002.jpg
    ✓ Saved to image_challenges/nejm_image_captions_mistral.json
    Analyzing: /path/to/scans/scan_003.jpg
    ✓ Saved to image_challenges/nejm_image_captions_mistral.json
    Analyzing: /path/to/scans/scan_004.jpg
    ✓ Saved to image_challenges/nejm_image_captions_mistral.json
    Analyzing: /path/to/scans/scan_005.jpg
    ✓ Saved to image_challenges/nejm_image_captions_mistral.json

Completed: image_challenges/nejm_image_captions_mistral.json
Successful analyses: 5/5
    """)

    print()
    print("KEY FEATURES:")
    print("-" * 80)
    features = [
        "Single model per run (focused analysis)",
        "Multiple input modes (file, directory, range, URL)",
        "Incremental writing (safe for long-running batches)",
        "Consistent output filename pattern",
        "Persistent results (no data loss on interruption)",
        "Professional NEJM-style analysis output",
        "Automatic ID extraction from filenames (nejm_20201210.jpg → id: 20201210)",
    ]

    for i, feature in enumerate(features, 1):
        print(f"{i}. {feature}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    demo_workflow()
