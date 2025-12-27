#!/usr/bin/env python3
"""NEJM-style medical image analysis CLI tool."""

import sys
import json
from datetime import datetime
from pathlib import Path
import re

from pydantic import BaseModel, Field
from typing import Optional

sys.path.insert(0, '..')
from orouter.openrouter_client import OpenRouterClient, ModelConfig, ModelInput


class ImageClassification(BaseModel):
    primary_category: str = Field(..., description="Primary medical category/specialty (e.g., 'Radiology', 'Dermatology', 'Pathology', 'Ophthalmology')")
    sub_category: Optional[str] = Field(None, description="More specific classification (e.g., 'Chest X-ray', 'Skin Lesion', 'Histology Slide')")
    imaging_modality: Optional[str] = Field(None, description="Type of imaging (e.g., 'CT scan', 'MRI', 'X-ray', 'Ultrasound', 'Gross specimen', 'Microscopy')")
    body_region: Optional[str] = Field(None, description="Anatomical region shown (e.g., 'Thorax', 'Abdomen', 'Extremity', 'Head/Brain')")


class Abnormality(BaseModel):
    finding: str = Field(..., description="Specific abnormal finding observed in the image")
    location: Optional[str] = Field(None, description="Where in the image is this finding located? Use anatomical or positional descriptors")
    severity: Optional[str] = Field(None, description="Severity assessment if applicable (e.g., 'mild', 'moderate', 'severe')")
    clinical_significance: Optional[str] = Field(None, description="Why this finding matters clinically")


class NEJMImageAnalysis(BaseModel):
    image_caption: str = Field(..., description="Detailed professional medical image caption (3-5 sentences) describing what is shown in the image, including specific findings, anatomical details, and clinical context, suitable for a medical journal")
    clinical_summary: str = Field(..., description="Brief clinical summary (2-3 sentences) providing context about what condition or finding this image demonstrates")
    classification: ImageClassification = Field(..., description="Classification of the image type and category")
    abnormalities: list[Abnormality] = Field(default_factory=list, description="List of abnormal findings visible in the image. Include at least 1-5 abnormalities depending on complexity")
    normal_findings: list[str] = Field(default_factory=list, description="Notable normal or unremarkable findings that help rule out certain conditions")
    differential_diagnosis: list[str] = Field(default_factory=list, description="Possible diagnoses suggested by the imaging findings (2-5 main differentials)")
    clinical_pearls: list[str] = Field(default_factory=list, description="Key teaching points or clinical insights relevant to this image")


def resolve_image_sources(input_spec: str) -> list[str]:
    """
    Resolve input specification to list of image sources.

    Supports:
    - Single file: image.jpg
    - Range: image_001.jpg:image_010.jpg
    - Directory: /path/to/images/
    - URL: https://example.com/image.jpg
    """
    # Check if it's a range (contains colon and both parts are files)
    if ':' in input_spec and not input_spec.startswith('http'):
        parts = input_spec.split(':')
        if len(parts) == 2:
            start_file = parts[0]
            end_file = parts[1]

            # Extract directory and filenames
            start_path = Path(start_file)
            end_path = Path(end_file)

            if start_path.parent == end_path.parent and start_path.exists():
                # Extract numbering pattern
                start_name = start_path.stem
                end_name = end_path.stem

                # Try to extract numeric portions
                start_match = re.search(r'(\d+)', start_name)
                end_match = re.search(r'(\d+)', end_name)

                if start_match and end_match:
                    start_num = int(start_match.group(1))
                    end_num = int(end_match.group(1))
                    num_digits = len(start_match.group(1))
                    prefix = start_name[:start_match.start()]
                    suffix = start_name[start_match.end():]
                    extension = start_path.suffix
                    directory = start_path.parent

                    images = []
                    for num in range(start_num, end_num + 1):
                        formatted_num = str(num).zfill(num_digits)
                        filename = f"{prefix}{formatted_num}{suffix}{extension}"
                        filepath = directory / filename
                        if filepath.exists():
                            images.append(str(filepath))

                    if images:
                        return images

    # Check if it's a directory
    path = Path(input_spec)
    if path.is_dir():
        # Find all image files in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        images = sorted([str(f) for f in path.iterdir()
                        if f.suffix.lower() in image_extensions])
        if images:
            return images
        else:
            print(f"ERROR: No image files found in directory: {input_spec}", file=sys.stderr)
            sys.exit(1)

    # Otherwise treat as single file or URL
    if path.exists() or input_spec.startswith(('http://', 'https://')):
        return [input_spec]
    else:
        print(f"ERROR: Image not found: {input_spec}", file=sys.stderr)
        sys.exit(1)


def sanitize_filename(s: str) -> str:
    """Create a safe filename from a string."""
    return "".join(c if c.isalnum() else "_" for c in s)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NEJM-style medical image analysis CLI tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nejm_image_challenges_cli.py -i chest_xray.jpg mistral
  python nejm_image_challenges_cli.py -d /path/to/scans/ gpt-4-vision
  python nejm_image_challenges_cli.py -r scan_001.jpg:scan_010.jpg claude
        """
    )

    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--image",
        type=str,
        help="Single image file or URL"
    )
    input_group.add_argument(
        "-r", "--range",
        type=str,
        help="Range of images: start_file.jpg:end_file.jpg"
    )
    input_group.add_argument(
        "-d", "--directory",
        type=str,
        help="Directory containing images"
    )

    # Model (required positional)
    parser.add_argument(
        "model",
        type=str,
        help="Model to use (e.g., 'mistral', 'gpt-4-vision', 'claude')"
    )

    args = parser.parse_args()

    # Determine input source
    if args.image:
        image_input = args.image
    elif args.range:
        image_input = args.range
    elif args.directory:
        image_input = args.directory

    model = args.model

    # Resolve image sources
    image_sources = resolve_image_sources(image_input)

    print(f"Found {len(image_sources)} image(s) to analyze", file=sys.stderr)
    print(f"Using model: {model}", file=sys.stderr)

    prompt = """You are an expert medical diagnostician analyzing a medical image, similar to a NEJM Image Challenge case.

CRITICAL INSTRUCTIONS:
- Provide a professional, concise analysis suitable for medical education
- Be SPECIFIC about locations, sizes, and characteristics when describing findings
- Use precise medical terminology
- Include differential diagnosis based on the imaging findings
- Highlight clinically significant findings
- Provide teaching points that would be valuable for medical students/residents
- Be objective and evidence-based in your assessment

Analyze the medical image and provide:
1. A DETAILED professional image caption (3-5 sentences) describing exactly what is shown, including specific findings, anatomical details, measurements, and clinical context
2. Classification of the image type and anatomical region
3. Specific abnormal findings with precise locations and clinical significance
4. Notable normal findings that help narrow differential diagnosis
5. Differential diagnosis considerations
6. Key clinical teaching points"""

    # Setup output file
    output_dir = Path("image_challenges")
    output_dir.mkdir(exist_ok=True)

    # Create output filename: nejm_image_captions_{model_name}.json
    safe_model_name = sanitize_filename(model)
    json_file = output_dir / f"nejm_image_captions_{safe_model_name}.json"

    # Load existing file or create new structure
    if json_file.exists():
        with open(json_file, 'r') as f:
            output = json.load(f)
    else:
        output = {
            "model": model,
            "analyses": []
        }

    # Process each image
    for image_source in image_sources:
        print(f"  Analyzing: {image_source}", file=sys.stderr)

        # Extract ID from filename (e.g., "nejm_20201210.jpg" -> "20201210")
        filename = Path(image_source).stem
        # Extract all numeric sequences and use the longest one
        numeric_sequences = re.findall(r'\d+', filename)
        image_id = max(numeric_sequences, key=len) if numeric_sequences else filename

        try:
            client = OpenRouterClient()
            config = ModelConfig(model=model, temperature=0.3)
            response = client.generate_content(
                ModelInput(
                    user_prompt=prompt,
                    image_source=image_source,
                    response_model=NEJMImageAnalysis
                ),
                config=config
            )

            # Create analysis record with flattened structure
            analysis_record = {
                "id": image_id,
                **response.model_dump()
            }

        except Exception as e:
            print(f"  Error analyzing {image_source}: {e}", file=sys.stderr)
            analysis_record = {
                "id": image_id,
                "error": str(e)
            }

        # Append to output and write immediately
        output["analyses"].append(analysis_record)
        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"    ✓ Saved to {json_file}", file=sys.stderr)

    print(f"\nCompleted: {json_file}", file=sys.stderr)
    successful = len([a for a in output['analyses'] if 'error' not in a])
    print(f"Successful analyses: {successful}/{len(image_sources)}", file=sys.stderr)
