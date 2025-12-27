# NEJM Image Challenges CLI

Professional medical image analysis tool producing NEJM-style clinical captions and structured analysis.

## Features

### Input Modes (Short Options)

1. **Single Image** `-i`
   ```bash
   python nejm_image_challenges_cli.py -i chest_xray.jpg mistral
   python nejm_image_challenges_cli.py -i https://example.com/scan.jpg mistral
   ```

2. **Image Range** `-r` (numbered sequence)
   ```bash
   python nejm_image_challenges_cli.py -r scan_001.jpg:scan_010.jpg claude
   ```
   - Automatically detects numeric pattern
   - Fills in missing files in sequence
   - Preserves zero-padding

3. **Directory Scanning** `-d`
   ```bash
   python nejm_image_challenges_cli.py -d /path/to/scans/ gpt-4-vision
   ```
   - Finds all image files (jpg, jpeg, png, gif, bmp, tiff, webp)
   - Processes in sorted order

### Output

- **Filename Pattern**: `nejm_image_captions_{model_name}.json`
- **Location**: `image_challenges/` directory
- **Writing Mode**: Incremental (immediate write after each image)
- **Append Behavior**: Results append to existing file if script runs again
- **ID Field**: Automatically extracted from image filename (longest numeric sequence)

### Analysis Output

Each image analysis record includes:

```python
{
    "id": str                       # Extracted from filename (e.g., "20201210")
    "image_caption": str            # Professional journal-style caption
    "clinical_summary": str         # Context about the condition
    "classification": {
        "primary_category": str     # Specialty (Radiology, etc.)
        "sub_category": str         # Specific type (Chest X-ray, etc.)
        "imaging_modality": str     # Modality (X-ray, CT, MRI, etc.)
        "body_region": str          # Anatomical region
    }
    "abnormalities": [
        {
            "finding": str          # Specific abnormal finding
            "location": str         # Where in image
            "severity": str         # mild/moderate/severe
            "clinical_significance": str # Why it matters
        }
    ]
    "normal_findings": list[str]   # Negative findings
    "differential_diagnosis": list[str]  # Likely diagnoses
    "clinical_pearls": list[str]   # Teaching points
}
```

## Workflow Example

```
$ python nejm_image_challenges_cli.py -d /path/to/scans/ mistral

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
```

## JSON Output Structure

```json
{
  "model": "mistral",
  "analyses": [
    {
      "id": "20201210",
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
          "clinical_significance": "Indicates consolidation consistent with pneumonia"
        }
      ],
      "normal_findings": ["No pleural effusions", "Normal cardiac silhouette"],
      "differential_diagnosis": ["CAP", "Atypical pneumonia"],
      "clinical_pearls": ["Bilateral lower lobe involvement is common"]
    }
  ]
}
```

### ID Extraction from Filename

The `id` field is automatically extracted from the image filename:

| Filename | Extracted ID |
|----------|--------------|
| `nejm_20201210.jpg` | `20201210` |
| `scan_001.jpg` | `001` |
| `image_123_v2.jpg` | `123` |
| `test.jpg` | `test` |

**Logic:** Extracts the longest numeric sequence from the filename. If no numbers found, uses the filename itself. This makes the output self-documenting and traceable to the source image.

## Model Naming

Model names are sanitized for safe filenames:

| Model | Output File |
|-------|------------|
| `mistral` | `nejm_image_captions_mistral.json` |
| `google/gemma-3-27b-it:free` | `nejm_image_captions_google_gemma_3_27b_it_free.json` |
| `openai/gpt-4-vision` | `nejm_image_captions_openai_gpt_4_vision.json` |
| `claude-opus-4-1` | `nejm_image_captions_claude_opus_4_1.json` |

## Key Design Decisions

1. **Single Model Per Run**: Focused, consistent analysis
2. **Incremental Writing**: Safe for long-running batch jobs
3. **Multiple Input Formats**: Maximum flexibility
4. **Professional Output**: NEJM-journal quality captions
5. **Persistent Results**: No data loss on interruption

## Test Files

- `test_image_input_resolution.py` - Tests all input resolution modes
- `test_nejm_image_challenges.py` - Tests Pydantic model validation
- `demo_nejm_workflow.py` - Demonstrates complete workflow

## Usage Tips

- **Large Batch Processing**: Use directory mode for bulk analysis
- **Specific Sequences**: Use range mode with zero-padded filenames
- **Resumable Processing**: Results append automatically
- **Model Comparison**: Run same images with different models (creates separate files)
