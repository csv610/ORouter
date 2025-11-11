import sys
import json
from datetime import datetime
from pathlib import Path
from openrouter_text_client import OpenRouterClient, ModelConfig
from pydantic import BaseModel, Field
from typing import Optional


class Epidemiology(BaseModel):
    prevalence: Optional[str] = Field(None, description="How common is this condition? Include specific percentages or rates (e.g., '1 in 1000 people', '5% of adults')")
    incidence: Optional[str] = Field(None, description="Rate of new cases per year. Include timeframe and population (e.g., '50,000 new cases annually in the US')")
    risk_factors: list[str] = Field(default_factory=list, description="Specific factors that increase risk. Be concrete: 'smoking >20 pack-years' not just 'smoking'. Include 3-8 most significant factors")
    demographics: Optional[str] = Field(None, description="Which populations are most affected? Include age, gender, ethnicity, geography with specifics")


class ClinicalPresentation(BaseModel):
    symptoms: list[str] = Field(default_factory=list, description="Patient-reported experiences. Be specific about timing, severity, frequency (e.g., 'sharp chest pain lasting 2-5 minutes' not 'chest pain'). Include 5-10 most common")
    signs: list[str] = Field(default_factory=list, description="Objective physical exam findings doctors observe (e.g., 'heart rate >100 bpm', 'bilateral crackles on lung auscultation'). Include 3-7 key findings")
    onset: Optional[str] = Field(None, description="How does it start? Sudden/gradual? Over hours/days/years? Include typical age of onset")
    severity_spectrum: Optional[str] = Field(None, description="Range from mild to severe with concrete examples of each level")


class DiagnosticApproach(BaseModel):
    clinical_criteria: list[str] = Field(default_factory=list, description="Standardized diagnostic criteria or scoring systems (e.g., 'DSM-5 criteria', 'Jones criteria'). Include threshold values when applicable")
    laboratory_tests: list[str] = Field(default_factory=list, description="Specific tests with normal vs abnormal ranges (e.g., 'HbA1c >6.5%', 'troponin elevation >99th percentile'). List 4-8 most useful tests")
    imaging_studies: list[str] = Field(default_factory=list, description="What imaging shows and why it's ordered (e.g., 'chest X-ray shows bilateral infiltrates', 'MRI to detect lesions >3mm')")
    differential_diagnosis: list[str] = Field(default_factory=list, description="Other conditions that present similarly and must be ruled out. Include 3-6 most important alternatives with key distinguishing features")
    gold_standard: Optional[str] = Field(None, description="The definitive test that confirms diagnosis. Explain what makes it definitive (e.g., 'tissue biopsy showing granulomas')")


class TreatmentPlan(BaseModel):
    pharmacological: list[str] = Field(default_factory=list, description="Specific medications with typical doses, routes, frequency (e.g., 'metformin 500mg PO BID', 'lisinopril 10-40mg daily'). Include drug classes and mechanisms. List 4-8 options")
    non_pharmacological: list[str] = Field(default_factory=list, description="Therapies beyond drugs: physical therapy protocols, psychotherapy types, devices, procedures. Be specific about techniques and frequency")
    surgical: list[str] = Field(default_factory=list, description="Surgical interventions with indications for when they're needed (e.g., 'coronary bypass for 3-vessel disease'). Include success rates if significant")
    lifestyle_modifications: list[str] = Field(default_factory=list, description="Concrete, actionable changes with targets (e.g., '150 min/week moderate exercise', 'sodium <2g/day', 'quit smoking')")
    first_line: Optional[str] = Field(None, description="What do doctors try first and why? Include evidence level (e.g., 'Class I recommendation')")
    duration: Optional[str] = Field(None, description="How long does treatment last? Distinguish acute vs maintenance phases if applicable")


class PrognosisOutcome(BaseModel):
    overall_prognosis: Optional[str] = Field(None, description="What's the typical outcome? Include survival rates, cure rates, or quality of life impact with timeframes (e.g., '5-year survival 85%', 'most recover fully in 6-12 months')")
    factors_affecting_outcome: list[str] = Field(default_factory=list, description="What makes prognosis better or worse? Be specific (e.g., 'early diagnosis improves 10-year survival by 40%', 'age >65 doubles mortality risk')")
    survival_rates: Optional[str] = Field(None, description="Specific survival statistics by stage/severity with timeframes (1-year, 5-year, 10-year)")
    chronic_considerations: Optional[str] = Field(None, description="For chronic conditions: monitoring frequency, long-term complications to watch for, quality of life impacts")


class MedicalTopic(BaseModel):
    condition_name: str = Field(..., description="Official medical name, not slang or abbreviations")
    summary: str = Field(..., description="2-3 sentence overview covering WHAT it is, WHO it affects, and WHY it matters. Write for educated non-medical audience")
    category: str = Field(..., description="Primary medical specialty/system (e.g., 'Cardiovascular', 'Infectious Disease', 'Endocrine', 'Neurological')")
    icd_codes: list[str] = Field(default_factory=list, description="ICD-10 or ICD-11 codes in format like 'E11.9' with brief label")
    pathophysiology: Optional[str] = Field(None, description="Explain the biological mechanism in 3-5 sentences. HOW does the disease process work at cellular/organ level? What breaks down? Use analogies when helpful")
    etiology: list[str] = Field(default_factory=list, description="ROOT CAUSES not just risk factors. Include genetic, environmental, infectious agents, autoimmune triggers. Be specific (e.g., 'HBV infection' not 'viral infection')")
    types: list[str] = Field(default_factory=list, description="Subtypes or classifications with key differentiating features (e.g., 'Type 1: autoimmune, childhood onset' vs 'Type 2: insulin resistance, adult onset')")
    stages: list[str] = Field(default_factory=list, description="Disease progression stages with clinical features of each (e.g., 'Stage 1: GFR >90, no symptoms')")
    epidemiology: Optional[Epidemiology] = Field(None, description="Population-level statistics and risk factors")
    clinical_presentation: Optional[ClinicalPresentation] = Field(None, description="How patients present and what doctors observe")
    diagnosis: DiagnosticApproach = Field(default_factory=DiagnosticApproach, description="How to confirm the diagnosis")
    treatment: TreatmentPlan = Field(default_factory=TreatmentPlan, description="How to manage the condition")
    complications: list[str] = Field(default_factory=list, description="What can go wrong if untreated or poorly managed? Include acute and chronic complications with approximate frequency/timing")
    prognosis: Optional[PrognosisOutcome] = Field(None, description="Expected outcomes and what influences them")
    prevention: list[str] = Field(default_factory=list, description="Evidence-based prevention strategies with efficacy data when available (e.g., 'vaccination reduces risk by 90%')")
    related_conditions: list[str] = Field(default_factory=list, description="Conditions that commonly co-occur or share pathophysiology. Explain the relationship briefly")
    key_considerations: list[str] = Field(default_factory=list, description="CLINICAL PEARLS: non-obvious insights, common pitfalls, important nuances that medical students/practitioners should know")
    references: list[str] = Field(default_factory=list, description="Major clinical guidelines or landmark studies (e.g., 'AHA/ACC 2023 Guidelines', 'Framingham Heart Study')")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <medical_topic> [model1 model2 ...]")
        print("Example: python script.py diabetes mistral gpt-4 claude")
        sys.exit(1)
    
    # Parse arguments
    topic = sys.argv[1]
    models = sys.argv[2:] if len(sys.argv) > 2 else ["mistral"]
    
    prompt = f"""You are a medical expert creating a comprehensive clinical reference on: {topic}

CRITICAL INSTRUCTIONS:
- Be SPECIFIC with numbers: percentages, rates, doses, thresholds, timeframes
- Include QUANTITATIVE data wherever possible
- Provide CONTEXT for statistics (e.g., "8% of US population" not just "25 million")
- Use CONCRETE examples with actual values
- Distinguish between different severity levels with numeric criteria
- For medications: ALWAYS include dose, route, frequency
- For tests: ALWAYS include cutoff values and units
- For risk factors: quantify the risk increase when known
- For complications: include approximate frequency/incidence
- Avoid vague terms like "some", "often", "usually" - use percentages instead

Provide comprehensive, evidence-based medical information covering:
- Pathophysiology with specific mechanisms
- Epidemiology with exact prevalence, incidence, and demographics
- Risk factors with quantified risk levels
- Clinical presentation with specific symptom patterns and timing
- Complete diagnostic criteria with thresholds
- Evidence-based treatments with exact regimens
- Complications with frequency data
- Prognosis with survival/outcome statistics"""
    
    print(f"Generating information on: {topic}", file=sys.stderr)
    print(f"Using models: {', '.join(models)}", file=sys.stderr)
    
    # Setup output file
    output_dir = Path("medical_topics")
    output_dir.mkdir(exist_ok=True)
    safe_topic = "".join(c if c.isalnum() else "_" for c in topic)
    json_file = output_dir / f"{safe_topic}.json"
    
    # Load existing file or create new structure
    if json_file.exists():
        with open(json_file, 'r') as f:
            output = json.load(f)
    else:
        output = {
            "topic": topic,
            "answers": []
        }
    
    # Query each model and append immediately
    for model in models:
        print(f"  Querying {model}...", file=sys.stderr)
        
        try:
            client = OpenRouterClient(config=ModelConfig(model=model, temperature=0.3))
            response = client.generate_structured(
                user_prompt=prompt,
                response_model=MedicalTopic
            )
            
            # Append new response
            output["answers"].append({
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "data": response.model_dump()
            })
            
        except Exception as e:
            print(f"  Error with {model}: {e}", file=sys.stderr)
            output["answers"].append({
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
        
        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2)
    
    print(f"\nSaved: {json_file}", file=sys.stderr)
    print(f"Total responses: {len([r for r in output['responses'] if 'data' in r])}", file=sys.stderr)
