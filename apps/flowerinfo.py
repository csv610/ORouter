import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from typing import Optional

sys.path.insert(0, '..')
from orouter.openrouter_client import OpenRouterClient, ModelConfig, ModelInput


class Taxonomy(BaseModel):
    family: Optional[str] = Field(None, description="Plant family (e.g., 'Rosaceae', 'Orchidaceae')")
    genus: Optional[str] = Field(None, description="Genus name (e.g., 'Rosa', 'Dendrobium')")
    species: Optional[str] = Field(None, description="Species name with author and year (e.g., 'Rosa damascena Mill., 1768')")
    common_names: list[str] = Field(default_factory=list, description="Common names in different regions/languages")
    native_region: Optional[str] = Field(None, description="Geographic origin and native habitat with specific continents/regions and elevation ranges")
    hardiness_zones: Optional[str] = Field(None, description="USDA hardiness zones suitable for growth (e.g., 'Zones 5-9')")
    kingdom: Optional[str] = Field(None, description="Plantae (always this for plants)")
    order: Optional[str] = Field(None, description="Plant order (e.g., 'Rosales', 'Asparagales')")
    chromosome_number: Optional[str] = Field(None, description="Chromosome count with ploidy level (e.g., '2n=32, diploid' or '2n=84, hexaploid')")
    conservation_status: Optional[str] = Field(None, description="IUCN status if applicable (e.g., 'Least Concern', 'Critically Endangered')")


class MorphologyStructure(BaseModel):
    plant_type: Optional[str] = Field(None, description="Growth type (e.g., 'herbaceous perennial', 'woody shrub', 'climbing vine')")
    lifespan: Optional[str] = Field(None, description="Annual, biennial, or perennial with specific duration")
    height: Optional[str] = Field(None, description="Typical mature height range (e.g., '2-4 feet', '30-60 cm')")
    spread: Optional[str] = Field(None, description="Typical width/spread at maturity")
    stem_characteristics: list[str] = Field(default_factory=list, description="Details about stems: color, texture, thorniness, vascular structure (e.g., 'green with sharp red thorns', 'terete hollow stems', 'secondary growth with cork layer')")
    stem_anatomy: Optional[str] = Field(None, description="Expert detail: vascular bundle arrangement, presence of pith, cortex structure")
    root_system: Optional[str] = Field(None, description="Root type and structure (e.g., 'fibrous root system', 'taproot with lateral roots', 'adventitious roots')")
    leaf_structure: list[str] = Field(default_factory=list, description="Leaf characteristics: shape, size, color, arrangement, venation pattern (e.g., 'compound pinnate leaves, 8-12 inches long', 'reticulate venation', 'opposite arrangement')")
    leaf_anatomy: Optional[str] = Field(None, description="Expert detail: cuticle thickness, stomatal type and distribution, mesophyll structure, presence of trichomes")
    flower_structure: list[str] = Field(default_factory=list, description="Flower anatomy: petal count, arrangement, stamen/pistil details, inflorescence type (e.g., '5 petals in whorled arrangement', 'hypogynous stamens', 'simple raceme inflorescence')")
    floral_formula: Optional[str] = Field(None, description="Expert detail: floral formula in standard notation (e.g., 'K5 C5 A5 G(5)')")
    floral_diagram: Optional[str] = Field(None, description="Expert detail: description of floral diagram with symmetry (actinomorphic, zygomorphic, asymmetric)")


class BloomCharacteristics(BaseModel):
    bloom_season: Optional[str] = Field(None, description="When does it bloom? Include months and duration (e.g., 'June-August', 'spring to fall (3-4 months)')")
    photoperiodism: Optional[str] = Field(None, description="Day-length sensitivity (e.g., 'short-day plant requiring <12 hours light', 'day-neutral', 'long-day plant')")
    flowering_time_from_germination: Optional[str] = Field(None, description="Time to first bloom from seed or propagule (e.g., '90-120 days', '3-4 weeks')")
    flower_color: list[str] = Field(default_factory=list, description="Colors of blooms, including pigment types (e.g., 'deep crimson red (anthocyanins)', 'pale pink fading to white (carotenoids)')")
    flower_size: Optional[str] = Field(None, description="Typical flower diameter or dimensions (e.g., '3-4 inches across', '2-3 cm blooms')")
    fragrance: Optional[str] = Field(None, description="Fragrance description, intensity, and chemical compounds (e.g., 'strong sweet honey scent (volatile organic compounds)', 'no fragrance')")
    bloom_abundance: Optional[str] = Field(None, description="How many flowers and frequency (e.g., '200-500 blooms per plant', 'continuous flowering', 'sparse single blooms')")
    flower_longevity: Optional[str] = Field(None, description="How long flowers last (e.g., 'individual flowers last 1-2 days', 'blooming season 3 months', 'each flower lasts 2 weeks')")
    pollination_type: Optional[str] = Field(None, description="Pollination mechanism (e.g., 'entomophilous (insect-pollinated)', 'anemophilous (wind-pollinated)', 'ornithophilous (bird-pollinated)')")
    anther_maturity: Optional[str] = Field(None, description="Timing of anther maturation relative to stigma (protandrous, protogynous, or herkogamous)")
    reproductive_cycle: Optional[str] = Field(None, description="Sexual or asexual reproduction modes (e.g., 'hermaphroditic flowers with perfect stamens and pistils', 'dioecious species')")


class GrowthRequirements(BaseModel):
    light_requirements: Optional[str] = Field(None, description="Light spectrum and intensity needs (e.g., 'full sun 6+ hours daily', 'partial shade 3-4 hours sun', 'shade-tolerant with high red/far-red ratio')")
    photosynthetic_pathway: Optional[str] = Field(None, description="C3, C4, or CAM photosynthesis type")
    soil_type: list[str] = Field(default_factory=list, description="Soil preferences with specific characteristics (e.g., 'well-draining loamy soil', 'acidic pH 5.5-6.5', 'sandy soil', 'clay loam with high organic matter')")
    soil_texture: Optional[str] = Field(None, description="Textural class and structure (e.g., 'loam with stable aggregates', 'sandy loam')")
    soil_nutrients: Optional[str] = Field(None, description="Soil fertility requirements (e.g., 'nitrogen-demanding', 'phosphorus-deficient soils tolerated', 'potassium-dependent')")
    mycorrhizal_associations: Optional[str] = Field(None, description="Fungal symbiosis if applicable (e.g., 'arbuscular mycorrhizal fungi', 'ectomycorrhizal associates', 'none')")
    water_requirements: Optional[str] = Field(None, description="Watering needs with frequency and moisture retention (e.g., 'moderate watering, keep soil at field capacity', 'drought tolerant once established', 'weekly deep watering in growing season')")
    water_stress_tolerance: Optional[str] = Field(None, description="Drought and flood tolerance mechanisms (e.g., 'xerophytic with reduced transpiration', 'mesophytic', 'hydrophytic')")
    humidity: Optional[str] = Field(None, description="Humidity preferences and transpiration rates (e.g., 'prefers dry air <30% RH', 'high humidity 60-80%', 'moderate humidity 40-60%')")
    temperature_range: Optional[str] = Field(None, description="Optimal and critical temperature ranges (e.g., '60-75°F (15-24°C) optimal, cold injury below 50°F')")
    chilling_requirements: Optional[str] = Field(None, description="Cold period needed for vernalization (e.g., '200-300 chilling hours below 45°F for flowering')")
    fertilizer_needs: Optional[str] = Field(None, description="Macronutrient and micronutrient requirements (e.g., 'balanced NPK 10-10-10', 'high phosphorus for fruiting', 'iron-loving plant')")


class CultivationCare(BaseModel):
    propagation_methods: list[str] = Field(default_factory=list, description="How to propagate with specific techniques and success rates (e.g., 'seeds sown in spring at 70°F with 80% germination', 'softwood cuttings in summer with 6-week rooting')")
    seed_requirements: Optional[str] = Field(None, description="Seed viability, dormancy breaking, stratification needs (e.g., 'seeds require 3 months cold stratification', 'short-lived seeds with 6-month viability')")
    breeding_notes: Optional[str] = Field(None, description="Hybrid vigor, cultivar development, breeding considerations (e.g., 'extensively hybridized, heterozygous cultivars', 'seed-grown plants show high variability')")
    pruning_deadheading: Optional[str] = Field(None, description="Pruning and deadheading guidance with timing (e.g., 'deadhead spent flowers weekly to encourage indeterminate blooming', 'hard prune in early spring before growth flush')")
    pest_diseases: list[str] = Field(default_factory=list, description="Common pests and diseases with causal agents and symptoms (e.g., 'Tetranychus urticae (spider mites) causing chlorotic stippling', 'Erysiphe (powdery mildew) on leaf surfaces', 'Aphis (aphids) causing leaf curl')")
    disease_prevention: list[str] = Field(default_factory=list, description="Prevention and treatment strategies with biological and cultural controls (e.g., 'improve air circulation to prevent fungal mildew', 'beneficial insects like ladybugs for pest control', 'avoid overhead watering to reduce leaf wetness')")
    resistance_traits: Optional[str] = Field(None, description="Disease resistance cultivars or natural resistance mechanisms (e.g., 'resistant to powdery mildew (Pm gene)', 'tolerant to root rot pathogens')")
    seasonal_care: Optional[str] = Field(None, description="Season-specific care tasks including dormancy periods (e.g., 'winter mulching in cold zones to protect crowns', 'reduce watering during senescence', 'spring growth flush begins at 50°F soil temperature')")


class UsesCompanions(BaseModel):
    landscape_uses: list[str] = Field(default_factory=list, description="Garden and landscape applications with ecological function (e.g., 'cutting garden specimen', 'habitat restoration', 'ground cover for erosion control', 'mixed perennial borders')")
    companion_plants: list[str] = Field(default_factory=list, description="Plants with compatible growing requirements and beneficial interactions (e.g., 'nitrogen-fixing legumes nearby improve soil', 'combines with ornamental grasses', 'underplant with ground covers')")
    cut_flower_potential: Optional[str] = Field(None, description="Suitability for cutting with vase life and post-harvest physiology (e.g., 'excellent cut flower, 7-10 days with ethylene sensitivity', 'poor as cut flower due to rapid senescence')")
    pollinator_attraction: Optional[str] = Field(None, description="Specific pollinators with behavioral interactions (e.g., 'highly attractive to honeybees (pollen feeders) and Lepidoptera', 'attracts hummingbirds with tubular red flowers', 'wind-pollinated, minimal insect interaction')")
    wildlife_value: Optional[str] = Field(None, description="Food source and habitat value for fauna (e.g., 'provides nectar for 15+ butterfly species', 'seed source for granivorous birds', 'shelter for ground-nesting bees')")


class BotanicalEcology(BaseModel):
    native_habitat_type: Optional[str] = Field(None, description="Ecosystem classification (e.g., 'temperate deciduous forest understory', 'Mediterranean scrubland', 'alpine meadow')")
    ecological_role: Optional[str] = Field(None, description="Ecological function and community role (e.g., 'pioneer species in disturbed areas', 'understory component', 'forage plant for herbivores')")
    allelopathy: Optional[str] = Field(None, description="Allelopathic compounds if applicable (e.g., 'produces root exudates inhibiting competitor growth', 'no allelopathic effects')")
    invasive_potential: Optional[str] = Field(None, description="Risk assessment for invasiveness (e.g., 'non-invasive, poor competitor', 'mildly invasive in disturbed habitats', 'highly invasive with high reproductive rate')")
    succession_role: Optional[str] = Field(None, description="Role in ecological succession (e.g., 'early pioneer species', 'late successional shade-tolerant', 'climax community member')")


class BiochemicalProfile(BaseModel):
    secondary_metabolites: list[str] = Field(default_factory=list, description="Biologically active compounds (e.g., 'alkaloids (nicotine analogues)', 'phenolic glycosides', 'essential oils (monoterpenes, sesquiterpenes)')")
    medicinal_compounds: list[str] = Field(default_factory=list, description="Pharmaceutical potential and compounds (e.g., 'salicylic acid (pain relief)', 'cardiac glycosides for arrhythmias', 'tannins for astringency')")
    toxicity: Optional[str] = Field(None, description="Toxicity profile if applicable (e.g., 'non-toxic', 'mildly toxic with skin irritation', 'highly toxic alkaloids >100 ppm')")
    ethnobotanical_uses: list[str] = Field(default_factory=list, description="Traditional medicinal or cultural uses (e.g., 'used in traditional Chinese medicine for inflammation', 'ceremonial importance in Indigenous cultures', 'culinary herb')")


class EducationalContent(BaseModel):
    learning_objectives_students: list[str] = Field(default_factory=list, description="Key concepts for botany students (e.g., 'understand angiosperm reproductive structures', 'identify monocot vs dicot characteristics', 'observe photoperiodic flowering responses')")
    anatomical_features_to_examine: list[str] = Field(default_factory=list, description="Structures suitable for microscopic study (e.g., 'transverse stem sections showing vascular bundles', 'leaf epidermis with stomatal apparatus', 'floral bud dissections')")
    historical_significance: Optional[str] = Field(None, description="Historical context and research importance (e.g., 'Mendel used garden peas for heredity studies', 'model organism for photosynthesis research', 'important food crop for 10,000 years')")
    related_species_for_comparison: list[str] = Field(default_factory=list, description="Similar species for comparative anatomy (e.g., 'compare with Rosa canina (wild rose) for cultivar differences', 'contrast with wind-pollinated grasses')")


class FlowerInfo(BaseModel):
    flower_name: str = Field(..., description="Common name and official botanical name with author and year (e.g., 'Rose (Rosa damascena Mill., 1768)')")
    summary: str = Field(..., description="2-3 sentence overview covering WHAT it is, KEY CHARACTERISTICS, and WHY it's valued. Write for general gardening audience")
    category: str = Field(..., description="Plant category (e.g., 'Perennial', 'Annual', 'Shrub', 'Climbing Vine', 'Bulb')")
    flower_type: list[str] = Field(default_factory=list, description="Type of flower (e.g., 'single bloom', 'double bloom', 'clustered flowers', 'composite flower')")

    # Core botanical sections
    taxonomy: Optional[Taxonomy] = Field(None, description="Scientific classification with taxonomic authority and origin")
    morphology: Optional[MorphologyStructure] = Field(None, description="Detailed plant and flower physical structure with anatomical details")
    bloom_characteristics: Optional[BloomCharacteristics] = Field(None, description="Flowering patterns, reproductive biology, and appearance")

    # Cultivation sections
    growth_requirements: Optional[GrowthRequirements] = Field(None, description="Environmental needs with physiological mechanisms")
    cultivation_care: Optional[CultivationCare] = Field(None, description="Planting, maintenance, propagation, and pest management")
    uses_companions: Optional[UsesCompanions] = Field(None, description="Garden applications, ecological interactions, and wildlife value")

    # Expert and specialist sections
    botanical_ecology: Optional[BotanicalEcology] = Field(None, description="Native habitat, ecological role, and conservation considerations")
    biochemical_profile: Optional[BiochemicalProfile] = Field(None, description="Secondary metabolites, medicinal compounds, and ethnobotanical uses")
    educational_content: Optional[EducationalContent] = Field(None, description="Learning resources for botany students and research applications")

    # Summary information
    hardiness_rating: Optional[str] = Field(None, description="Overall ease of growing (e.g., 'easy for beginners', 'moderate, requires experience', 'challenging, expert only')")
    maintenance_level: Optional[str] = Field(None, description="Time and effort required (e.g., 'low maintenance', 'moderate care needed', 'high maintenance')")
    special_features: list[str] = Field(default_factory=list, description="Unique botanical or horticultural characteristics (e.g., 'extended dormancy period', 'genetically diverse cultivars', 'photoperiodsensitive')")
    similar_species: list[str] = Field(default_factory=list, description="Related species with distinguishing morphological features (e.g., 'Rosa canina differs by single vs double flowers', 'Dahlia coccinea has simple orange flowers vs complex hybrids')")
    references: list[str] = Field(default_factory=list, description="Primary botanical references, research institutions, and scientific societies (e.g., 'Kew Royal Botanic Gardens', 'Flora of North America Editorial Committee', 'Flora Europaea')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate comprehensive flower information using multiple AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flowerinfo.py rose
  python flowerinfo.py "tulip" --models gemma3n 
  python flowerinfo.py sunflower --temperature 0.5 --output-dir my_flowers
        """
    )

    parser.add_argument(
        "flower",
        help="Name of the flower to get information about"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama"],
        help="AI models to query (default: llama)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for model responses (default: 0.3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="flowers",
        help="Directory to save JSON files (default: flowers)"
    )

    args = parser.parse_args()

    flower = args.flower
    models = args.models

    prompt = f"""You are a professional botanist and horticulturist creating a comprehensive flower reference for both botany students and expert practitioners on: {flower}

CRITICAL INSTRUCTIONS FOR ACCURACY:
- Use CORRECT scientific nomenclature: include species author and year (e.g., "Rosa damascena Mill., 1768")
- Verify chromosome numbers and ploidy levels (2n=XX, diploid/triploid/etc.)
- Include correct family, order, and higher taxa
- Distinguish between wild species, subspecies, and cultivated hybrids
- For hardy zones: correctly specify true hardiness (distinguish between tender perennials needing lift vs true perennials)

SPECIFIC BOTANICAL DATA REQUIREMENTS:
- Morphology: include vascular anatomy, venation patterns, trichome types, stomatal types
- Flowers: provide floral formula (K/C/A/G structure), floral symmetry, ovary position
- Reproduction: specify pollination type, sexual compatibility, breeding systems
- Ecology: native habitat type, elevation range, soil indicators, ecological relationships
- Biochemistry: secondary metabolites (with chemical classes), alkaloid content, medicinal compounds
- Propagation: germination percentages, stratification requirements, rooting hormone sensitivity

QUANTITATIVE SPECIFICITY:
- Always use ranges with units: "1.5-2.5 meters tall" not "medium height"
- Include soil parameters: pH 6.0-7.0, not "slightly acidic"
- Specify light intensity: full sun = 6-8+ hours direct light, not just "full sun"
- Temperature thresholds: "cold injury below 32°F" with frost dates
- Bloom counts: "200-500 flowers per mature plant" not "abundant"
- Photoperiodism: specify hours required (e.g., "<12 hours for flowering")

FOR STUDENTS (LEARNING OBJECTIVES):
- Explain WHY structures exist: how do adaptations serve the plant's ecology?
- Compare with related species: what distinguishes this taxon?
- Suggest microscopic observations: which tissues to examine for anatomy lessons
- Connect to plant physiology: how do environmental factors affect processes?

FOR EXPERTS (RESEARCH CONTEXT):
- Cite notable cultivars and their breeding pedigrees
- Include genetic/taxonomic complexity: hybrid vigor, polyploidy effects
- Reference landmark botanical research on this species
- Discuss conservation status and population genetics if applicable
- Describe tissue culture and micropropagation potential

Provide COMPLETE information covering all these sections:
1. TAXONOMY: Correct scientific name with authority, family, order, chromosome number
2. MORPHOLOGY: Anatomical details, vascular tissue arrangement, detailed leaf/stem/root description
3. BLOOM CHARACTERISTICS: Floral formula, pollination type, reproductive biology, flowering triggers
4. GROWTH REQUIREMENTS: Photosynthetic pathway, soil requirements, physiological limits
5. CULTIVATION: Propagation success rates, seed viability, breeding notes
6. ECOLOGY: Native habitat classification, ecological role, invasive potential, succession role
7. BIOCHEMISTRY: Secondary metabolites with chemical classes, medicinal compounds, toxicity
8. EDUCATION: Learning objectives, anatomical features for study, historical significance, comparison species"""

    print(f"Generating information on: {flower}", file=sys.stderr)
    print(f"Using models: {', '.join(models)}", file=sys.stderr)
    print(f"Temperature: {args.temperature}", file=sys.stderr)

    # Setup output file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    safe_flower = "".join(c if c.isalnum() else "_" for c in flower)
    json_file = output_dir / f"{safe_flower}.json"

    # Create new structure (overwrite existing file)
    output = {
        "flower": flower,
        "answers": []
    }

    # Query each model and append immediately
    for model in models:
        print(f"  Querying {model}...", file=sys.stderr)

        try:
            client = OpenRouterClient()
            # Create config with the model specified
            config = ModelConfig(model=model, temperature=args.temperature)
            response = client.generate_content(
                ModelInput(
                    user_prompt=prompt,
                    response_model=FlowerInfo
                ),
                config=config
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
    print(f"Total responses: {len([r for r in output['answers'] if 'data' in r])}", file=sys.stderr)
