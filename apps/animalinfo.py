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
    kingdom: Optional[str] = Field(None, description="Kingdom (e.g., 'Animalia')")
    phylum: Optional[str] = Field(None, description="Phylum (e.g., 'Chordata', 'Arthropoda', 'Mollusca')")
    class_name: Optional[str] = Field(None, description="Class (e.g., 'Mammalia', 'Aves', 'Reptilia')")
    order: Optional[str] = Field(None, description="Order (e.g., 'Carnivora', 'Primates', 'Passeriformes')")
    family: Optional[str] = Field(None, description="Family (e.g., 'Felidae', 'Hominidae', 'Accipitridae')")
    genus: Optional[str] = Field(None, description="Genus name (e.g., 'Panthera', 'Homo', 'Aquila')")
    species: Optional[str] = Field(None, description="Species name with binomial nomenclature (e.g., 'Panthera leo Linnaeus, 1758')")
    common_names: list[str] = Field(default_factory=list, description="Common names in different regions/languages")
    native_range: Optional[str] = Field(None, description="Geographic distribution with specific continents, regions, and habitat types")
    conservation_status: Optional[str] = Field(None, description="IUCN status (e.g., 'Least Concern', 'Vulnerable', 'Critically Endangered')")
    chromosome_number: Optional[str] = Field(None, description="Chromosome count (e.g., '2n=38' for domestic dog)")
    subspecies: list[str] = Field(default_factory=list, description="Recognized subspecies with geographic distributions")


class MorphologyStructure(BaseModel):
    body_type: Optional[str] = Field(None, description="Body plan (e.g., 'quadrupedal mammal', 'bipedal primate', 'aquatic cetacean')")
    size_weight: Optional[str] = Field(None, description="Typical weight and size range with sexual dimorphism if applicable (e.g., 'males 150-200 kg, females 100-150 kg')")
    height_length: Optional[str] = Field(None, description="Typical body length or height (e.g., '2.5-3.5 meters long', '170-180 cm tall')")
    head_structure: list[str] = Field(default_factory=list, description="Cranial features: skull shape, jaw structure, dentition (e.g., 'elongated snout with fused premaxilla', 'carnassial teeth adapted for shearing')")
    skeletal_features: list[str] = Field(default_factory=list, description="Skeletal adaptations and bone structure (e.g., 'hollow bones for flight', 'robust vertebral column', 'fused caudal vertebrae')")
    limbs_digits: Optional[str] = Field(None, description="Limb structure and digit modifications (e.g., 'pentadactyl limbs with retractable claws', 'forelimbs modified as wings', 'webbed feet for aquatic locomotion')")
    skin_covering: list[str] = Field(default_factory=list, description="Skin, fur, feathers, or scales with specific adaptations (e.g., 'thick double-layer coat for insulation', 'iridescent feathers with structural coloration', 'keeled scales on dorsal surface')")
    sensory_organs: list[str] = Field(default_factory=list, description="Eyes, ears, nose with sensory capabilities (e.g., 'forward-facing eyes for binocular vision', 'highly developed olfactory system with 1000+ receptors', 'compound eyes with 30,000 ommatidia')")
    coloration_patterns: Optional[str] = Field(None, description="Base coloration and markings with pigmentation types (e.g., 'golden-tan with dark mane', 'black and white stripes', 'cryptic brown coloration with disruptive spots')")


class Physiology(BaseModel):
    metabolic_rate: Optional[str] = Field(None, description="Basal metabolic rate and thermoregulation type (e.g., 'endothermic with core temperature 37°C', 'ectothermic, temperature-dependent metabolism')")
    respiration: Optional[str] = Field(None, description="Respiratory system details (e.g., 'lungs with diaphragm for breathing', 'gill slits with counter-current exchange', 'tracheal system with spiracles')")
    circulation: Optional[str] = Field(None, description="Circulatory system (e.g., 'four-chambered heart with complete septation', 'three-chambered heart with mixing', 'open hemocoel system')")
    digestion: Optional[str] = Field(None, description="Digestive system and feeding adaptations (e.g., 'herbivorous with ruminant stomachs and cecal fermentation', 'carnivorous with short simple digestive tract', 'filter-feeding mechanism')")
    lifespan: Optional[str] = Field(None, description="Typical lifespan in wild and captivity (e.g., '15-20 years in wild, 20-25 in captivity')")
    energy_requirements: Optional[str] = Field(None, description="Daily caloric intake and feeding frequency (e.g., '2-3 kg of food daily', '60% body weight per week')")
    water_requirements: Optional[str] = Field(None, description="Hydration needs and water conservation mechanisms (e.g., 'requires daily water access', 'adapted for desert with minimal water loss', 'marine salt excretion via salt glands')")


class ReproductionDevelopment(BaseModel):
    reproductive_system: Optional[str] = Field(None, description="Sexual or asexual reproduction details (e.g., 'dioecious with internal fertilization', 'hermaphroditic with sequential hermaphroditism', 'parthenogenetic')")
    mating_system: Optional[str] = Field(None, description="Breeding behavior (e.g., 'polygynous harem system', 'monogamous pair bond', 'polyandrous with female competition')")
    gestation_incubation: Optional[str] = Field(None, description="Duration of pregnancy/gestation or egg incubation (e.g., 'gestational period 263-280 days', '21-day incubation period')")
    offspring_number: Optional[str] = Field(None, description="Number of young per breeding event (e.g., 'single calf per 2-3 years', 'clutch of 8-12 eggs', 'litter of 3-5 pups')")
    parental_care: Optional[str] = Field(None, description="Type and duration of parental investment (e.g., 'extensive maternal care for 12+ months', 'biparental care with division of labor', 'no parental care')")
    sexual_maturity: Optional[str] = Field(None, description="Age at reproductive maturity (e.g., 'reaches sexual maturity at 5-7 years', 'breeds at 4-5 weeks of age')")
    breeding_season: Optional[str] = Field(None, description="Annual breeding patterns (e.g., 'seasonal breeder, mating in December', 'year-round breeding', 'biennial reproduction')")


class Behavior(BaseModel):
    social_structure: Optional[str] = Field(None, description="Group organization (e.g., 'highly social, groups of 20-50 individuals', 'solitary except during breeding', 'hierarchical packs with alpha pair')")
    activity_pattern: Optional[str] = Field(None, description="Daily activity rhythm (e.g., 'diurnal, active during daylight', 'nocturnal hunter', 'crepuscular, most active at dawn/dusk')")
    movement_locomotion: list[str] = Field(default_factory=list, description="Movement methods and daily travel distances (e.g., 'quadrupedal runner reaching 60 km/h', 'arboreal climber with brachiation', 'aquatic swimmer covering 10-20 km daily')")
    feeding_behavior: list[str] = Field(default_factory=list, description="Foraging and feeding strategies (e.g., 'apex predator with cooperative hunting in packs', 'solitary ambush predator', 'selective browser on specific plant species')")
    communication: list[str] = Field(default_factory=list, description="Vocalizations, visual signals, chemical communication (e.g., 'complex vocalizations with 15+ call types', 'scent marking with anal glands', 'visual threat displays with ear positions')")
    territorial_behavior: Optional[str] = Field(None, description="Territory maintenance and home range (e.g., 'territorial with 5-10 km² home range', 'range overlap with non-violent coexistence', 'nomadic migration pattern')")
    play_behavior: Optional[str] = Field(None, description="Play activities especially in juveniles (e.g., 'extensive juvenile play-fighting for skill development', 'minimal play behavior', 'object manipulation and problem-solving play')")
    sleep_rest: Optional[str] = Field(None, description="Sleep patterns and rest requirements (e.g., 'polyphasic sleeper with 16+ naps daily', 'consolidated 6-8 hour sleep period', 'unihemispheric slow-wave sleep in aquatic species')")


class Diet(BaseModel):
    feeding_type: Optional[str] = Field(None, description="Dietary classification (e.g., 'carnivore', 'herbivore', 'omnivore', 'insectivore', 'granivore')")
    primary_prey: list[str] = Field(default_factory=list, description="Main food sources with size ranges (e.g., 'medium ungulates 100-500 kg', 'insects and small vertebrates', 'seeds and fruits from specific plant species')")
    hunting_method: Optional[str] = Field(None, description="Predation technique (e.g., 'pursuit predator with 100 km/h sprint speed', 'ambush predator from vegetation', 'filter feeder with baleen plates')")
    prey_selection: Optional[str] = Field(None, description="Prey preferences and selection criteria (e.g., 'preferentially hunts young or weakened individuals', 'size-selective foraging')")
    food_consumption: Optional[str] = Field(None, description="Daily/weekly food intake (e.g., '2-3% of body weight daily', 'single large kill lasts 3-5 days', 'continuous feeding as a filter feeder')")
    seasonal_diet_variation: Optional[str] = Field(None, description="Changes in diet by season (e.g., 'switches from fruit to insects seasonally', 'winter diet relies on stored fat reserves')")
    trophic_level: Optional[str] = Field(None, description="Position in food chain (e.g., 'apex predator', 'secondary consumer', 'primary consumer')")


class HabitatRange(BaseModel):
    habitat_type: list[str] = Field(default_factory=list, description="Preferred habitats with specific characteristics (e.g., 'tropical rainforest canopy', 'grassland savanna with sparse trees', 'coral reef', 'deep ocean hydrothermal vents')")
    altitude_depth_range: Optional[str] = Field(None, description="Elevation or depth tolerance (e.g., 'sea level to 4000 m elevation', '0-200 m ocean depth', 'deep ocean 1000-3000 m')")
    climate_preference: Optional[str] = Field(None, description="Temperature and precipitation needs (e.g., 'tropical 20-30°C, high humidity', 'desert adapted to <500 mm annual rainfall')")
    home_range_size: Optional[str] = Field(None, description="Territory or range size (e.g., '10-50 km² per individual', '100+ km² migratory range')")
    habitat_generalist_specialist: Optional[str] = Field(None, description="Ecological niche breadth (e.g., 'habitat specialist requiring pristine old-growth forest', 'generalist adapting to multiple habitat types')")


class Ecology(BaseModel):
    ecological_role: Optional[str] = Field(None, description="Function in ecosystem (e.g., 'apex predator regulating prey populations', 'seed disperser for forest regeneration', 'detritivore in decomposition cycle')")
    predators: list[str] = Field(default_factory=list, description="Natural predators and threats")
    symbiosis_parasitism: list[str] = Field(default_factory=list, description="Symbiotic relationships and parasites (e.g., 'mutualistic with cleaner fish', 'host to tick parasites', 'endosymbiotic bacteria in digestive system')")
    population_dynamics: Optional[str] = Field(None, description="Population growth rates and density dependence (e.g., 'r-strategy with rapid reproduction', 'K-strategy with low fecundity', 'cyclic population fluctuations')")
    migration_patterns: Optional[str] = Field(None, description="Migration routes and triggers (e.g., 'long-distance migration of 8000 km annually', 'latitudinal migration following prey', 'local vertical migration')")
    invasive_potential: Optional[str] = Field(None, description="Risk as invasive species (e.g., 'non-invasive, poor competitor', 'invasive with rapid population growth in new habitats')")


class CognitionBehavior(BaseModel):
    intelligence_level: Optional[str] = Field(None, description="Cognitive capabilities (e.g., 'highly intelligent with problem-solving ability', 'simple reflexive responses', 'tool use documented')")
    learning_capacity: Optional[str] = Field(None, description="Ability to learn from experience (e.g., 'learns through observational learning', 'minimal learning capacity', 'develops individual hunting strategies')")
    memory: Optional[str] = Field(None, description="Memory capabilities and temporal scope (e.g., 'episodic memory spanning months', 'spatial memory for 50+ locations', 'recognition of 100+ individuals')")
    personality_differences: Optional[str] = Field(None, description="Individual behavioral variation (e.g., 'bold vs shy personality types', 'consistent individual differences in aggression')")


class EducationalContent(BaseModel):
    evolutionary_significance: Optional[str] = Field(None, description="Evolutionary history and phylogenetic position (e.g., 'living fossil with lineage dating to Triassic', 'model organism for evolutionary research')")
    conservation_efforts: list[str] = Field(default_factory=list, description="Current conservation initiatives (e.g., 'breeding programs in zoos', 'habitat protection legislation', 'population monitoring projects')")
    human_interaction: list[str] = Field(default_factory=list, description="Cultural significance and human relationships (e.g., 'domesticated for 10,000 years', 'sacred animal in multiple cultures', 'crop pest causing agricultural damage')")
    research_model: Optional[str] = Field(None, description="Scientific research applications (e.g., 'model organism for neuroscience research', 'indicator species for ecosystem health')")
    related_species_for_comparison: list[str] = Field(default_factory=list, description="Similar species for comparative study (e.g., 'compare with congener X for niche partitioning', 'contrast with sister species Y for morphological evolution')")


class AnimalInfo(BaseModel):
    animal_name: str = Field(..., description="Common name and scientific binomial nomenclature with authority (e.g., 'African Lion (Panthera leo Linnaeus, 1758)')")
    summary: str = Field(..., description="2-3 sentence overview covering WHAT it is, KEY CHARACTERISTICS, and ECOLOGICAL IMPORTANCE. Write for general educated audience")
    category: str = Field(..., description="Animal group (e.g., 'Mammal', 'Bird', 'Reptile', 'Amphibian', 'Fish', 'Insect', 'Marine Mammal')")
    animal_type: list[str] = Field(default_factory=list, description="Specific type designation (e.g., 'carnivore', 'great ape', 'migratory bird', 'venomous snake')")

    # Core zoological sections
    taxonomy: Optional[Taxonomy] = Field(None, description="Scientific classification with nomenclatural authority and evolutionary relationships")
    morphology: Optional[MorphologyStructure] = Field(None, description="Detailed anatomical structure and morphological adaptations")
    physiology: Optional[Physiology] = Field(None, description="Physiological systems and metabolic processes")
    reproduction: Optional[ReproductionDevelopment] = Field(None, description="Reproductive biology and developmental patterns")

    # Behavioral and ecological sections
    behavior: Optional[Behavior] = Field(None, description="Behavioral patterns, social structures, and communication")
    diet: Optional[Diet] = Field(None, description="Feeding ecology and nutritional requirements")
    habitat_range: Optional[HabitatRange] = Field(None, description="Geographic distribution and habitat requirements")
    ecology: Optional[Ecology] = Field(None, description="Ecological role, population dynamics, and ecosystem interactions")
    cognition: Optional[CognitionBehavior] = Field(None, description="Intelligence, learning, and cognitive abilities")

    # Synthesis sections
    adaptation_summary: Optional[str] = Field(None, description="Key adaptations and evolutionary fitness advantages (e.g., 'nocturnal adaptations enable night hunting', 'cryptic coloration provides camouflage')")
    conservation_status_detail: Optional[str] = Field(None, description="Conservation challenges and current protection status with details (e.g., 'Endangered due to habitat loss of 95% in last century', 'Protected under CITES Appendix I')")
    educational_content: Optional[EducationalContent] = Field(None, description="Scientific significance and educational value")

    # Summary information
    hardiness_rating: Optional[str] = Field(None, description="Resilience and adaptability (e.g., 'highly adaptable generalist', 'specialized with narrow ecological requirements', 'vulnerable to environmental change')")
    unique_features: list[str] = Field(default_factory=list, description="Distinctive characteristics (e.g., 'only mammal capable of true flight', 'longest migration of any animal', 'venomous but docile temperament')")
    similar_species: list[str] = Field(default_factory=list, description="Related species with key differences (e.g., 'differs from Species X by size and coloration', 'sympatric with closely related Species Y, occupies different niche')")
    references: list[str] = Field(default_factory=list, description="Scientific literature and research institutions (e.g., 'IUCN Red List', 'Zoological Society of London', 'major peer-reviewed wildlife studies')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate comprehensive animal information using multiple AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python animalinfo.py "African Lion"
  python animalinfo.py "red panda" --models gemma3n
  python animalinfo.py "gray wolf" --temperature 0.5 --output-dir my_animals
        """
    )

    parser.add_argument(
        "animal",
        help="Name of the animal to get information about"
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
        default="animals",
        help="Directory to save JSON files (default: animals)"
    )

    args = parser.parse_args()

    animal = args.animal
    models = args.models

    prompt = f"""You are a professional zoologist and animal behaviorist creating a comprehensive animal reference for both biology students and expert practitioners on: {animal}

CRITICAL INSTRUCTIONS FOR ACCURACY:
- Use CORRECT scientific nomenclature: include species author and year (e.g., "Panthera leo Linnaeus, 1758")
- Verify taxonomic classification: kingdom, phylum, class, order, family, genus, species
- Include recognized subspecies and their geographic ranges
- Distinguish between wild populations and domesticated/captive variants
- Verify chromosome numbers and karyotype information
- Include current IUCN conservation status with accurate assessment year

SPECIFIC ZOOLOGICAL DATA REQUIREMENTS:
- Morphology: include skeletal features, muscular adaptations, organ system specializations
- Physiology: metabolic rate, thermoregulation mechanisms, cardiovascular adaptations
- Reproduction: gestation periods, clutch/litter sizes, breeding season timing, parental care systems
- Behavior: social structures, communication types with specific examples, daily activity patterns
- Diet: detailed prey preferences with size ranges, hunting or feeding methods, caloric requirements
- Ecology: habitat requirements, home range sizes, population density, trophic role, predators
- Cognition: documented problem-solving, tool use, social learning, individual recognition abilities

QUANTITATIVE SPECIFICITY:
- Always use ranges with units: "2.5-3.5 meters long" not "large"
- Include weight ranges with sexual dimorphism: "males 70-110 kg, females 50-75 kg"
- Specify speed capabilities: "reaching speeds of 60 km/h" not just "fast"
- Metabolism rates: "basal metabolic rate of 2500 kcal/day" with context
- Group sizes: "typically lives in groups of 15-30 individuals" not "social groups"
- Geographic precision: specific countries/regions, not just continents
- Age at maturity: "sexual maturity at 4-6 years" with behavioral milestones

FOR STUDENTS (LEARNING OBJECTIVES):
- Explain WHY morphological features exist: how do adaptations enhance survival?
- Compare with related species: what distinguishes this taxon from close relatives?
- Connect to evolutionary history: what environmental pressures shaped these traits?
- Link behavior to ecology: how do behavioral patterns match habitat requirements?

FOR EXPERTS (RESEARCH CONTEXT):
- Include population genetics and genetic diversity if known
- Reference landmark behavioral or ecological studies
- Discuss conservation genetics and breeding programs
- Describe unusual cognitive abilities or problem-solving capabilities
- Include information on human-wildlife conflict where relevant
- Discuss physiological limits and stress tolerance thresholds

Provide COMPLETE information covering all these sections:
1. TAXONOMY: Correct scientific name with authority, classification, chromosome number, subspecies
2. MORPHOLOGY: Skeletal adaptations, sensory organs, size/weight ranges, coloration patterns
3. PHYSIOLOGY: Metabolic rate, thermoregulation, circulatory system, digestive specializations
4. REPRODUCTION: Breeding season, gestation/incubation, offspring numbers, parental care duration
5. BEHAVIOR: Social structure, communication systems, activity patterns, territorial behavior
6. DIET: Feeding type, prey preferences, hunting methods, caloric requirements
7. HABITAT/RANGE: Geographic distribution, habitat specificity, home range size, altitude/depth
8. ECOLOGY: Ecological role, predators, migrations, population dynamics, symbiotic relationships
9. COGNITION: Intelligence level, learning capacity, problem-solving, individual recognition"""

    print(f"Generating information on: {animal}", file=sys.stderr)
    print(f"Using models: {', '.join(models)}", file=sys.stderr)
    print(f"Temperature: {args.temperature}", file=sys.stderr)

    # Setup output file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    safe_animal = "".join(c if c.isalnum() else "_" for c in animal)
    json_file = output_dir / f"{safe_animal}.json"

    # Create new structure (overwrite existing file)
    output = {
        "animal": animal,
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
                    response_model=AnimalInfo
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
