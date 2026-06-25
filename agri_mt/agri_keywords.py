"""Agricultural domain lexicon for mining/filtering parallel sentences.

Two tiers:
  * GLOSSARY terms (from the KG) -> high-precision, Bangladesh-rice-specific.
  * BROAD agri keywords         -> higher-recall English cues for estimating how
                                   much of a general corpus is agriculture.
The broad list is intentionally generic (crops, inputs, operations, agronomy)
so the mining-yield probe is not under-counted by the tiny 28-term glossary.
"""
from __future__ import annotations

BROAD_AGRI_KEYWORDS_EN = {
    # crops / plant parts
    "rice", "paddy", "wheat", "jute", "potato", "maize", "corn", "tomato", "mango",
    "crop", "crops", "seedling", "seed", "seeds", "grain", "grains", "tiller",
    "panicle", "leaf", "leaves", "root", "roots", "stem", "shoot", "harvest",
    # diseases / pests
    "disease", "diseases", "blight", "blast", "tungro", "rot", "pest", "pests",
    "insect", "insects", "borer", "planthopper", "aphid", "fungus", "fungal",
    "bacterial", "infestation", "infection", "weed", "weeds",
    # inputs / chemicals
    "fertilizer", "fertiliser", "urea", "compost", "manure", "pesticide",
    "insecticide", "fungicide", "herbicide", "nutrient", "nitrogen", "phosphorus",
    "potassium", "potash", "zinc", "spray", "dose", "dosage", "application rate",
    # operations / agronomy
    "irrigation", "irrigate", "sowing", "transplant", "transplanting", "cultivation",
    "cultivate", "tillage", "plough", "ploughing", "weeding", "drainage", "drain",
    "soil", "field", "farm", "farmer", "farmers", "agriculture", "agricultural",
    "yield", "agronomic", "germination", "flowering", "ripening", "acre", "hectare",
}


def load_glossary_en_forms(glossary_json: str = "data/glossary.json") -> set[str]:
    import json
    from pathlib import Path

    forms: set[str] = set()
    for g in json.loads(Path(glossary_json).read_text(encoding="utf-8")):
        for f in [g["en"], *g.get("en_aliases", [])]:
            if f:
                forms.add(f.lower())
    return forms
