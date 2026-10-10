"""
AgriIntel — Government Scheme Matcher
Matches farmer profile to eligible central/state government schemes.

DISCLAIMER: scheme benefits/eligibility are a simplified snapshot (Oct 2026) and
are NOT officially verified. They change often - always confirm on the official
portals before advising a real farmer.
"""

SCHEMES = [
    {
        "id": "pm_kisan",
        "name": "PM-KISAN",
        "benefit": "₹6,000/year direct income support in 3 installments",
        "apply": "pmkisan.gov.in or nearest CSC",
        "eligibility": lambda f: f.get("ownership") in ["owned", "own"],
        "category": "income_support"
    },
    {
        "id": "kcc",
        "name": "Kisan Credit Card (KCC)",
        "benefit": "Short-term crop loan at 7% interest (4% with subsidy)",
        "apply": "Any nationalized bank branch",
        "eligibility": lambda f: f.get("land_size", 0) > 0,
        "category": "credit"
    },
    {
        "id": "pmfby",
        "name": "PM Fasal Bima Yojana (PMFBY)",
        "benefit": "Crop insurance — up to full sum insured on crop failure",
        "apply": "Bank or insurance company at loan time",
        "eligibility": lambda f: True,  # All farmers eligible
        "category": "insurance"
    },
    {
        "id": "soil_health",
        "name": "Soil Health Card Scheme",
        "benefit": "Free soil testing + fertilizer recommendations",
        "apply": "District agriculture office",
        "eligibility": lambda f: True,
        "category": "soil"
    },
    {
        "id": "drip_subsidy",
        "name": "PM Krishi Sinchayee Yojana",
        "benefit": "55-75% subsidy on drip/sprinkler irrigation systems",
        "apply": "State agriculture department",
        "eligibility": lambda f: f.get("irrigation_type","").lower() in ["rainfed","canal","none",""],
        "category": "irrigation"
    },
    {
        "id": "agri_infra",
        "name": "Agriculture Infrastructure Fund",
        "benefit": "Low-cost loans for storage, processing units (3% interest subvention)",
        "apply": "agriinfra.dac.gov.in",
        "eligibility": lambda f: f.get("land_size", 0) >= 2,
        "category": "infrastructure"
    },
    {
        "id": "nabard_rig",
        "name": "NABARD — Rural Infrastructure Development",
        "benefit": "Subsidized loans for rural godowns and cold storage",
        "apply": "Nearest NABARD district office",
        "eligibility": lambda f: f.get("land_size", 0) >= 3,
        "category": "storage"
    },
    {
        "id": "fpo_scheme",
        "name": "Formation & Promotion of FPOs",
        "benefit": "Up to ₹18 lakh equity grant for forming Farmer Producer Org.",
        "apply": "SFAC or NABARD district office",
        "eligibility": lambda f: f.get("land_size", 0) >= 1,
        "category": "collective"
    },
    {
        "id": "e_nam",
        "name": "e-NAM (National Agriculture Market)",
        "benefit": "Online mandi access — sell crops directly at best price",
        "apply": "enam.gov.in — register with Aadhaar",
        "eligibility": lambda f: True,
        "category": "market"
    },
    {
        "id": "paramparagat",
        "name": "Paramparagat Krishi Vikas Yojana",
        "benefit": "₹50,000/hectare support for organic farming certification",
        "apply": "State agriculture department",
        "eligibility": lambda f: f.get("crop_type","").lower() not in ["cotton","sugarcane"],
        "category": "organic"
    },
]


def match_schemes(farmer: dict) -> list:
    """
    Match farmer to eligible government schemes.
    Returns list of matched scheme dicts.
    """
    matched = []
    for scheme in SCHEMES:
        try:
            if scheme["eligibility"](farmer):
                matched.append({
                    "name":    scheme["name"],
                    "benefit": scheme["benefit"],
                    "apply":   scheme["apply"],
                    "category": scheme["category"]
                })
        except Exception:
            continue
    return matched


def get_crop_switch_recommendation(
    farmer: dict,
    current_income: float
) -> dict | None:
    """
    Recommend a crop switch if a better-yielding crop fits the farmer's profile.
    Returns dict with crop name and estimated income uplift, or None.
    """
    # Income uplift estimates by crop (simplified — in production use ML)
    # Deliberately conservative relative-income factors (illustrative only).
    # Real uplift depends on local prices, inputs, risk and market access.
    CROP_INCOME_FACTORS = {
        "paddy":      1.00,
        "rice":       1.00,
        "wheat":      1.05,
        "mustard":    1.15,
        "soybean":    1.12,
        "maize":      1.10,
        "vegetables": 1.30,
        "onion":      1.35,
        "tomato":     1.30,
        "cotton":     1.15,
        "sugarcane":  1.15,
        "pulses":     1.08,
        "sunflower":  1.10,
    }
    MAX_UPLIFT_PCT = 0.35   # never suggest more than +35% of current income

    current_crop = farmer.get("crop_type","").lower().strip()
    current_factor = CROP_INCOME_FACTORS.get(current_crop, 1.0)

    # Find best alternative given irrigation
    has_irrigation = farmer.get("irrigation_type","").lower() not in ["rainfed","none",""]
    land_size = farmer.get("land_size", 0)

    # Filter candidates
    candidates = []
    for crop, factor in CROP_INCOME_FACTORS.items():
        if crop == current_crop:
            continue
        if factor <= current_factor:
            continue
        # Water-intensive crops only if irrigation exists
        if crop in ["vegetables","onion","tomato","sugarcane"] and not has_irrigation:
            continue
        # Large land only for certain crops
        if crop == "sugarcane" and land_size < 2:
            continue
        candidates.append((crop, factor))

    if not candidates:
        return None

    # Best candidate
    best_crop, best_factor = max(candidates, key=lambda x: x[1])
    uplift = min(current_income * (best_factor - current_factor),
                 current_income * MAX_UPLIFT_PCT)

    if uplift < 5000:  # Not worth recommending tiny uplifts
        return None

    return {
        "crop":           best_crop.capitalize(),
        "uplift":         round(uplift, -2),
        "estimated_income": round(current_income + uplift, -2),
        "requires_irrigation": best_crop in ["vegetables","onion","tomato","sugarcane"],
        "note": "Indicative estimate - validate with local prices and risk"
    }
