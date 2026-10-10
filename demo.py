"""
╔══════════════════════════════════════════════════════════╗
║           AgriIntel — LIVE DEMO                          ║
║     Run this file to see ALL features in action          ║
║                                                          ║
║  HOW TO RUN:                                             ║
║     python demo.py                                       ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import os
import math
import json
import io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── colors for terminal output
class C:
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    GRAY   = "\033[90m"
    WHITE  = "\033[97m"
    RESET  = "\033[0m"

def header(title):
    print(f"\n{C.CYAN}{'═'*60}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}  {title}{C.RESET}")
    print(f"{C.CYAN}{'═'*60}{C.RESET}")

def section(title):
    print(f"\n{C.YELLOW}  ▶  {title}{C.RESET}")
    print(f"  {'─'*50}")

def ok(label, value, unit=""):
    print(f"  {C.GREEN}✓{C.RESET}  {C.BOLD}{label:<30}{C.RESET} {C.WHITE}{value}{unit}{C.RESET}")

def info(msg):
    print(f"  {C.GRAY}    {msg}{C.RESET}")

def warn(msg):
    print(f"  {C.YELLOW}⚠  {msg}{C.RESET}")

def alert(msg):
    print(f"  {C.RED}🔴  {msg}{C.RESET}")

# ════════════════════════════════════════════════════════════
# CORE PREDICTION ENGINE (same logic as main.py but standalone)
# ════════════════════════════════════════════════════════════

CROP_BOOST = {
    "vegetables": 1.8, "onion": 2.0, "tomato": 1.7,
    "cotton": 1.4,  "sugarcane": 1.3, "mustard": 1.35,
    "soybean": 1.25,"maize": 1.2,  "wheat": 1.1,
    "pulses": 1.15, "paddy": 1.0,  "rice": 1.0,
}

IRR_BOOST = {
    "drip": 1.3, "sprinkler": 1.2, "borewell": 1.15,
    "canal": 1.1, "rainfed": 0.85
}

def predict_income(farmer: dict) -> dict:
    base       = farmer["land_size"] * 40000
    irr        = IRR_BOOST.get(farmer["irrigation_type"].lower(), 1.0)
    crop       = CROP_BOOST.get(farmer["crop_type"].lower(), 1.0)
    own        = 1.1 if farmer["ownership"].lower() == "owned" else 0.9
    income     = max(30000, min(base * irr * crop * own, 800000))
    return {
        "predicted_income": round(income, -2),
        "range_low":        round(income * 0.85, -2),
        "range_high":       round(income * 1.15, -2),
    }

def agri_score(farmer: dict, income: float) -> dict:
    score = 0
    # Income (40 pts)
    if income > 200000: score += 40
    elif income > 150000: score += 33
    elif income > 100000: score += 26
    elif income > 75000:  score += 18
    elif income > 50000:  score += 12
    else:                 score += 6
    # Irrigation (20 pts)
    score += {"drip":20,"sprinkler":17,"borewell":14,"canal":12,"rainfed":6}.get(
        farmer["irrigation_type"].lower(), 8)
    # Ownership (20 pts)
    score += {"owned":20,"leased":12,"rented":10,"share":7}.get(
        farmer["ownership"].lower(), 8)
    # Land size (20 pts)
    ls = farmer["land_size"]
    score += 20 if ls>10 else 16 if ls>5 else 12 if ls>2 else 8 if ls>1 else 4
    score = min(100, max(0, score))
    decision = "APPROVE" if score >= 70 else "REVIEW" if score >= 50 else "REJECT"
    return {"agri_score": score, "decision": decision}

def repayment_prob(farmer: dict, income: float, score: int) -> float:
    base = 50 + (score - 50) * 0.4
    if farmer["ownership"].lower() == "owned":           base += 10
    if farmer["irrigation_type"].lower() == "drip":      base += 8
    if income > 150000:                                  base += 10
    if farmer["crop_type"].lower() in ["paddy","wheat"]: base += 5
    return round(min(99, max(30, base)), 1)

def loan_recommend(income: float, score: int) -> dict:
    if score >= 70:   mult, tenure = 0.65, 36
    elif score >= 60: mult, tenure = 0.55, 24
    elif score >= 50: mult, tenure = 0.40, 24
    else:             mult, tenure = 0.20, 12
    loan   = round(income * mult, -3)
    rate   = 7.0 / 100 / 12
    emi    = loan * rate * (1+rate)**tenure / ((1+rate)**tenure - 1)
    return {"loan": loan, "emi": round(emi, 0), "tenure": tenure}

DISTRICT_AVG = {
    "jorhat":85000,"kamrup":105000,"dibrugarh":88000,
    "golaghat":92000,"nagaon":82000,"tinsukia":90000
}

def vs_district(district: str, income: float) -> float:
    avg = DISTRICT_AVG.get(district.lower(), 90000)
    return round((income - avg) / avg * 100, 1)

def crop_switch(farmer: dict, income: float):
    FACTORS = {
        "mustard":1.15,"soybean":1.12,"maize":1.10,"vegetables":1.30,
        "onion":1.35,"cotton":1.15,"wheat":1.05,"pulses":1.08
    }  # conservative, illustrative factors
    has_irr = farmer["irrigation_type"].lower() not in ["rainfed","none"]
    curr    = CROP_BOOST.get(farmer["crop_type"].lower(), 1.0)
    best_crop, best_factor = None, curr
    for c, f in FACTORS.items():
        if c == farmer["crop_type"].lower(): continue
        if f <= curr: continue
        if c in ["vegetables","onion","tomato"] and not has_irr: continue
        if f > best_factor:
            best_factor, best_crop = f, c
    if not best_crop: return None
    uplift = min(income * (best_factor - curr), income * 0.35)  # cap at +35%
    if uplift < 5000: return None
    return {"crop": best_crop.capitalize(), "uplift": round(uplift,-2),
            "new_income": round(income + uplift,-2)}

SCHEMES = [
    ("PM-KISAN", "₹6,000/year direct support", "pmkisan.gov.in",
     lambda f: f["ownership"].lower() == "owned"),
    ("Kisan Credit Card (KCC)", "Crop loan @ 4-7% interest", "Any bank branch",
     lambda f: True),
    ("PM Fasal Bima Yojana", "Crop insurance — full coverage", "Bank at loan time",
     lambda f: True),
    ("PM Krishi Sinchayee", "55-75% drip irrigation subsidy", "State agri dept",
     lambda f: f["irrigation_type"].lower() in ["rainfed","canal","none",""]),
    ("Soil Health Card", "Free soil testing + recommendations", "District agri office",
     lambda f: True),
    ("e-NAM", "Online mandi — sell at best price", "enam.gov.in",
     lambda f: True),
    ("Agri Infrastructure Fund", "Low-cost storage/processing loans", "agriinfra.dac.gov.in",
     lambda f: f["land_size"] >= 2),
]

def match_schemes(farmer: dict) -> list:
    return [(name, benefit, apply_at)
            for name, benefit, apply_at, check in SCHEMES
            if check(farmer)]

def insurance_premium(farmer: dict, income: float) -> dict:
    rate        = 0.02 if farmer.get("season","kharif").lower()=="kharif" else 0.015
    sum_insured = income * 1.5
    farmer_prem = round(sum_insured * rate, 0)
    govt_subsidy= round(sum_insured * 0.06, 0)
    return {"sum_insured": round(sum_insured,-2),
            "farmer_premium": farmer_prem,
            "govt_subsidy": govt_subsidy,
            "scheme": "PMFBY"}

def npa_risk_demo(temp: float, humidity: float, rainfall: float) -> dict:
    score = 0
    reasons = []
    if humidity < 30:  score += 3; reasons.append("Very low humidity — drought risk")
    elif humidity < 45: score += 1; reasons.append("Low humidity")
    if temp > 42:      score += 3; reasons.append("Extreme heat — crop stress")
    elif temp > 38:    score += 1; reasons.append("High temperature")
    if rainfall > 50:  score += 4; reasons.append("Heavy rainfall — flood risk")
    elif rainfall > 20: score += 2; reasons.append("Moderate rainfall")
    level  = "HIGH" if score>=6 else "MEDIUM" if score>=3 else "LOW"
    action = ("Immediate contact — consider restructuring" if level=="HIGH"
              else "Schedule courtesy call" if level=="MEDIUM"
              else "No action required")
    return {"level": level, "score": score, "reasons": reasons, "action": action}

# ════════════════════════════════════════════════════════════
# DEMO FARMERS
# ════════════════════════════════════════════════════════════

FARMERS = [
    {
        "name": "Ramesh Kumar",
        "district": "Jorhat", "state": "Assam",
        "crop_type": "paddy", "land_size": 2.5,
        "irrigation_type": "canal", "ownership": "owned",
        "season": "kharif", "phone": "9876543210"
    },
    {
        "name": "Sunita Devi",
        "district": "Kamrup", "state": "Assam",
        "crop_type": "vegetables", "land_size": 1.2,
        "irrigation_type": "drip", "ownership": "rented",
        "season": "rabi", "phone": "9123456789"
    },
    {
        "name": "Bijay Singh",
        "district": "Nagaon", "state": "Assam",
        "crop_type": "wheat", "land_size": 0.8,
        "irrigation_type": "rainfed", "ownership": "rented",
        "season": "rabi", "phone": "9988776655"
    },
]

# ════════════════════════════════════════════════════════════
# RUN DEMO
# ════════════════════════════════════════════════════════════

def run_demo():
    print(f"\n{C.BOLD}{C.CYAN}")
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║          AgriIntel — Live Platform Demo              ║")
    print("  ║     AI-Powered Farmer Credit Intelligence            ║")
    print("  ║              IIT Guwahati • 2025                     ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(f"{C.RESET}")

    # ── DEMO 1: FULL PREDICTION FOR EACH FARMER
    header("FEATURE 1–5: COMPLETE CREDIT ASSESSMENT")
    for farmer in FARMERS:
        section(f"Farmer: {farmer['name']} | {farmer['district']}, {farmer['state']}")
        ok("Crop Type",      farmer['crop_type'].capitalize())
        ok("Land Size",      farmer['land_size'], " acres")
        ok("Irrigation",     farmer['irrigation_type'].capitalize())
        ok("Ownership",      farmer['ownership'].capitalize())

        # Predictions
        inc   = predict_income(farmer)
        sc    = agri_score(farmer, inc["predicted_income"])
        rp    = repayment_prob(farmer, inc["predicted_income"], sc["agri_score"])
        ln    = loan_recommend(inc["predicted_income"], sc["agri_score"])
        vd    = vs_district(farmer["district"], inc["predicted_income"])

        print()
        ok("Predicted Annual Income",
           f"₹{inc['predicted_income']:,.0f}",
           f"  (Range: ₹{inc['range_low']:,.0f}–₹{inc['range_high']:,.0f})")
        ok("vs District Average",
           f"{'↑' if vd>=0 else '↓'} {abs(vd)}%",
           f"  ({'above' if vd>=0 else 'below'} average)")

        score_color = C.GREEN if sc['agri_score']>=70 else C.YELLOW if sc['agri_score']>=50 else C.RED
        dec_color   = C.GREEN if sc['decision']=="APPROVE" else C.YELLOW if sc['decision']=="REVIEW" else C.RED
        print(f"  {C.GREEN}✓{C.RESET}  {C.BOLD}{'AgriScore':<30}{C.RESET} "
              f"{score_color}{C.BOLD}{sc['agri_score']}/100{C.RESET}")
        print(f"  {C.GREEN}✓{C.RESET}  {C.BOLD}{'Decision':<30}{C.RESET} "
              f"{dec_color}{C.BOLD}{sc['decision']}{C.RESET}")
        ok("Repayment Probability",  f"{rp}%")
        ok("Recommended Loan",       f"₹{ln['loan']:,.0f}")
        ok("Monthly EMI",            f"₹{ln['emi']:,.0f}", f"  for {ln['tenure']} months @ 7%")

    # ── DEMO 2: BATCH PREDICTION
    header("FEATURE 6: BATCH PREDICTION")
    section("Simulating batch upload of 3 farmers")
    print(f"\n  {'Name':<20} {'Income':>12} {'Score':>8} {'Decision':>10} {'Loan':>12}")
    print(f"  {'─'*20} {'─'*12} {'─'*8} {'─'*10} {'─'*12}")
    for farmer in FARMERS:
        inc = predict_income(farmer)
        sc  = agri_score(farmer, inc["predicted_income"])
        ln  = loan_recommend(inc["predicted_income"], sc["agri_score"])
        dec_color = (C.GREEN if sc["decision"]=="APPROVE"
                     else C.YELLOW if sc["decision"]=="REVIEW" else C.RED)
        print(f"  {farmer['name']:<20} "
              f"₹{inc['predicted_income']:>10,.0f} "
              f"{sc['agri_score']:>7}/100 "
              f"{dec_color}{sc['decision']:>10}{C.RESET} "
              f"₹{ln['loan']:>10,.0f}")
    info("→ In real use: upload Excel with 500 rows, get this back as downloadable Excel")

    # ── DEMO 3: NPA EARLY WARNING
    header("FEATURE 7: NPA EARLY WARNING SYSTEM")
    weather_scenarios = [
        ("Jorhat — Normal monsoon",    28, 72, 5),
        ("Nagaon — Moderate dry spell",38, 38, 0),
        ("Dibrugarh — Drought alert",  44, 20, 0),
        ("Tinsukia — Flood risk",      30, 90, 65),
    ]
    for label, temp, humidity, rainfall in weather_scenarios:
        risk = npa_risk_demo(temp, humidity, rainfall)
        color = C.RED if risk["level"]=="HIGH" else C.YELLOW if risk["level"]=="MEDIUM" else C.GREEN
        print(f"\n  {color}●{C.RESET}  {C.BOLD}{label}{C.RESET}")
        print(f"     Weather: {temp}°C | Humidity: {humidity}% | Rainfall: {rainfall}mm/hr")
        print(f"     Risk: {color}{C.BOLD}{risk['level']}{C.RESET}  |  Action: {risk['action']}")
        for r in risk["reasons"]:
            print(f"     {C.GRAY}→ {r}{C.RESET}")

    # ── DEMO 4: PDF HEALTH CARD
    header("FEATURE 8: FARMER PDF HEALTH CARD")
    section("Generating PDF for Ramesh Kumar...")
    from utils.pdf_generator import generate_health_card
    farmer = FARMERS[0]
    inc = predict_income(farmer)
    sc  = agri_score(farmer, inc["predicted_income"])
    rp  = repayment_prob(farmer, inc["predicted_income"], sc["agri_score"])
    ln  = loan_recommend(inc["predicted_income"], sc["agri_score"])
    vd  = vs_district(farmer["district"], inc["predicted_income"])
    sw  = crop_switch(farmer, inc["predicted_income"])
    sch = match_schemes(farmer)

    pdf_data = {
        **farmer,
        "predicted_income":      inc["predicted_income"],
        "agri_score":            sc["agri_score"],
        "decision":              sc["decision"],
        "recommended_loan":      ln["loan"],
        "repayment_probability": rp,
        "income_vs_district":    vd,
        "generated_by":          "Demo Officer",
        "branch_name":           "Jorhat Branch, Assam Gramin Bank",
        "recommendations": [
            "Switch to drip irrigation — may raise income (illustrative estimate)",
            "Apply for KCC — interest subvention may apply (verify current terms)",
            "Register on e-NAM for better mandi prices",
        ],
        "matched_schemes": [
            {"name": s[0], "benefit": s[1], "apply": s[2]}
            for s in sch[:3]
        ],
        "crop_switch_suggestion": sw,
    }

    pdf_bytes = generate_health_card(pdf_data)
    os.makedirs("examples", exist_ok=True)
    pdf_path  = os.path.join("examples", "demo_farmer_health_card.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    ok("PDF generated", pdf_path, f"  ({len(pdf_bytes):,} bytes)")
    info("→ Open examples/demo_farmer_health_card.pdf to see the complete farmer report")

    # ── DEMO 5: GOVERNMENT SCHEMES
    header("FEATURE 12–13: GOVERNMENT SCHEMES + CROP SWITCH")
    for farmer in FARMERS:
        section(f"{farmer['name']} — {farmer['crop_type'].capitalize()}, {farmer['land_size']} acres")
        schemes = match_schemes(farmer)
        ok("Eligible Schemes", f"{len(schemes)} schemes found")
        for name, benefit, apply_at in schemes:
            print(f"  {C.GREEN}  ✓{C.RESET}  {C.BOLD}{name:<30}{C.RESET} {benefit}")

        inc = predict_income(farmer)
        sw  = crop_switch(farmer, inc["predicted_income"])
        if sw:
            print(f"\n  {C.YELLOW}💡 Crop Switch Recommendation:{C.RESET}")
            print(f"     Current: {farmer['crop_type'].capitalize()} → "
                  f"Recommended: {C.BOLD}{sw['crop']}{C.RESET}")
            print(f"     Income uplift: {C.GREEN}+₹{sw['uplift']:,.0f}/year{C.RESET}"
                  f"  |  New income: ₹{sw['new_income']:,.0f}")
        else:
            info(f"Current crop ({farmer['crop_type']}) is already optimal")

    # ── DEMO 6: INSURANCE PREMIUM
    header("FEATURE 11: CROP INSURANCE PREMIUM CALCULATOR")
    section("PMFBY Insurance Calculation")
    for farmer in FARMERS:
        inc  = predict_income(farmer)
        prem = insurance_premium(farmer, inc["predicted_income"])
        print(f"\n  {C.BOLD}{farmer['name']}{C.RESET}")
        ok("Sum Insured",      f"₹{prem['sum_insured']:,.0f}")
        ok("Farmer Pays",      f"₹{prem['farmer_premium']:,.0f}/year")
        ok("Govt Subsidy",     f"₹{prem['govt_subsidy']:,.0f}/year")
        info(f"Scheme: {prem['scheme']}")

    # ── DEMO 7: PORTFOLIO ANALYTICS
    header("FEATURE 9: PORTFOLIO ANALYTICS")
    section("Branch Portfolio — Jorhat Branch (5 sample loans)")
    portfolio = [
        ("Ramesh Kumar",  "paddy",      95000,   "Jorhat"),
        ("Meena Borah",   "paddy",      88000,   "Jorhat"),
        ("Akhil Das",     "mustard",    128000,  "Jorhat"),
        ("Puja Sharma",   "paddy",      76000,   "Jorhat"),
        ("Rajiv Gogoi",   "vegetables", 168000,  "Jorhat"),
    ]
    from collections import Counter
    crops  = Counter(p[1] for p in portfolio)
    total  = sum(p[2] for p in portfolio)
    avg    = total / len(portfolio)
    top_c  = crops.most_common(1)[0]
    top_pct = top_c[1] / len(portfolio) * 100

    ok("Total Farmers",      len(portfolio))
    ok("Total Exposure",     f"₹{total:,.0f}")
    ok("Average Income",     f"₹{avg:,.0f}")
    ok("Top Crop",           f"{top_c[0].capitalize()} ({top_pct:.0f}% of portfolio)")

    if top_pct > 60:
        warn(f"HIGH CONCENTRATION: {top_pct:.0f}% of loans are in {top_c[0]}")
        warn("Recommendation: Pause new paddy loans this season")
    else:
        ok("Concentration Risk", "LOW — portfolio diversified")

    # ── DEMO 8: SEASON COMPARISON
    header("FEATURE 10: MULTI-SEASON INCOME TRACKING")
    section("Ramesh Kumar — Income Trend (3 seasons)")
    seasons_data = [
        ("Kharif 2023", 88000),
        ("Rabi 2024",   95000),
        ("Kharif 2024", 121000),
    ]
    for season, income in seasons_data:
        bar_len = int(income / 5000)
        bar     = "█" * min(bar_len, 30)
        print(f"  {season:<15} ₹{income:>8,.0f}  {C.CYAN}{bar}{C.RESET}")

    growth = (seasons_data[-1][1] - seasons_data[0][1]) / seasons_data[0][1] * 100
    ok("Income Growth",   f"+{growth:.1f}%", " over 3 seasons")
    ok("Trend",           "GROWING")
    ok("Recommendation",  "Eligible for KCC limit enhancement")

    # ── WHATSAPP BOT DEMO
    header("FEATURE 15: WHATSAPP BOT (Sample Responses)")
    section("Simulating farmer WhatsApp messages")
    commands = [
        ("Hello",            "AgriIntel Farmer Assistant — send 'help' for commands"),
        ("score",            "Visit bank branch for personalized AgriScore report"),
        ("schemes",          "You may be eligible for PM-KISAN, KCC, PMFBY..."),
        ("loan",             "Visit branch with land records, Aadhaar, crop photo"),
        ("crop",             "Mustard yields 30-80% higher income than paddy in NE India"),
    ]
    for msg, response in commands:
        print(f"\n  {C.YELLOW}Farmer:{C.RESET} {msg}")
        print(f"  {C.CYAN}AgriIntel Bot:{C.RESET} {response}")

    # ── SUMMARY
    header("SUMMARY — What AgriIntel Does")
    print(f"""
  {C.BOLD}For Bank Officers:{C.RESET}
  {C.GREEN}✓{C.RESET}  Enter 5 farmer details → Get full credit assessment in seconds
  {C.GREEN}✓{C.RESET}  Upload 500 farmers in Excel → Download predictions Excel
  {C.GREEN}✓{C.RESET}  Get NPA early warnings before bad loans happen
  {C.GREEN}✓{C.RESET}  Download one-page PDF report per farmer
  {C.GREEN}✓{C.RESET}  See full branch portfolio health dashboard

  {C.BOLD}For Farmers (via WhatsApp):{C.RESET}
  {C.GREEN}✓{C.RESET}  Check loan status by sending WhatsApp message
  {C.GREEN}✓{C.RESET}  Get matched to government schemes automatically
  {C.GREEN}✓{C.RESET}  Receive crop switch recommendations

  {C.BOLD}The Numbers:{C.RESET}
  {C.GREEN}✓{C.RESET}  24 API endpoints ready to deploy
  {C.GREEN}✓{C.RESET}  17 features built and working
  {C.GREEN}✓{C.RESET}  ₹800 total cost to launch
  {C.GREEN}✓{C.RESET}  ₹550 Crore addressable market
  {C.GREEN}✓{C.RESET}  0 direct competitors

  {C.CYAN}{'─'*50}{C.RESET}
  {C.BOLD}  PDF saved: examples/demo_farmer_health_card.pdf{C.RESET}
  {C.BOLD}  To run the full API: uvicorn main:app --reload{C.RESET}
  {C.BOLD}  API docs: http://localhost:8000/docs{C.RESET}
  {C.CYAN}{'─'*50}{C.RESET}
""")

if __name__ == "__main__":
    run_demo()
