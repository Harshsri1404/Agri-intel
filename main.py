"""
AgriIntel — Complete FastAPI Backend
=====================================
ALL FEATURES:
 1. Income Prediction
 2. AgriScore (0-100 credit score)
 3. Repayment Probability
 4. Loan Amount Recommender
 5. Full Combined Prediction (all 4 above in one call)
 6. Batch Upload (Excel → predictions Excel)
 7. NPA Early Warning System
 8. Farmer Financial Health Card (PDF)
 9. Portfolio Analytics
10. Multi-Season Income Tracking
11. Crop Switch Recommender
12. Government Scheme Matcher
13. Insurance Premium Calculator
14. Upsell Opportunity Identifier
15. WhatsApp Bot Webhook
16. Health Check

RUN:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

DOCS:
    http://localhost:8000/docs
"""

import os
import io
import json
import logging
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, date
from typing import Optional, List
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv
    load_dotenv()  # load .env BEFORE utils/* read env vars
except ImportError:
    pass

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.pdf_generator import generate_health_card
from utils.schemes import match_schemes, get_crop_switch_recommendation
from utils.weather import get_district_npa_risk

# ─── LOGGING ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agriintel")

# ─── GLOBALS ────────────────────────────────────────────────────────
ARTIFACTS = {}
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "artifacts", "agriintel_artifacts.pkl")

def load_model():
    """Load model artifacts. Falls back to mock if not found."""
    global ARTIFACTS
    if os.path.exists(MODEL_PATH):
        ARTIFACTS = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded. MAPE: {ARTIFACTS.get('mape', 'unknown')}")
    else:
        logger.warning("Model file not found — using mock predictions for demo.")
        ARTIFACTS = {"mock": True}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

# ─── APP ────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgriIntel API",
    description="AI-Powered Farmer Income Prediction & Credit Intelligence Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS: set ALLOWED_ORIGINS="https://your-frontend.com,https://other.com" in production.
# Default "*" is for local development / demos only.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=(_origins != ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── SCHEMAS ────────────────────────────────────────────────────────
class FarmerInput(BaseModel):
    name:           str  = Field(..., example="Ramesh Kumar")
    phone:          str  = Field("", example="9876543210")
    crop_type:      str  = Field(..., example="paddy")
    land_size:      float= Field(..., gt=0, example=2.5)
    district:       str  = Field(..., example="Jorhat")
    state:          str  = Field("Assam", example="Assam")
    irrigation_type:str  = Field(..., example="canal")
    ownership:      str  = Field(..., example="owned")
    season:         str  = Field("kharif", example="kharif")
    # Optional enrichment fields
    soil_ph:        Optional[float] = None
    nitrogen:       Optional[float] = None
    phosphorus:     Optional[float] = None
    potassium:      Optional[float] = None
    annual_rainfall:Optional[float] = None
    branch_id:      Optional[str]   = None
    officer_name:   Optional[str]   = None
    branch_name:    Optional[str]   = None


class LoanRecord(BaseModel):
    farmer_name:    str
    farmer_id:      str
    district:       str
    state:          str
    crop_type:      str
    loan_amount:    float
    disbursed_date: str
    branch_id:      str


class PortfolioRequest(BaseModel):
    branch_id: str
    loans:     List[LoanRecord]


class WhatsAppMessage(BaseModel):
    object: str
    entry:  list


# ─── PREDICTION ENGINE ──────────────────────────────────────────────
def preprocess_input(farmer: FarmerInput) -> pd.DataFrame:
    """Convert FarmerInput to model-ready DataFrame."""
    d = farmer.dict()

    # Fill optional enrichment fields with neutral defaults (None breaks the model)
    for k, default in {"annual_rainfall": 1000.0, "soil_ph": 6.5, "nitrogen": 50.0,
                       "phosphorus": 50.0, "potassium": 50.0}.items():
        if d.get(k) is None:
            d[k] = default

    # Irrigation score
    irr_map = {"drip":1.0,"sprinkler":0.85,"canal":0.7,
               "borewell":0.75,"rainfed":0.4,"other":0.5}
    d["irrigation_score"] = irr_map.get(d["irrigation_type"].lower(), 0.5)

    # Ownership score
    own_map = {"owned":1.0,"leased":0.6,"rented":0.5,"share":0.4}
    d["ownership_score"] = own_map.get(d["ownership"].lower(), 0.5)

    # Crop risk
    low_risk  = ["wheat","rice","paddy","maize"]
    high_risk = ["cotton","sugarcane","tobacco","jute"]
    crop = d["crop_type"].lower()
    d["crop_risk"] = 0 if crop in low_risk else 2 if crop in high_risk else 1

    # Season code
    season_map = {"kharif":1,"rabi":2,"zaid":3,"summer":3,"winter":2,"monsoon":1}
    d["season_code"] = season_map.get(d["season"].lower(), 1)

    # Land category
    ls = d["land_size"]
    d["land_category"] = 0 if ls<0.5 else 1 if ls<1 else 2 if ls<2 else 3 if ls<5 else 4 if ls<10 else 5

    # NPK composite
    if all(d.get(k) for k in ["nitrogen","phosphorus","potassium"]):
        d["npk_composite"] = d["nitrogen"]*0.4 + d["phosphorus"]*0.3 + d["potassium"]*0.3
    else:
        d["npk_composite"] = 50.0  # Default median

    # pH optimal
    ph = d.get("soil_ph") or 6.5
    d["ph_optimal"] = 1 - abs(ph - 6.5) / 6.5

    # Rainfall interaction
    rain = d.get("annual_rainfall") or 1000
    d["land_rain_interaction"] = d["land_size"] * rain

    df = pd.DataFrame([d])
    return df


def predict_income(farmer: FarmerInput) -> dict:
    """Core income prediction."""
    if ARTIFACTS.get("mock") or "feature_names" not in ARTIFACTS:
        # Demo mode: rule-based estimate
        base = farmer.land_size * 40000
        irr_boost = {"drip":1.3,"canal":1.1,"borewell":1.15,"rainfed":0.85}.get(
            farmer.irrigation_type.lower(), 1.0)
        crop_boost = {"vegetables":1.8,"onion":2.0,"cotton":1.4,"paddy":1.0,"wheat":1.1}.get(
            farmer.crop_type.lower(), 1.0)
        income = base * irr_boost * crop_boost * (1.1 if farmer.ownership.lower()=="owned" else 0.9)
        income = max(30000, min(income, 500000))
        return {"predicted_income": round(income, -2),
                "range_low": round(income*0.85, -2),
                "range_high": round(income*1.15, -2),
                "model": "demo"}

    df = preprocess_input(farmer)
    arts = ARTIFACTS

    # Align features
    feat_names = arts["feature_names"]
    encoders   = arts["encoders"]
    for col in feat_names:
        if col not in df.columns:
            df[col] = 0
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    X = df[feat_names].apply(pd.to_numeric, errors="coerce").fillna(0)
    pred_log = arts["model"].predict(X)[0]
    income   = float(np.expm1(pred_log))
    income   = max(10000, income)

    return {
        "predicted_income": round(income, -2),
        "range_low":        round(income * 0.85, -2),
        "range_high":       round(income * 1.15, -2),
        "model":            "ensemble"
    }


def compute_agri_score(farmer: FarmerInput, income: float) -> dict:
    """Compute 0-100 AgriScore."""
    score = 0

    # Income band (40 pts)
    if income > 200000: score += 40
    elif income > 150000: score += 33
    elif income > 100000: score += 26
    elif income > 75000:  score += 18
    elif income > 50000:  score += 12
    else:                 score += 6

    # Irrigation (20 pts)
    irr_pts = {"drip":20,"sprinkler":17,"borewell":14,"canal":12,"rainfed":6}
    score += irr_pts.get(farmer.irrigation_type.lower(), 8)

    # Ownership (20 pts)
    own_pts = {"owned":20,"leased":12,"rented":10,"share":7}
    score += own_pts.get(farmer.ownership.lower(), 8)

    # Land size (20 pts)
    ls = farmer.land_size
    if ls > 10: score += 20
    elif ls > 5: score += 16
    elif ls > 2: score += 12
    elif ls > 1: score += 8
    else:        score += 4

    score = min(100, max(0, score))

    if score >= 70:   decision = "APPROVE"
    elif score >= 50: decision = "REVIEW"
    else:             decision = "REJECT"

    return {"agri_score": score, "decision": decision}


def compute_repayment_prob(farmer: FarmerInput, income: float,
                            agri_score: int) -> float:
    """Repayment score (0-100).

    NOTE: this is a heuristic score, not a calibrated default probability.
    The trained model (if present) is fit on a proxy label, not real repayment
    history. Replace with a model trained on actual loan outcomes before any
    real lending use.
    """
    if ARTIFACTS.get("mock") or "repay_model" not in ARTIFACTS or "feature_names" not in ARTIFACTS:
        # Rule-based estimate
        base = 50
        base += (agri_score - 50) * 0.4
        if farmer.ownership.lower() == "owned": base += 10
        if farmer.irrigation_type.lower() == "drip": base += 8
        if income > 150000: base += 10
        return round(min(99, max(30, base)), 1)

    df = preprocess_input(farmer)
    arts = ARTIFACTS
    feat_names = arts["feature_names"]
    encoders   = arts["encoders"]
    for col in feat_names:
        if col not in df.columns:
            df[col] = 0
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])
    X = df[feat_names].apply(pd.to_numeric, errors="coerce").fillna(0)
    prob = arts["repay_model"].predict_proba(X)[0][1]
    return round(float(prob) * 100, 1)


def recommend_loan(income: float, agri_score: int) -> dict:
    """Recommend loan amount based on income and score."""
    if agri_score >= 70:      multiplier = 0.65; tenure = 36
    elif agri_score >= 60:    multiplier = 0.55; tenure = 24
    elif agri_score >= 50:    multiplier = 0.40; tenure = 24
    else:                     multiplier = 0.20; tenure = 12

    loan_amount = round(income * multiplier, -3)
    interest_rate = 7.0  # KCC standard
    monthly_rate  = interest_rate / 100 / 12
    if monthly_rate > 0:
        emi = loan_amount * monthly_rate * (1+monthly_rate)**tenure / \
              ((1+monthly_rate)**tenure - 1)
    else:
        emi = loan_amount / tenure

    return {
        "recommended_loan": loan_amount,
        "monthly_emi":      round(emi, 2),
        "tenure_months":    tenure,
        "interest_rate":    interest_rate,
        "multiplier_used":  multiplier
    }


def get_district_average_income(district: str) -> float:
    """Approximate district average income (can be replaced with real data)."""
    DISTRICT_AVERAGES = {
        "jorhat": 95000, "kamrup": 105000, "dibrugarh": 88000,
        "golaghat": 92000, "nagaon": 82000, "tinsukia": 90000,
        "default": 90000
    }
    return DISTRICT_AVERAGES.get(district.lower(), DISTRICT_AVERAGES["default"])


# ─── ROUTES ─────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    return {
        "name": "AgriIntel API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": not ARTIFACTS.get("mock", False),
        "docs": "/docs"
    }


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "model":  "loaded" if not ARTIFACTS.get("mock") else "demo_mode",
        "mape":   f"{ARTIFACTS.get('mape', 0)*100:.1f}%" if not ARTIFACTS.get("mock") else "N/A",
        "timestamp": datetime.now().isoformat()
    }


# ── 1. INCOME PREDICTION ────────────────────────────────────────────
@app.post("/predict/income", tags=["Prediction"])
def predict_income_endpoint(farmer: FarmerInput):
    """Predict annual income for a farmer."""
    result = predict_income(farmer)
    return result


# ── 2. AGRISCORE ────────────────────────────────────────────────────
@app.post("/predict/score", tags=["Prediction"])
def predict_score(farmer: FarmerInput):
    """Generate AgriScore (0-100) and loan decision."""
    income_data = predict_income(farmer)
    score_data  = compute_agri_score(farmer, income_data["predicted_income"])
    return {**income_data, **score_data}


# ── 3. REPAYMENT PROBABILITY ────────────────────────────────────────
@app.post("/predict/repayment", tags=["Prediction"])
def predict_repayment(farmer: FarmerInput):
    """Predict repayment probability."""
    income_data = predict_income(farmer)
    score_data  = compute_agri_score(farmer, income_data["predicted_income"])
    repay_prob  = compute_repayment_prob(
        farmer, income_data["predicted_income"], score_data["agri_score"])

    risk = "LOW" if repay_prob >= 75 else "MEDIUM" if repay_prob >= 55 else "HIGH"
    return {
        **income_data, **score_data,
        "repayment_probability": repay_prob,
        "risk_category": risk
    }


# ── 4. FULL PREDICTION (all features in one call) ───────────────────
@app.post("/predict/full", tags=["Prediction"])
def predict_full(farmer: FarmerInput):
    """
    Complete farmer credit assessment — all predictions in one API call.
    Returns income, AgriScore, repayment probability, loan recommendation,
    crop switch suggestion, matched schemes, and income vs district comparison.
    """
    # Core predictions
    income_data  = predict_income(farmer)
    income       = income_data["predicted_income"]
    score_data   = compute_agri_score(farmer, income)
    repay_prob   = compute_repayment_prob(farmer, income, score_data["agri_score"])
    loan_data    = recommend_loan(income, score_data["agri_score"])

    # District comparison
    district_avg    = get_district_average_income(farmer.district)
    vs_district_pct = round((income - district_avg) / district_avg * 100, 1)

    # Crop switch
    crop_switch = get_crop_switch_recommendation(farmer.dict(), income)

    # Government schemes
    schemes = match_schemes(farmer.dict())

    # Risk category
    repay_risk = "LOW" if repay_prob >= 75 else "MEDIUM" if repay_prob >= 55 else "HIGH"

    # Build recommendations
    recommendations = []
    if farmer.irrigation_type.lower() == "rainfed":
        recommendations.append("Install drip irrigation — can increase income by 20-30%")
    if farmer.ownership.lower() == "rented":
        recommendations.append("Consider purchasing land under PM-KISAN + KCC scheme")
    if vs_district_pct < -10:
        recommendations.append(
            f"Income is {abs(vs_district_pct):.0f}% below district average — consider crop diversification")
    if score_data["agri_score"] >= 70:
        recommendations.append("Eligible for KCC limit enhancement — strong credit profile")
    if len(schemes) > 0:
        recommendations.append(f"Eligible for {len(schemes)} government schemes — see schemes section")

    return {
        # Income
        "predicted_income":      income,
        "income_range_low":      income_data["range_low"],
        "income_range_high":     income_data["range_high"],
        "income_vs_district_pct": vs_district_pct,
        "district_avg_income":   district_avg,
        # Score
        "agri_score":            score_data["agri_score"],
        "decision":              score_data["decision"],
        # Repayment
        "repayment_probability": repay_prob,
        "risk_category":         repay_risk,
        # Loan
        "recommended_loan":      loan_data["recommended_loan"],
        "monthly_emi":           loan_data["monthly_emi"],
        "tenure_months":         loan_data["tenure_months"],
        "interest_rate":         loan_data["interest_rate"],
        # Enrichment
        "crop_switch_suggestion": crop_switch,
        "matched_schemes":        schemes,
        "recommendations":        recommendations,
        # Meta
        "farmer_name":           farmer.name,
        "generated_at":          datetime.now().isoformat(),
        "model":                 income_data["model"]
    }


# ── 5. LOAN RECOMMENDER ─────────────────────────────────────────────
@app.post("/predict/loan-amount", tags=["Prediction"])
def predict_loan_amount(farmer: FarmerInput):
    """Recommend safe loan amount and EMI."""
    income_data = predict_income(farmer)
    score_data  = compute_agri_score(farmer, income_data["predicted_income"])
    loan_data   = recommend_loan(income_data["predicted_income"], score_data["agri_score"])
    return {**income_data, **score_data, **loan_data}


# ── 6. BATCH UPLOAD ─────────────────────────────────────────────────
@app.post("/predict/batch", tags=["Batch"])
async def batch_predict(file: UploadFile = File(...)):
    """
    Batch prediction from Excel upload.
    Required columns: name, crop_type, land_size, district, state,
                      irrigation_type, ownership, phone (optional)
    Returns downloadable Excel with all predictions.
    """
    if not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, "Only .xlsx, .xls, or .csv files accepted")

    contents = await file.read()

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {str(e)}")

    required = ["crop_type", "land_size", "district", "irrigation_type", "ownership"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}. "
                                 f"Required: {required}")

    results = []
    for i, row in df.iterrows():
        try:
            farmer = FarmerInput(
                name           = str(row.get("name", f"Farmer {i+1}")),
                phone          = str(row.get("phone", "")),
                crop_type      = str(row["crop_type"]),
                land_size      = float(row["land_size"]),
                district       = str(row["district"]),
                state          = str(row.get("state", "India")),
                irrigation_type= str(row["irrigation_type"]),
                ownership      = str(row["ownership"]),
                season         = str(row.get("season", "kharif")),
            )
            pred = predict_full(farmer)
            results.append({
                "Farmer Name":         pred["farmer_name"],
                "Predicted Income (₹)":pred["predicted_income"],
                "Income Range":        f"₹{pred['income_range_low']:,.0f} – ₹{pred['income_range_high']:,.0f}",
                "vs District (%)":     pred["income_vs_district_pct"],
                "AgriScore":           pred["agri_score"],
                "Decision":            pred["decision"],
                "Recommended Loan (₹)":pred["recommended_loan"],
                "Monthly EMI (₹)":     pred["monthly_emi"],
                "Repayment Prob (%)":  pred["repayment_probability"],
                "Risk Category":       pred["risk_category"],
                "Crop Switch":         pred["crop_switch_suggestion"]["crop"]
                                       if pred["crop_switch_suggestion"] else "—",
                "Schemes Available":   len(pred["matched_schemes"]),
            })
        except Exception as e:
            results.append({
                "Farmer Name": row.get("name", f"Row {i+1}"),
                "Error": str(e)
            })

    result_df = pd.DataFrame(results)

    # Apply color coding in Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="AgriIntel Predictions")
        ws = writer.sheets["AgriIntel Predictions"]

        # Header styling
        from openpyxl.styles import PatternFill, Font, Alignment
        header_fill = PatternFill("solid", fgColor="0D1B2A")
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = Font(color="00A8CC", bold=True)
            cell.alignment = Alignment(horizontal="center")

        # Decision color coding
        decision_col = None
        for col_idx, cell in enumerate(ws[1], 1):
            if cell.value == "Decision":
                decision_col = col_idx
                break

        if decision_col:
            green_fill  = PatternFill("solid", fgColor="E8F8F5")
            yellow_fill = PatternFill("solid", fgColor="FEF9E7")
            red_fill    = PatternFill("solid", fgColor="FDEDEC")
            for row in ws.iter_rows(min_row=2, min_col=decision_col,
                                    max_col=decision_col):
                for cell in row:
                    if cell.value == "APPROVE": cell.fill = green_fill
                    elif cell.value == "REVIEW": cell.fill = yellow_fill
                    elif cell.value == "REJECT": cell.fill = red_fill

        # Auto-width columns
        for column in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in column)
            ws.column_dimensions[column[0].column_letter].width = min(max_len + 2, 30)

    output.seek(0)
    filename = f"agriintel_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/predict/batch/template", tags=["Batch"])
def download_batch_template():
    """Download Excel template for batch upload."""
    template_data = {
        "name":            ["Ramesh Kumar", "Sunita Devi"],
        "phone":           ["9876543210", "9123456789"],
        "crop_type":       ["paddy", "wheat"],
        "land_size":       [2.5, 1.8],
        "district":        ["Jorhat", "Kamrup"],
        "state":           ["Assam", "Assam"],
        "irrigation_type": ["canal", "drip"],
        "ownership":       ["owned", "rented"],
        "season":          ["kharif", "rabi"],
    }
    df = pd.DataFrame(template_data)
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=agriintel_template.xlsx"}
    )


# ── 7. NPA EARLY WARNING ────────────────────────────────────────────
@app.get("/alerts/npa-warning", tags=["Alerts"])
def npa_warning(
    district:  str = Query(..., example="Jorhat"),
    state:     str = Query("Assam", example="Assam"),
    crop_type: str = Query("paddy", example="paddy")
):
    """
    Check NPA risk for a district based on current weather conditions.
    Use this to proactively alert bank officers about at-risk loans.
    """
    risk = get_district_npa_risk(district, crop_type, state)
    return {
        "district":   district,
        "crop_type":  crop_type,
        "npa_risk":   risk,
        "checked_at": datetime.now().isoformat()
    }


@app.post("/alerts/portfolio-npa-scan", tags=["Alerts"])
def portfolio_npa_scan(request: PortfolioRequest):
    """
    Scan entire branch loan portfolio for NPA risk.
    Returns prioritized list of at-risk farmers.
    """
    warnings = []
    district_cache = {}

    for loan in request.loans:
        cache_key = f"{loan.district}_{loan.crop_type}"
        if cache_key not in district_cache:
            district_cache[cache_key] = get_district_npa_risk(
                loan.district, loan.crop_type, loan.state)

        risk = district_cache[cache_key]
        if risk["risk_level"] in ["HIGH", "MEDIUM"]:
            warnings.append({
                "farmer_name":  loan.farmer_name,
                "farmer_id":    loan.farmer_id,
                "loan_amount":  loan.loan_amount,
                "district":     loan.district,
                "crop_type":    loan.crop_type,
                "risk_level":   risk["risk_level"],
                "risk_score":   risk["risk_score"],
                "reasons":      risk["reasons"],
                "action":       risk["action"]
            })

    # Sort by risk score descending
    warnings.sort(key=lambda x: x["risk_score"], reverse=True)

    high_count   = sum(1 for w in warnings if w["risk_level"] == "HIGH")
    medium_count = sum(1 for w in warnings if w["risk_level"] == "MEDIUM")

    return {
        "branch_id":       request.branch_id,
        "total_loans":     len(request.loans),
        "at_risk":         len(warnings),
        "high_risk_count": high_count,
        "medium_risk_count": medium_count,
        "warnings":        warnings,
        "scanned_at":      datetime.now().isoformat()
    }


# ── 8. FARMER HEALTH CARD PDF ───────────────────────────────────────
@app.post("/report/farmer-card", tags=["Reports"])
def generate_farmer_card(farmer: FarmerInput):
    """
    Generate a downloadable PDF Farmer Financial Health Card.
    Contains income prediction, AgriScore, loan recommendation,
    government schemes, and crop switch suggestions.
    """
    pred = predict_full(farmer)

    farmer_data = {
        **farmer.dict(),
        **pred,
        "generated_by": farmer.officer_name or "Bank Officer",
        "branch_name":  farmer.branch_name or "—",
    }

    try:
        pdf_bytes = generate_health_card(farmer_data)
        filename  = f"AgriIntel_{farmer.name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")


# ── 9. PORTFOLIO ANALYTICS ──────────────────────────────────────────
@app.post("/analytics/portfolio", tags=["Analytics"])
def portfolio_analytics(request: PortfolioRequest):
    """
    Branch portfolio health analytics.
    Analyzes crop concentration, risk distribution, income trends.
    """
    if not request.loans:
        raise HTTPException(400, "No loans provided")

    df = pd.DataFrame([l.dict() for l in request.loans])

    # Crop concentration
    crop_dist = df["crop_type"].value_counts()
    top_crop  = crop_dist.index[0]
    top_pct   = round(crop_dist.iloc[0] / len(df) * 100, 1)

    # District concentration
    dist_dist = df["district"].value_counts()
    top_dist  = dist_dist.index[0]
    top_dist_pct = round(dist_dist.iloc[0] / len(df) * 100, 1)

    # Loan amount stats
    total_exposure = df["loan_amount"].sum()
    avg_loan       = df["loan_amount"].mean()
    max_loan       = df["loan_amount"].max()

    # Concentration risk
    concentration_risk = "HIGH" if top_pct > 60 else "MEDIUM" if top_pct > 40 else "LOW"

    alerts = []
    if top_pct > 60:
        alerts.append({
            "type":    "CROP_CONCENTRATION",
            "message": f"{top_pct}% of loans are in {top_crop} — high concentration risk",
            "action":  "Pause new loans for this crop until diversification improves"
        })
    if top_dist_pct > 70:
        alerts.append({
            "type":    "GEOGRAPHIC_CONCENTRATION",
            "message": f"{top_dist_pct}% of loans in {top_dist} district",
            "action":  "Consider expanding to neighboring districts"
        })

    return {
        "branch_id":            request.branch_id,
        "total_loans":          len(df),
        "total_exposure_inr":   round(total_exposure, 2),
        "average_loan_inr":     round(avg_loan, 2),
        "max_loan_inr":         max_loan,
        "crop_distribution":    crop_dist.to_dict(),
        "district_distribution":dist_dist.to_dict(),
        "top_crop":             top_crop,
        "top_crop_pct":         top_pct,
        "concentration_risk":   concentration_risk,
        "alerts":               alerts,
        "analyzed_at":          datetime.now().isoformat()
    }


# ── 10. UPSELL OPPORTUNITY IDENTIFIER ──────────────────────────────
@app.post("/analytics/upsell-opportunities", tags=["Analytics"])
def upsell_opportunities(
    current_assessments: List[dict],
    previous_assessments: List[dict]
):
    """
    Identify farmers whose income has grown — candidates for loan enhancement.
    Compare current vs previous season predictions.
    """
    opportunities = []

    curr_map = {a.get("farmer_id") or a.get("name"): a for a in current_assessments}
    prev_map = {a.get("farmer_id") or a.get("name"): a for a in previous_assessments}

    for key, curr in curr_map.items():
        if key in prev_map:
            prev = prev_map[key]
            curr_income = curr.get("predicted_income", 0)
            prev_income = prev.get("predicted_income", 0)
            if prev_income > 0:
                growth_pct = (curr_income - prev_income) / prev_income * 100
                if growth_pct >= 15:
                    curr_loan  = curr.get("recommended_loan", 0)
                    prev_loan  = prev.get("recommended_loan", 0)
                    loan_uplift = curr_loan - prev_loan
                    if loan_uplift > 0:
                        opportunities.append({
                            "farmer":          key,
                            "income_growth_pct": round(growth_pct, 1),
                            "prev_income":     prev_income,
                            "curr_income":     curr_income,
                            "prev_loan":       prev_loan,
                            "enhanced_loan":   curr_loan,
                            "loan_uplift":     loan_uplift,
                            "recommendation":  f"Offer KCC limit enhancement of ₹{loan_uplift:,.0f}"
                        })

    opportunities.sort(key=lambda x: x["income_growth_pct"], reverse=True)
    return {
        "total_opportunities": len(opportunities),
        "total_additional_loan_possible": sum(o["loan_uplift"] for o in opportunities),
        "opportunities": opportunities
    }


# ── 11. INSURANCE PREMIUM CALCULATOR ────────────────────────────────
@app.post("/tools/insurance-premium", tags=["Tools"])
def insurance_premium(farmer: FarmerInput):
    """
    Calculate recommended crop insurance premium under PMFBY.
    Banks use this to cross-sell insurance during loan processing.
    """
    income_data = predict_income(farmer)
    income      = income_data["predicted_income"]

    # PMFBY: farmer pays 2% (kharif), 1.5% (rabi), 5% (horticulture)
    season = farmer.season.lower()
    farmer_rate = 0.02 if season == "kharif" else 0.015 if season == "rabi" else 0.05

    crop_type = farmer.crop_type.lower()
    # Sum insured = 1.5x expected income (standard)
    sum_insured = income * 1.5

    farmer_premium   = round(sum_insured * farmer_rate, 2)
    govt_subsidy     = round(sum_insured * 0.06, 2)  # Approx govt share
    total_premium    = round(farmer_premium + govt_subsidy, 2)

    # Risk assessment for premium adjustment
    is_rainfed = farmer.irrigation_type.lower() == "rainfed"
    risk_notes = []
    if is_rainfed:
        farmer_premium *= 1.1
        risk_notes.append("Rainfed farming — slight premium adjustment applied")

    return {
        "farmer_name":       farmer.name,
        "crop_type":         farmer.crop_type,
        "season":            farmer.season,
        "sum_insured":       round(sum_insured, -2),
        "farmer_premium":    round(farmer_premium, 2),
        "government_subsidy":govt_subsidy,
        "total_premium":     round(total_premium, 2),
        "scheme":            "PM Fasal Bima Yojana (PMFBY)",
        "coverage":          "Full crop loss, partial damage, post-harvest losses",
        "risk_notes":        risk_notes,
        "predicted_income":  income
    }


# ── 12. GOVERNMENT SCHEMES ──────────────────────────────────────────
@app.post("/tools/schemes", tags=["Tools"])
def get_schemes(farmer: FarmerInput):
    """Match farmer to all eligible government schemes."""
    schemes  = match_schemes(farmer.dict())
    income   = predict_income(farmer)["predicted_income"]
    crop_sw  = get_crop_switch_recommendation(farmer.dict(), income)

    return {
        "farmer_name":        farmer.name,
        "matched_schemes":    schemes,
        "schemes_count":      len(schemes),
        "crop_switch":        crop_sw,
        "predicted_income":   income
    }


# ── 13. CROP SWITCH RECOMMENDER ─────────────────────────────────────
@app.post("/tools/crop-switch", tags=["Tools"])
def crop_switch_recommendation(farmer: FarmerInput):
    """Recommend best crop switch for income maximization."""
    income_data = predict_income(farmer)
    income      = income_data["predicted_income"]
    suggestion  = get_crop_switch_recommendation(farmer.dict(), income)

    if not suggestion:
        return {
            "farmer_name":     farmer.name,
            "current_crop":    farmer.crop_type,
            "current_income":  income,
            "recommendation":  "Current crop is already optimal for your profile",
            "switch_suggested": False
        }

    return {
        "farmer_name":        farmer.name,
        "current_crop":       farmer.crop_type,
        "current_income":     income,
        "recommended_crop":   suggestion["crop"],
        "estimated_income":   suggestion["estimated_income"],
        "income_uplift":      suggestion["uplift"],
        "uplift_pct":         round(suggestion["uplift"]/income*100, 1),
        "requires_irrigation":suggestion["requires_irrigation"],
        "switch_suggested":   True
    }


# ── 14. WHATSAPP BOT WEBHOOK ────────────────────────────────────────
WHATSAPP_TOKEN  = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_VERIFY = os.getenv("WHATSAPP_VERIFY_TOKEN", "")  # required; no default on purpose
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")


def send_whatsapp_message(to: str, message: str):
    """Send WhatsApp message via Meta Business API."""
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        logger.warning("WhatsApp credentials not set")
        return
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}",
               "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    try:
        requests.post(url, headers=headers, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"WhatsApp send failed: {e}")


@app.get("/webhook/whatsapp", tags=["WhatsApp"])
def whatsapp_verify(
    hub_mode:         str = Query(None, alias="hub.mode"),
    hub_challenge:    str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token")
):
    """WhatsApp webhook verification (Meta requires this)."""
    if WHATSAPP_VERIFY and hub_mode == "subscribe" and hub_verify_token == WHATSAPP_VERIFY:
        return int(hub_challenge)
    raise HTTPException(403, "Verification failed")


@app.post("/webhook/whatsapp", tags=["WhatsApp"])
async def whatsapp_webhook(request_body: dict):
    """
    Handle incoming WhatsApp messages from farmers.
    Commands:
    - 'score'       → Get their AgriScore
    - 'loan'        → Get loan status
    - 'income'      → Get predicted income
    - 'schemes'     → Get eligible schemes
    - 'help'        → List commands
    """
    try:
        entry   = request_body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value   = changes.get("value", {})
        messages = value.get("messages", [])

        if not messages:
            return {"status": "no_message"}

        msg      = messages[0]
        phone    = msg.get("from", "")
        text     = msg.get("text", {}).get("body", "").lower().strip()
        msg_type = msg.get("type", "")

        if msg_type != "text":
            send_whatsapp_message(phone,
                "🌾 *AgriIntel*\nSend a text message:\n"
                "• *score* — Your AgriScore\n"
                "• *income* — Your income estimate\n"
                "• *schemes* — Govt schemes for you\n"
                "• *help* — All commands")
            return {"status": "non_text"}

        # Route commands
        if "help" in text or "hello" in text or "hi" in text:
            reply = (
                "🌾 *AgriIntel Farmer Assistant*\n\n"
                "Available commands:\n"
                "• *score* — Your AgriScore\n"
                "• *income* — Your income estimate\n"
                "• *loan* — Loan eligibility\n"
                "• *schemes* — Government schemes\n"
                "• *crop* — Crop switch advice\n\n"
                "For loan applications, visit your nearest bank branch."
            )

        elif "score" in text:
            reply = (
                "🏆 *Your AgriScore*\n\n"
                "To get your AgriScore, please visit your bank branch "
                "with your land records and crop details.\n\n"
                "Your bank officer will generate a full report in seconds using AgriIntel."
            )

        elif "income" in text:
            reply = (
                "💰 *Income Estimation*\n\n"
                "Your income estimate depends on your crop, land size, "
                "and irrigation type.\n\n"
                "Visit your bank officer for a personalized AgriIntel income report — it's free!"
            )

        elif "loan" in text:
            reply = (
                "🏦 *Loan Information*\n\n"
                "For KCC loan status, contact your bank branch directly.\n\n"
                "For a new loan assessment, visit your nearest branch with:\n"
                "• Land records (7/12)\n"
                "• Aadhaar card\n"
                "• Recent crop photo"
            )

        elif "scheme" in text or "yojana" in text or "subsidy" in text:
            reply = (
                "📋 *Government Schemes for Farmers*\n\n"
                "You may be eligible for:\n"
                "✓ PM-KISAN — ₹6,000/year\n"
                "✓ KCC — 4% interest crop loan\n"
                "✓ PMFBY — Crop insurance\n"
                "✓ PM Krishi Sinchayee — Drip irrigation subsidy\n\n"
                "Visit pmkisan.gov.in or your bank for applications."
            )

        elif "crop" in text:
            reply = (
                "🌾 *Crop Advice*\n\n"
                "For personalized crop switch recommendations based on your soil "
                "and location, visit your bank officer for a full AgriIntel report.\n\n"
                "General tip: Mustard and vegetables typically yield 30-80% higher "
                "income than paddy in Northeast India."
            )

        else:
            reply = (
                "🌾 *AgriIntel*\n\n"
                "I didn't understand that. Send *help* to see available commands."
            )

        send_whatsapp_message(phone, reply)
        return {"status": "replied", "to": phone}

    except Exception as e:
        logger.error(f"WhatsApp webhook error: {e}")
        return {"status": "error", "detail": str(e)}


# ── 15. MULTI-SEASON TRACKING ───────────────────────────────────────
@app.post("/analytics/season-comparison", tags=["Analytics"])
def season_comparison(seasons: List[dict]):
    """
    Compare farmer income across multiple seasons.
    Input: List of {season, year, predicted_income, crop_type}
    Returns: trend analysis and stability score
    """
    if len(seasons) < 2:
        raise HTTPException(400, "Need at least 2 seasons to compare")

    incomes = [s["predicted_income"] for s in seasons]
    trend   = (incomes[-1] - incomes[0]) / incomes[0] * 100 if incomes[0] > 0 else 0
    volatility = (max(incomes) - min(incomes)) / np.mean(incomes) * 100

    stability_score = max(0, 100 - volatility)
    trend_label     = "GROWING" if trend > 10 else "STABLE" if trend > -10 else "DECLINING"

    return {
        "seasons":          seasons,
        "income_trend_pct": round(trend, 1),
        "trend_label":      trend_label,
        "volatility_pct":   round(volatility, 1),
        "stability_score":  round(stability_score, 1),
        "avg_income":       round(np.mean(incomes), -2),
        "min_income":       min(incomes),
        "max_income":       max(incomes),
        "recommendation":   (
            "Strong income growth — consider KCC limit enhancement"
            if trend > 20 else
            "Stable income — maintain current loan terms"
            if trend > -10 else
            "Declining income — review loan terms, offer restructuring"
        )
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
