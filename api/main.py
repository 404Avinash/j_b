"""
JATAYU - FastAPI Backend
=========================
REST API for attack prediction and intel analysis.

Endpoints:
- GET /predict - Attack probability prediction
- GET /explain - Risk explanation
- GET /intel - Get intel by date/region
- GET /stats - Dashboard statistics
"""

import os
import sys
import pickle
from datetime import datetime, date
from typing import Optional, List

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Try to import model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'production_model.pkl')
INTEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'all_intel_2020_2026.csv')

# Initialize FastAPI
app = FastAPI(
    title="JATAYU API",
    description="Predictive Intelligence Fusion System for IED Attack Prevention",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded data
model_data = None
intel_sample = None


# Response models
class PredictionResponse(BaseModel):
    date: str
    region: str
    attack_probability: float
    risk_level: str
    top_signals: List[dict]
    recommendation: str


class ExplainResponse(BaseModel):
    region: str
    date: str
    risk_factors: List[dict]
    intel_summary: dict
    why_risky: str


class StatsResponse(BaseModel):
    total_intel_records: int
    date_range: dict
    by_region: dict
    by_type: dict
    by_label: dict
    daily_average: float


def load_model():
    """Load trained model."""
    global model_data
    if model_data is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        print(f"[API] Loaded model from {MODEL_PATH}")
    return model_data


def load_intel_sample(n_rows: int = 100000):
    """Load sample of intel data."""
    global intel_sample
    if intel_sample is None and os.path.exists(INTEL_PATH):
        intel_sample = pd.read_csv(INTEL_PATH, nrows=n_rows)
        print(f"[API] Loaded {len(intel_sample):,} intel records")
    return intel_sample


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_model()
    load_intel_sample()


@app.get("/")
async def root():
    """API health check."""
    return {
        "status": "online",
        "system": "JATAYU - Predictive Intelligence Fusion",
        "version": "1.0.0",
        "endpoints": ["/predict", "/explain", "/intel", "/stats"]
    }


@app.get("/predict", response_model=PredictionResponse)
async def predict_attack(
    region: str = Query(..., description="Region/District name"),
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")
):
    """
    Predict attack probability for a region.
    
    Returns risk level and recommendations.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Get intel for region
    intel = load_intel_sample()
    if intel is None:
        raise HTTPException(status_code=500, detail="Intel data not loaded")
    
    region_intel = intel[intel['District'].str.contains(region, case=False, na=False)]
    
    if len(region_intel) == 0:
        raise HTTPException(status_code=404, detail=f"No intel found for region: {region}")
    
    # Calculate features
    total = len(region_intel)
    pct_true = (region_intel['Label'] == 'TRUE_SIGNAL').sum() / total
    avg_intensity = region_intel['Signal_Intensity'].mean()
    pct_high_urgency = (region_intel['Urgency'] == 'HIGH').sum() / total
    
    # Simple prediction based on features
    # (In production, would use trained model)
    attack_prob = min(1.0, 0.3 * pct_true + 0.4 * avg_intensity + 0.3 * pct_high_urgency)
    attack_prob = round(attack_prob, 3)
    
    # Risk level
    if attack_prob >= 0.7:
        risk_level = "CRITICAL"
        recommendation = "Suspend all patrols. Deploy EOD teams. Maximum alert."
    elif attack_prob >= 0.5:
        risk_level = "HIGH"
        recommendation = "Enhanced patrols with mine-protected vehicles. Increase SIGINT."
    elif attack_prob >= 0.3:
        risk_level = "MEDIUM"
        recommendation = "Standard precautions. Continue monitoring."
    else:
        risk_level = "LOW"
        recommendation = "Normal operations."
    
    # Top signals
    true_signals = region_intel[region_intel['Label'] == 'TRUE_SIGNAL'].head(5)
    top_signals = [
        {
            "type": row['Intel_Type'],
            "content": row['Content'][:100] if 'Content' in row else "Signal detected",
            "reliability": row['Reliability'],
            "urgency": row['Urgency']
        }
        for _, row in true_signals.iterrows()
    ]
    
    return PredictionResponse(
        date=date,
        region=region,
        attack_probability=attack_prob,
        risk_level=risk_level,
        top_signals=top_signals,
        recommendation=recommendation
    )


@app.get("/explain", response_model=ExplainResponse)
async def explain_risk(
    region: str = Query(..., description="Region/District name"),
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")
):
    """
    Explain why a region is risky.
    
    Provides breakdown of contributing factors.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    intel = load_intel_sample()
    if intel is None:
        raise HTTPException(status_code=500, detail="Intel data not loaded")
    
    region_intel = intel[intel['District'].str.contains(region, case=False, na=False)]
    
    if len(region_intel) == 0:
        raise HTTPException(status_code=404, detail=f"No intel found for region: {region}")
    
    total = len(region_intel)
    
    # Risk factors
    risk_factors = [
        {
            "factor": "True Signal Density",
            "value": round((region_intel['Label'] == 'TRUE_SIGNAL').sum() / total * 100, 1),
            "impact": "HIGH" if (region_intel['Label'] == 'TRUE_SIGNAL').sum() / total > 0.5 else "MEDIUM"
        },
        {
            "factor": "Average Signal Intensity",
            "value": round(region_intel['Signal_Intensity'].mean() * 100, 1),
            "impact": "HIGH" if region_intel['Signal_Intensity'].mean() > 0.6 else "MEDIUM"
        },
        {
            "factor": "High Urgency Intel %",
            "value": round((region_intel['Urgency'] == 'HIGH').sum() / total * 100, 1),
            "impact": "HIGH" if (region_intel['Urgency'] == 'HIGH').sum() / total > 0.3 else "LOW"
        },
        {
            "factor": "Deception Attempts",
            "value": (region_intel['Label'] == 'DECEPTION').sum(),
            "impact": "WARNING" if (region_intel['Label'] == 'DECEPTION').sum() > 10 else "LOW"
        }
    ]
    
    # Intel summary
    intel_summary = {
        "total_records": total,
        "humint": (region_intel['Intel_Type'] == 'HUMINT').sum(),
        "sigint": (region_intel['Intel_Type'] == 'SIGINT').sum(),
        "patrol": (region_intel['Intel_Type'] == 'PATROL').sum(),
        "osint": (region_intel['Intel_Type'] == 'OSINT').sum(),
        "true_signals": (region_intel['Label'] == 'TRUE_SIGNAL').sum(),
        "noise": (region_intel['Label'] == 'NOISE').sum(),
        "deception": (region_intel['Label'] == 'DECEPTION').sum(),
    }
    
    # Generate explanation
    true_pct = (region_intel['Label'] == 'TRUE_SIGNAL').sum() / total * 100
    intensity = region_intel['Signal_Intensity'].mean() * 100
    
    why_risky = f"{region} has {true_pct:.1f}% true signals with {intensity:.1f}% average intensity. "
    if intensity > 60:
        why_risky += "High signal intensity indicates imminent threat. "
    if true_pct > 50:
        why_risky += "Elevated true signal rate suggests active reconnaissance or preparation. "
    
    return ExplainResponse(
        region=region,
        date=date,
        risk_factors=risk_factors,
        intel_summary=intel_summary,
        why_risky=why_risky
    )


@app.get("/intel")
async def get_intel(
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    region: Optional[str] = Query(None, description="Filter by region"),
    intel_type: Optional[str] = Query(None, description="Filter by type (HUMINT/SIGINT/PATROL/OSINT)"),
    label: Optional[str] = Query(None, description="Filter by label (TRUE_SIGNAL/NOISE/DECEPTION)"),
    limit: int = Query(50, description="Max records to return")
):
    """
    Get intel records with optional filters.
    """
    intel = load_intel_sample()
    if intel is None:
        raise HTTPException(status_code=500, detail="Intel data not loaded")
    
    result = intel.copy()
    
    if date:
        result = result[result['Date'] == date]
    if region:
        result = result[result['District'].str.contains(region, case=False, na=False)]
    if intel_type:
        result = result[result['Intel_Type'] == intel_type.upper()]
    if label:
        result = result[result['Label'] == label.upper()]
    
    result = result.head(limit)
    
    return {
        "count": len(result),
        "records": result.to_dict(orient='records')
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get overall system statistics for dashboard.
    """
    intel = load_intel_sample()
    if intel is None:
        raise HTTPException(status_code=500, detail="Intel data not loaded")
    
    total = len(intel)
    
    # By region
    by_region = intel['District'].value_counts().to_dict()
    
    # By type
    by_type = {
        "HUMINT": int((intel['Intel_Type'] == 'HUMINT').sum()),
        "SIGINT": int((intel['Intel_Type'] == 'SIGINT').sum()),
        "PATROL": int((intel['Intel_Type'] == 'PATROL').sum()),
        "OSINT": int((intel['Intel_Type'] == 'OSINT').sum()),
    }
    
    # By label
    by_label = {
        "TRUE_SIGNAL": int((intel['Label'] == 'TRUE_SIGNAL').sum()),
        "NOISE": int((intel['Label'] == 'NOISE').sum()),
        "DECEPTION": int((intel['Label'] == 'DECEPTION').sum()),
    }
    
    # Daily average
    if 'Date' in intel.columns:
        daily_counts = intel.groupby('Date').size()
        daily_avg = daily_counts.mean()
    else:
        daily_avg = 0
    
    return StatsResponse(
        total_intel_records=total,
        date_range={
            "start": str(intel['Date'].min()),
            "end": str(intel['Date'].max())
        },
        by_region=by_region,
        by_type=by_type,
        by_label=by_label,
        daily_average=round(daily_avg, 1)
    )


@app.get("/regions")
async def get_regions():
    """Get list of all regions in the data."""
    intel = load_intel_sample()
    if intel is None:
        raise HTTPException(status_code=500, detail="Intel data not loaded")
    
    regions = intel['District'].dropna().unique().tolist()
    return {"regions": sorted(regions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
