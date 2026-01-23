# JATAYU - Full Stack Setup

## Quick Start

### 1. Start Backend (Terminal 1)

```bash
cd c:\Users\aashu\Downloads\jatayu_beta
pip install fastapi uvicorn
python -m uvicorn backend.server:app --reload --port 8000
```

### 2. Start Frontend (Terminal 2)

```bash
cd c:\Users\aashu\Downloads\jatayu_beta\frontend
npm install
npm run dev
```

### 3. Open in Browser

- Frontend: <http://localhost:3000>
- Backend API: <http://localhost:8000>

## How It Works

1. Enter 2-3 recent attacks (location, date, type)
2. Click "PREDICT NEXT ATTACK"
3. ML model analyzes patterns and predicts:
   - Next attack date
   - Likely location
   - Risk level
   - Recommendations

## Project Structure

```
jatayu_beta/
├── backend/
│   └── server.py          # FastAPI prediction server
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   ├── main.jsx       # Entry point
│   │   └── index.css      # Styling
│   ├── package.json
│   └── vite.config.js
├── data/
│   └── all_intel_2020_2026.csv  # 8.2M intel records
├── models/
│   └── production_model.pkl     # Trained XGBoost model
├── start_backend.bat
└── start_frontend.bat
```
