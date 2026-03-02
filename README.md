# Liveness Detection — Deploy Guide

## ফোল্ডার Structure
```
liveness-app/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── Procfile
└── frontend/
    └── index.html
```

---

## ধাপ ১ — Backend → Railway

1. https://railway.app এ যান → GitHub দিয়ে login
2. "New Project" → "Deploy from GitHub repo"
3. `backend/` folder টা আলাদা GitHub repo হিসেবে push করুন
4. Railway automatically detect করবে Procfile দেখে
5. Deploy হলে URL পাবেন — যেমন: `https://liveness-xxx.railway.app`

---

## ধাপ ২ — Frontend → Vercel

1. https://vercel.com এ যান → GitHub দিয়ে login
2. "New Project" → `frontend/` folder টা GitHub repo হিসেবে push করুন
3. Deploy করুন — Vercel আপনাকে একটা URL দেবে

---

## ধাপ ৩ — Connect করুন

Vercel এর page খুলুন → Railway এর URL বক্সে paste করুন → Done!

---

## Local Test (deploy ছাড়া)

```powershell
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

তারপর `frontend/index.html` browser এ খুলুন এবং URL দিন:
`http://localhost:8000`
