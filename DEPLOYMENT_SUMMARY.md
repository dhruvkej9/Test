
# Deployment Structure Summary

## Vercel Deployment
- Serverless API functions in /api/ directory
- Routes configured in vercel.json
- Environment detection for API URLs

## Streamlit Community Cloud Deployment  
- Main app: frontend/app.py
- Advanced app: frontend/app_advanced.py
- Shared utilities in frontend/src/

## API Endpoints
- /api/predict - Molecular property prediction
- /api/agent - Agentic AI interactions  
- /api/accuracy - Model accuracy metrics

## Key Features
- Automatic localhost/cloud API detection
- Shared utility functions
- Comprehensive dependency management
- Ready for one-click deployment

Fri Aug  1 16:45:22 UTC 2025: Refactoring complete and validated

