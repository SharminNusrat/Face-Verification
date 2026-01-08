# Face Verification API

A FastAPI-based microservice for Face Verification (1:1 Face Matching).

## Features
- Verify if two images belong to the same person.
- Uses Deep Learning models via `deepface`.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Documentation
Once running, go to `http://localhost:8000/docs` to verify endpoints.
