from fastapi import FastAPI

app = FastAPI(
    title="Privacy Policy Analysis System",
    description="AI-based system for analyzing privacy policies and DPDP compliance",
    version="1.0"
)

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Privacy Policy Analysis API is working"
    }
