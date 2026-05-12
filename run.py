"""
SafeSite AI — Entry Point
Run this script to start the FastAPI backend server.
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("  SafeSite AI — Construction Site Safety Monitor")
    print("  Starting server at http://localhost:8000")
    print("=" * 60)
    uvicorn.run(
        "backend.app:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info",
    )
