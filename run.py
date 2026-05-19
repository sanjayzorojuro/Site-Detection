"""
SafeSite AI — Entry Point
Run this script to start the FastAPI backend server.
"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("=" * 60)
    print("  SafeSite AI — Construction Site Safety Monitor")
    print(f"  Starting server at http://localhost:{port}")
    print("=" * 60)
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
