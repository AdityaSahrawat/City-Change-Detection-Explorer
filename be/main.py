"""
Entry point for the City Change Detection Explorer backend.

Run with:
    uv run uvicorn app.api:app --reload --port 8000
or:
    uv run python main.py
"""
import uvicorn


def main() -> None:
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
