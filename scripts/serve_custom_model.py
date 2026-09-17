"""
Script to serve your custom trained XRFM model live via FastAPI server.

Usage:
    python scripts/serve_custom_model.py --port 8000
"""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Serve custom XRFM live API server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"🚀 Starting Custom XRFM Server on http://{args.host}:{args.port}")
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=True)


if __name__ == "__main__":
    main()
