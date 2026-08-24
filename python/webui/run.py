"""Entry point: run the web UI server."""

import uvicorn


def main() -> None:
    uvicorn.run("webui.server:app", host="0.0.0.0", port=3000, log_level="info")


if __name__ == "__main__":
    main()
