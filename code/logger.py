import logging
import sys

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,   # Change to DEBUG for more details
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output to console
        logging.FileHandler("app.log", mode="a")  # Save to file
    ]
)

# Optional: get a project-wide logger
logger = logging.getLogger("aihero")

# --- Suppress noisy third-party loggers ---
logging.getLogger("httpx").setLevel(logging.WARNING)   # silence httpx INFO
logging.getLogger("openai").setLevel(logging.WARNING)  # silence openai INFO