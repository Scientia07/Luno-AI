"""
Luno-AI API Configuration
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    app_name: str = "Luno-AI"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = True

    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    integrations_path: Path = project_root / "integrations"
    research_path: Path = project_root / "research"
    labs_path: Path = project_root / "labs"
    data_path: Path = project_root / "data"

    # Database
    database_url: str = "sqlite:///data/luno.db"

    # ChromaDB
    chroma_persist_dir: str = "data/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"  # Local model, no API needed

    # Domain configuration
    domains: dict = {
        "visual-ai": {
            "name": "Visual AI",
            "description": "Object detection, segmentation, classification",
            "icon": "eye",
            "color": "#3B82F6"
        },
        "generative": {
            "name": "Generative AI",
            "description": "Image, video, and content generation",
            "icon": "sparkles",
            "color": "#EC4899"
        },
        "audio": {
            "name": "Audio AI",
            "description": "Speech recognition, synthesis, and music",
            "icon": "volume-2",
            "color": "#F59E0B"
        },
        "llms": {
            "name": "LLMs",
            "description": "Large language models and inference",
            "icon": "message-square",
            "color": "#10B981"
        },
        "agents": {
            "name": "Agentic AI",
            "description": "Autonomous agents and workflows",
            "icon": "bot",
            "color": "#8B5CF6"
        },
        "ml": {
            "name": "Classical ML",
            "description": "Traditional machine learning and AutoML",
            "icon": "trending-up",
            "color": "#06B6D4"
        },
        "deploy": {
            "name": "Edge & Deployment",
            "description": "Model optimization and deployment",
            "icon": "cloud",
            "color": "#6366F1"
        },
        "robotics": {
            "name": "Robotics",
            "description": "Robot control and simulation",
            "icon": "cpu",
            "color": "#EF4444"
        },
        "specialized": {
            "name": "Specialized",
            "description": "Time series, anomaly detection, translation",
            "icon": "zap",
            "color": "#84CC16"
        }
    }

    class Config:
        env_file = ".env"


settings = Settings()
