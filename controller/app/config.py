"""Central configuration for the mdai controller service."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = ROOT_DIR / ".env"


class Settings(BaseSettings):
    """Environment-driven settings for controller subsystems."""

    device_id: str = Field(..., description="Unique device identifier used during pairing")
    evm_address: str = Field(..., description="Public EVM address for signature association")
    private_key: str = Field(..., description="Hex-encoded private key used for nonce signing")
    public_key: str = Field(..., description="Hex-encoded public key for diagnostics")

    backend_api_url: str = Field(..., description="Base URL for backend REST calls")
    backend_ws_url: str = Field(..., description="Base URL for backend WebSocket endpoint")

    controller_host: str = Field("0.0.0.0", description="Host interface for local FastAPI server")
    controller_port: int = Field(5000, description="Port for FastAPI server")

    tof_threshold_mm: int = Field(450, description="Distance threshold that triggers workflow")
    tof_debounce_ms: int = Field(200, description="Debounce period before treating ToF trigger as valid")

    preview_frame_width: int = Field(640, description="Preview width for MJPEG streaming")
    preview_frame_height: int = Field(480, description="Preview height for MJPEG streaming")
    preview_fps: int = Field(15, description="Target FPS for preview stream")

    mediapipe_stride: int = Field(3, description="Stride used by MediaPipe liveness worker")
    mediapipe_confidence: float = Field(0.6, description="Minimum face detector confidence")
    stability_seconds: float = Field(4.0, description="Duration the user must stay stable")

    realsense_enable_hardware: bool = Field(
        False, description="Enable RealSense hardware pipeline (set True on Jetson with camera attached)"
    )

    log_level: str = Field("INFO", description="Logging level for controller")

    class Config:
        env_file = str(DEFAULT_ENV_FILE)
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings(override_env_file: Optional[Path] = None) -> Settings:
    """Cached Settings instance; accepts optional env override for tests."""

    if override_env_file:
        return Settings(_env_file=str(override_env_file))
    return Settings()
