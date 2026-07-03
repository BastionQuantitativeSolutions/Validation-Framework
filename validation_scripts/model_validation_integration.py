"""Model Validation Integration Module"""
# Author: JG

import logging
from pathlib import Path
import joblib
from typing import Dict, Any

log = logging.getLogger(__name__)


class ModelValidationIntegration:
    """Integrate model validation into the trading system"""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.validated_models: Dict[str, Any] = {}

    def validate_model(self, symbol: str, timeframe: str) -> bool:
        """Validate a single model"""
        model_path = self.models_dir / f"{symbol}_{timeframe}"

        if not model_path.exists():
            log.warning(f"[VALIDATE] Model not found: {symbol}_{timeframe}")
            return False

        try:
            # Try to load the model
            model_data = joblib.load(model_path / "model.pkl")

            # Check for required attributes
            if not hasattr(model_data, "predict"):
                log.error(f"[VALIDATE] Model missing predict method: {symbol}_{timeframe}")
                return False

            self.validated_models[f"{symbol}_{timeframe}"] = {
                "path": str(model_path),
                "valid": True,
                "validated_at": str(Path.cwd()),
            }

            log.info(f"[VALIDATE] Model validated: {symbol}_{timeframe}")
            return True

        except Exception as e:
            log.error(f"[VALIDATE] Model validation failed: {symbol}_{timeframe}: {e}")
            return False

    def get_validated_models(self) -> Dict[str, Any]:
        """Get all validated models"""
        return self.validated_models
