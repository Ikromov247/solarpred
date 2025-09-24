import os

from solar_pred.core.config import config

async def check_model_health(app_state):
    """Check if model is loaded and ready."""
    try:
        if not hasattr(app_state, 'model') or app_state.model is None:
            return {"status": "unhealthy", "details": "Model not loaded"}
        
        model = app_state.model
        if not hasattr(model, 'is_trained') or not model.is_trained:
            return {"status": "unhealthy", "details": "Model not trained"}
        
        return {"status": "healthy", "details": "Model loaded and trained"}
    
    except Exception as e:
        return {"status": "unhealthy", "details": f"Model check failed: {str(e)}"}


async def check_filesystem_health():
    """Check file system accessibility."""
    try:
        # Check model directory
        if not os.path.exists(config.model_dir):
            return {"status": "unhealthy", "details": f"Model directory not found: {config.model_dir}"}
        
        if not os.access(config.model_dir, os.W_OK):
            return {"status": "unhealthy", "details": f"Model directory not writable: {config.model_dir}"}
        
        # Check log directory
        log_dir = os.path.join(config.volume_path, "logs")
        if not os.path.exists(log_dir):
            return {"status": "unhealthy", "details": f"Log directory not found: {log_dir}"}
        
        return {"status": "healthy", "details": "File system accessible"}
    
    except Exception as e:
        return {"status": "unhealthy", "details": f"Filesystem check failed: {str(e)}"}
