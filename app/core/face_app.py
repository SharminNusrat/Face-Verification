import threading
from insightface.app import FaceAnalysis
from app.core.config import settings

class FaceAppProvider:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_app(cls):
        """
        Returns the singleton instance of FaceAnalysis.
        Thread-safe implementation.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print(f"Initializing InsightFace app with model: {settings.MODEL_NAME}")
                    app = FaceAnalysis(name=settings.MODEL_NAME)
                    # Use ctx_id from settings or default to 0 (GPU 0) or -1 (CPU) depending on config
                    # Assuming settings.DEVICE might need parsing if it's 'cuda' vs 'cpu'
                    # insightface uses ctx_id: -1 for CPU, 0+ for GPU
                    
                    ctx_id = 0 if torch_available_and_cuda() else -1
                    
                    # For simplicity, using 0 if cuda is available, else -1 based on simple check
                    # But verifying 'settings.DEVICE' is safer if that's what we want to rely on.
                    # Config has prompt "DEVICE: str = 'cpu'". Let's stick to auto-detection or config.
                    
                    # Let's perform a robust check or use the config value if mapped.
                    # Given previous files used ctx_id=0, let's try that first if GPU is desired,
                    # but safe fallback is important.
                    
                    try:
                        app.prepare(ctx_id=0, det_size=settings.DET_SIZE)
                    except Exception as e:
                        print(f"Failed to initialize on GPU (ctx_id=0), falling back to CPU (ctx_id=-1). Error: {e}")
                        app.prepare(ctx_id=-1, det_size=settings.DET_SIZE)
                        
                    cls._instance = app
        return cls._instance

# Helper to check CUDA without importing torch globally if not needed yet, 
# but we likely have torch installed now.
def torch_available_and_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
