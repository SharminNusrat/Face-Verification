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
                    
                    ctx_id = 0 if torch_available_and_cuda() else -1 # 0 == GPU | -1 == CPU
                    # if possible set device in the config.settings file.
                    
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
