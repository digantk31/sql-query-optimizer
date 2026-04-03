import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from app import app 
if __name__ == "__main__": 
    import uvicorn 
    port = int(os.environ.get("PORT", "7860")) 
    uvicorn.run(app, host="0.0.0.0", port=port) 
