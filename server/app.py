import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from app import app 
def main(): 
    import uvicorn 
    port = int(os.environ.get("PORT", "7860")) 
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info") 
if __name__ == "__main__": 
    main() 
