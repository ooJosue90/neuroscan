import sys
import importlib.util

def check_module(module_name: str, specific_attrs: list = None):
    print(f"\n--- Checking '{module_name}' ---")
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ ERROR: Module '{module_name}' is not installed.")
        return False
    
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported '{module_name}'.")
        if hasattr(module, '__version__'):
            print(f"ℹ️ Version: {module.__version__}")
            
        if specific_attrs:
            for attr in specific_attrs:
                if hasattr(module, attr):
                    print(f"✅ Found attribute '{attr}'.")
                else:
                    print(f"❌ ERROR: Missing attribute '{attr}' in '{module_name}'.")
                    
        # Extra check for mediapipe
        if module_name == 'mediapipe':
            try:
                import mediapipe.python.solutions as solutions
                print("✅ Successfully accessed 'mediapipe.python.solutions'.")
            except Exception as e:
                print(f"❌ ERROR accessing mediapipe solutions: {e}")
                
        return True
    except Exception as e:
        print(f"❌ ERROR loading '{module_name}': {e}")
        return False

def main():
    print("="*50)
    print("Nueroscan V4 - Dependency Diagnostic Tool")
    print("="*50)
    
    print(f"Python version: {sys.version}")
    
    modules_to_test = [
        ("cv2", ["VideoCapture", "cuda"]),
        ("numpy", ["array"]),
        ("torch", ["cuda"]),
        ("transformers", ["pipeline"]),
        ("librosa", ["load", "feature"]),
        ("parselmouth", ["Sound"]),
        ("mediapipe", ["solutions"]),
        ("scipy", ["signal"])
    ]
    
    for mod, attrs in modules_to_test:
        check_module(mod, attrs)

if __name__ == "__main__":
    main()
