def safe_load_model(file_path, device=None):
    """
    Safely load a PyTorch model using multiple approaches
    to handle different PyTorch versions and model formats.
    
    Args:
        file_path: Path to the model file
        device: Optional device to load the model to
    
    Returns:
        Loaded model state or None if failed
    """
    import torch
    import importlib
    
    # Determine device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try multiple loading approaches
    loading_errors = []
    
    # Approach 1: Traditional loading (weights_only=False)
    try:
        model_data = torch.load(file_path, weights_only=False, map_location=device)
        print(f"Successfully loaded model with weights_only=False")
        return model_data
    except Exception as e:
        loading_errors.append(f"weights_only=False error: {str(e)}")
    
    # Approach 2: Add safe globals for common classes
    try:
        # Add commonly used globals
        import argparse
        import collections
        import dataclasses
        import enum
        
        # Create a context where these globals are allowed
        with torch.serialization.safe_globals([
            argparse.Namespace, 
            collections.OrderedDict,
            collections.defaultdict,
            dataclasses.field,
            dataclasses.dataclass,
            enum.Enum,
            enum.IntEnum
        ]):
            model_data = torch.load(file_path, map_location=device)
            print(f"Successfully loaded model with safe_globals context")
            return model_data
    except Exception as e:
        loading_errors.append(f"safe_globals error: {str(e)}")
    
    # Approach 3: Use weights_only=True mode (safest)
    try:
        model_data = torch.load(file_path, weights_only=True, map_location=device)
        print(f"Successfully loaded model with weights_only=True")
        return model_data
    except Exception as e:
        loading_errors.append(f"weights_only=True error: {str(e)}")
    
    # Approach 4: Try manually registering known classes
    try:
        # Dynamically register common classes used in checkpoints
        torch.serialization.add_safe_globals([argparse.Namespace])
        
        # For PyTorch Lightning and other common ML frameworks
        for module_name in ["pytorch_lightning", "transformers", "timm"]:
            try:
                if importlib.util.find_spec(module_name):
                    module = importlib.import_module(module_name)
                    # Register the module itself as safe
                    torch.serialization.add_safe_globals([module])
            except:
                pass
        
        model_data = torch.load(file_path, map_location=device)
        print(f"Successfully loaded model with manually registered classes")
        return model_data
    except Exception as e:
        loading_errors.append(f"Manual registration error: {str(e)}")
    
    # All approaches failed, return detailed error
    error_message = "Failed to load model with all approaches:\n" + "\n".join(loading_errors)
    print(error_message)
    return None