"""
Generate Security Hashes for Existing Models
Run this script to create .hash files for all existing model files
"""

import sys
from pathlib import Path

# Add gui directory to path
sys.path.append(str(Path(__file__).parent / 'gui'))

from security_utils import ModelIntegrityVerifier


def generate_hashes_for_models():
    """Generate hash files for all existing model files in the project"""
    
    verifier = ModelIntegrityVerifier()
    
    # Directories to scan for models
    model_dirs = [
        Path('models'),
        Path('gui/build/RIAWELC'),
        Path('cae/models')
    ]
    
    total_models = 0
    successful = 0
    
    print("=" * 70)
    print("GENERATING SECURITY HASHES FOR EXISTING MODELS")
    print("=" * 70)
    
    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"\nSkipping {model_dir} (not found)")
            continue
        
        print(f"\n[{model_dir}]")
        
        # Find all .pth and .pt files
        model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
        
        if not model_files:
            print("  No model files found")
            continue
        
        for model_file in model_files:
            total_models += 1
            print(f"\n  Processing: {model_file.name}")
            
            try:
                # Check if hash already exists
                hash_file = Path(str(model_file) + ".hash")
                if hash_file.exists():
                    print(f"    ⚠ Hash file already exists, skipping...")
                    successful += 1
                    continue
                
                # Generate hash
                success = verifier.save_model_with_verification(str(model_file))
                
                if success:
                    print(f"    ✓ Hash generated successfully")
                    successful += 1
                else:
                    print(f"    ✗ Failed to generate hash")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {successful}/{total_models} models processed successfully")
    print("=" * 70)
    
    if successful < total_models:
        print("\n⚠ Some models failed to process. Check the output above for details.")
    else:
        print("\n✓ All model files now have security hashes!")
    
    print("\nThese .hash files will be used to verify model integrity when loading.")
    print("Keep them alongside your .pth model files.")
    

if __name__ == "__main__":
    try:
        generate_hashes_for_models()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
