"""
Migrate Pickle Embeddings to NumPy Format
Converts existing embeddings.pkl files to secure embeddings.npy files
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import json

def migrate_embeddings(kb_path: str = "gui/knowledge_base"):
    """
    Migrate embeddings from pickle to numpy format
    
    Args:
        kb_path: Path to knowledge base directory
    """
    kb_dir = Path(kb_path)
    
    if not kb_dir.exists():
        print(f"Knowledge base directory not found: {kb_path}")
        return False
    
    pickle_file = kb_dir / "embeddings.pkl"
    numpy_file = kb_dir / "embeddings.npy"
    backup_file = kb_dir / "embeddings.pkl.backup"
    
    print("=" * 70)
    print("MIGRATING EMBEDDINGS FROM PICKLE TO NUMPY FORMAT")
    print("=" * 70)
    
    # Check if pickle file exists
    if not pickle_file.exists():
        print(f"\n✓ No pickle file found at {pickle_file}")
        print("  Nothing to migrate. System is already using numpy format.")
        return True
    
    # Check if numpy file already exists
    if numpy_file.exists():
        print(f"\n⚠ NumPy file already exists: {numpy_file}")
        response = input("  Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("  Migration cancelled.")
            return False
    
    try:
        print(f"\n[1/4] Loading embeddings from pickle file...")
        with open(pickle_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        print(f"      ✓ Loaded {len(embeddings)} embeddings")
        
        print(f"\n[2/4] Converting to NumPy array...")
        embeddings_array = np.array(embeddings)
        print(f"      ✓ Shape: {embeddings_array.shape}")
        
        print(f"\n[3/4] Saving to NumPy format...")
        np.save(numpy_file, embeddings_array, allow_pickle=False)
        print(f"      ✓ Saved to {numpy_file}")
        
        print(f"\n[4/4] Creating backup of pickle file...")
        import shutil
        shutil.copy2(pickle_file, backup_file)
        print(f"      ✓ Backup created at {backup_file}")
        
        print("\n" + "=" * 70)
        print("MIGRATION SUCCESSFUL!")
        print("=" * 70)
        
        print(f"\n✓ Embeddings migrated to secure NumPy format")
        print(f"✓ Original pickle file backed up to: {backup_file}")
        print(f"\nYou can now safely delete the pickle file:")
        print(f"  {pickle_file}")
        
        # Verify the migration
        print(f"\n[Verification] Loading numpy file to verify...")
        loaded_embeddings = np.load(numpy_file, allow_pickle=False)
        print(f"✓ Successfully loaded {len(loaded_embeddings)} embeddings from numpy file")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_status(kb_path: str = "gui/knowledge_base"):
    """Check the status of embeddings storage"""
    kb_dir = Path(kb_path)
    
    print("=" * 70)
    print("EMBEDDINGS STORAGE STATUS")
    print("=" * 70)
    
    pickle_file = kb_dir / "embeddings.pkl"
    numpy_file = kb_dir / "embeddings.npy"
    documents_file = kb_dir / "documents.json"
    
    print(f"\nKnowledge Base Directory: {kb_dir}")
    print(f"")
    
    if pickle_file.exists():
        size_kb = pickle_file.stat().st_size / 1024
        print(f"🔴 Pickle file (INSECURE):  {pickle_file.name} ({size_kb:.1f} KB)")
        print(f"   ⚠ This file uses pickle format which has security vulnerabilities")
    else:
        print(f"✓  No pickle file found")
    
    if numpy_file.exists():
        size_kb = numpy_file.stat().st_size / 1024
        print(f"✓  NumPy file (SECURE):    {numpy_file.name} ({size_kb:.1f} KB)")
    else:
        print(f"   No numpy file found")
    
    if documents_file.exists():
        with open(documents_file, 'r') as f:
            docs = json.load(f)
        print(f"✓  Documents file:         {documents_file.name} ({len(docs)} documents)")
    else:
        print(f"   No documents file found")
    
    print("\n" + "=" * 70)
    
    if pickle_file.exists() and not numpy_file.exists():
        print("⚠ ACTION REQUIRED: Run migration to convert to secure format")
        print("  Run: python migrate_embeddings_to_numpy.py migrate")
    elif pickle_file.exists() and numpy_file.exists():
        print("✓ Migration complete. You can now delete the pickle file.")
    elif numpy_file.exists():
        print("✓ Using secure NumPy format. No action needed.")
    else:
        print("ℹ No embeddings files found. Will use NumPy format when created.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        kb_path = sys.argv[2] if len(sys.argv) > 2 else "gui/knowledge_base"
        
        if command == "migrate":
            migrate_embeddings(kb_path)
        elif command == "status":
            check_status(kb_path)
        else:
            print("Usage:")
            print("  python migrate_embeddings_to_numpy.py status [kb_path]")
            print("  python migrate_embeddings_to_numpy.py migrate [kb_path]")
    else:
        # Default: show status
        check_status()
