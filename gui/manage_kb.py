"""
Utility script to manage knowledge base
Upload documents and metrics offline
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_base import initialize_default_knowledge_base, MetricsKnowledgeBase


def upload_document(kb, file_path, title=None, category=None):
    """
    Upload a document to knowledge base
    
    Args:
        kb: KnowledgeBase instance
        file_path: Path to document
        title: Document title
        category: Document category
    """
    metadata = {}
    if title:
        metadata['title'] = title
    if category:
        metadata['category'] = category
    
    kb.add_document_from_file(file_path, metadata)
    print(f"✓ Uploaded: {file_path}")


def upload_folder(kb, folder_path, category=None):
    """
    Upload all documents from a folder
    
    Args:
        kb: KnowledgeBase instance
        folder_path: Path to folder
        category: Category for all documents
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"✗ Folder not found: {folder_path}")
        return
    
    supported_extensions = ['.txt', '.md', '.json']
    files = []
    
    for ext in supported_extensions:
        files.extend(folder.glob(f'**/*{ext}'))
    
    print(f"Found {len(files)} documents in {folder_path}")
    
    for file in files:
        metadata = {
            'category': category or 'general',
            'subfolder': str(file.parent.relative_to(folder))
        }
        kb.add_document_from_file(str(file), metadata)
        print(f"✓ Uploaded: {file.name}")


def add_confusion_matrix_from_json(kb, json_path):
    """
    Add confusion matrix from JSON file
    
    JSON format:
    {
        "matrix": [[tp, fp, ...], [fn, tn, ...]],
        "labels": ["CR", "LP", "ND", "PO"],
        "metadata": {"model": "CNN", "date": "2024-01-01"}
    }
    
    Args:
        kb: KnowledgeBase instance
        json_path: Path to JSON file
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cm = np.array(data['matrix'])
    labels = data['labels']
    metadata = data.get('metadata', {})
    
    metrics_kb = MetricsKnowledgeBase(kb)
    metrics_kb.add_confusion_matrix(cm, labels, metadata)
    print(f"✓ Added confusion matrix from {json_path}")


def add_metrics_from_json(kb, json_path):
    """
    Add training metrics from JSON file
    
    Args:
        kb: KnowledgeBase instance
        json_path: Path to JSON file
    """
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    metrics_kb = MetricsKnowledgeBase(kb)
    metrics_kb.add_training_metrics(metrics['metrics'], metrics.get('metadata', {}))
    print(f"✓ Added metrics from {json_path}")


def show_statistics(kb):
    """Show knowledge base statistics"""
    stats = kb.get_statistics()
    print("\n" + "="*50)
    print("Knowledge Base Statistics")
    print("="*50)
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Total Embeddings: {stats['total_embeddings']}")
    print(f"Storage Path: {stats['storage_path']}")
    print(f"Documents File Size: {stats['documents_file_size'] / 1024:.2f} KB")
    print(f"Embeddings File Size: {stats['embeddings_file_size'] / 1024:.2f} KB")
    print("="*50 + "\n")


def search_knowledge_base(kb, query, top_k=3):
    """
    Search knowledge base
    
    Args:
        kb: KnowledgeBase instance
        query: Search query
        top_k: Number of results
    """
    print(f"\nSearching for: '{query}'")
    print("-"*50)
    
    results = kb.search(query, top_k)
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n[Result {i}] Similarity: {result['similarity']:.4f}")
        if 'metadata' in result:
            print(f"Metadata: {result['metadata']}")
        print(f"Content Preview:\n{result['content'][:300]}...\n")


def interactive_menu():
    """Interactive menu for knowledge base management"""
    print("\n" + "="*50)
    print("Knowledge Base Management")
    print("="*50)
    
    # Initialize knowledge base
    kb_path = Path(__file__).parent / "knowledge_base"
    kb = initialize_default_knowledge_base(str(kb_path))
    
    while True:
        print("\nOptions:")
        print("1. Upload single document")
        print("2. Upload folder of documents")
        print("3. Add confusion matrix (JSON)")
        print("4. Add training metrics (JSON)")
        print("5. Search knowledge base")
        print("6. Show statistics")
        print("7. Clear knowledge base")
        print("8. Exit")
        
        choice = input("\nEnter choice (1-8): ").strip()
        
        if choice == '1':
            file_path = input("Enter file path: ").strip()
            title = input("Enter title (optional): ").strip() or None
            category = input("Enter category (optional): ").strip() or None
            upload_document(kb, file_path, title, category)
            
        elif choice == '2':
            folder_path = input("Enter folder path: ").strip()
            category = input("Enter category for all documents: ").strip() or None
            upload_folder(kb, folder_path, category)
            
        elif choice == '3':
            json_path = input("Enter JSON file path: ").strip()
            add_confusion_matrix_from_json(kb, json_path)
            
        elif choice == '4':
            json_path = input("Enter JSON file path: ").strip()
            add_metrics_from_json(kb, json_path)
            
        elif choice == '5':
            query = input("Enter search query: ").strip()
            top_k = int(input("Number of results (default 3): ").strip() or "3")
            search_knowledge_base(kb, query, top_k)
            
        elif choice == '6':
            show_statistics(kb)
            
        elif choice == '7':
            confirm = input("Are you sure you want to clear the knowledge base? (yes/no): ").strip().lower()
            if confirm == 'yes':
                kb.clear()
                print("✓ Knowledge base cleared")
            
        elif choice == '8':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1]
        
        kb_path = Path(__file__).parent / "knowledge_base"
        kb = initialize_default_knowledge_base(str(kb_path))
        
        if command == "stats":
            show_statistics(kb)
            
        elif command == "upload" and len(sys.argv) > 2:
            file_path = sys.argv[2]
            title = sys.argv[3] if len(sys.argv) > 3 else None
            category = sys.argv[4] if len(sys.argv) > 4 else None
            upload_document(kb, file_path, title, category)
            
        elif command == "upload-folder" and len(sys.argv) > 2:
            folder_path = sys.argv[2]
            category = sys.argv[3] if len(sys.argv) > 3 else None
            upload_folder(kb, folder_path, category)
            
        elif command == "search" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            search_knowledge_base(kb, query)
            
        else:
            print("Usage:")
            print("  python manage_kb.py stats")
            print("  python manage_kb.py upload <file_path> [title] [category]")
            print("  python manage_kb.py upload-folder <folder_path> [category]")
            print("  python manage_kb.py search <query>")
            print("\nOr run without arguments for interactive mode")
    else:
        # Interactive mode
        interactive_menu()
