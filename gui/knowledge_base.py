"""
Knowledge Base System for Weld Defect Detection
Handles document embeddings and semantic search using Azure OpenAI
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv


class KnowledgeBase:
    """Manages document embeddings and semantic search"""
    
    def __init__(self, storage_path: str = "knowledge_base"):
        """
        Initialize knowledge base
        
        Args:
            storage_path: Directory to store embeddings and documents
        """
        load_dotenv()
        
        # Azure OpenAI setup
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT")
        )
        
        self.embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
        
        # Storage setup
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.embeddings_file = self.storage_path / "embeddings.pkl"
        self.documents_file = self.storage_path / "documents.json"
        
        # Load existing data
        self.documents = []
        self.embeddings = []
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load existing embeddings and documents"""
        try:
            if self.documents_file.exists():
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                    
            print(f"Loaded {len(self.documents)} documents from knowledge base")
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.documents = []
            self.embeddings = []
    
    def save_knowledge_base(self):
        """Save embeddings and documents to disk"""
        try:
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
            
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
                
            print(f"Saved {len(self.documents)} documents to knowledge base")
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using Azure OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def add_document(self, content: str, metadata: Dict = None, chunk_size: int = 500):
        """
        Add document to knowledge base with chunking
        
        Args:
            content: Document content
            metadata: Additional metadata (title, source, category, etc.)
            chunk_size: Size of text chunks
        """
        if not content or not content.strip():
            print("Empty content, skipping")
            return
        
        # Split into chunks if content is long
        chunks = self._chunk_text(content, chunk_size)
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            # Get embedding
            embedding = self.get_embedding(chunk)
            if embedding is None:
                continue
            
            # Create document entry
            doc = {
                "content": chunk,
                "metadata": metadata or {},
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            
            self.documents.append(doc)
            self.embeddings.append(embedding)
        
        # Save after adding
        self.save_knowledge_base()
        print(f"Added document with {len(chunks)} chunks")
    
    def add_document_from_file(self, file_path: str, metadata: Dict = None):
        """
        Add document from file
        
        Args:
            file_path: Path to document file (.txt, .md, .json)
            metadata: Additional metadata
        """
        path = Path(file_path)
        
        if not path.exists():
            print(f"File not found: {file_path}")
            return
        
        try:
            # Read file based on extension
            if path.suffix.lower() in ['.txt', '.md']:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, indent=2)
            else:
                print(f"Unsupported file type: {path.suffix}")
                return
            
            # Add metadata
            if metadata is None:
                metadata = {}
            metadata['source'] = str(path)
            metadata['filename'] = path.name
            
            self.add_document(content, metadata)
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search knowledge base for relevant documents
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents with similarity scores
        """
        if not self.embeddings:
            return []
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
        
        # Calculate cosine similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results
        results = []
        for idx, score in similarities[:top_k]:
            doc = self.documents[idx].copy()
            doc['similarity'] = score
            results.append(doc)
        
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Text to chunk
            chunk_size: Approximate size of each chunk
            
        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def clear(self):
        """Clear all documents and embeddings"""
        self.documents = []
        self.embeddings = []
        self.save_knowledge_base()
        print("Knowledge base cleared")
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        return {
            "total_documents": len(self.documents),
            "total_embeddings": len(self.embeddings),
            "storage_path": str(self.storage_path),
            "documents_file_size": self.documents_file.stat().st_size if self.documents_file.exists() else 0,
            "embeddings_file_size": self.embeddings_file.stat().st_size if self.embeddings_file.exists() else 0
        }


class MetricsKnowledgeBase:
    """Specialized knowledge base for model metrics and results"""
    
    def __init__(self, kb: KnowledgeBase):
        """
        Initialize metrics KB
        
        Args:
            kb: Base knowledge base instance
        """
        self.kb = kb
    
    def add_confusion_matrix(self, cm: np.ndarray, labels: List[str], metadata: Dict = None):
        """
        Add confusion matrix to knowledge base
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            metadata: Additional metadata (model name, date, etc.)
        """
        # Create textual representation
        content = "Confusion Matrix Results:\n\n"
        
        # Add matrix
        content += "Matrix:\n"
        for i, label_i in enumerate(labels):
            row = []
            for j, label_j in enumerate(labels):
                row.append(f"{label_j}: {cm[i, j]}")
            content += f"{label_i} - {', '.join(row)}\n"
        
        # Calculate metrics
        content += "\nMetrics per class:\n"
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            content += f"\n{label}:\n"
            content += f"  Precision: {precision:.4f}\n"
            content += f"  Recall: {recall:.4f}\n"
            content += f"  F1-Score: {f1:.4f}\n"
            content += f"  True Positives: {tp}\n"
            content += f"  False Positives: {fp}\n"
            content += f"  False Negatives: {fn}\n"
        
        # Add overall accuracy
        accuracy = np.trace(cm) / cm.sum()
        content += f"\nOverall Accuracy: {accuracy:.4f}\n"
        
        if metadata is None:
            metadata = {}
        metadata['type'] = 'confusion_matrix'
        metadata['labels'] = labels
        
        self.kb.add_document(content, metadata)
    
    def add_training_metrics(self, metrics: Dict, metadata: Dict = None):
        """
        Add training metrics to knowledge base
        
        Args:
            metrics: Dictionary of training metrics
            metadata: Additional metadata
        """
        content = "Training Metrics:\n\n"
        
        for key, value in metrics.items():
            if isinstance(value, (list, np.ndarray)):
                content += f"{key}:\n"
                if len(value) <= 20:
                    content += f"  {value}\n"
                else:
                    content += f"  First 5: {value[:5]}\n"
                    content += f"  Last 5: {value[-5:]}\n"
                    content += f"  Best: {max(value) if 'acc' in key.lower() or 'auc' in key.lower() else min(value)}\n"
            else:
                content += f"{key}: {value}\n"
        
        if metadata is None:
            metadata = {}
        metadata['type'] = 'training_metrics'
        
        self.kb.add_document(content, metadata)
    
    def add_model_performance(self, performance: Dict, metadata: Dict = None):
        """
        Add model performance summary
        
        Args:
            performance: Performance dictionary
            metadata: Additional metadata
        """
        content = "Model Performance Summary:\n\n"
        
        for key, value in performance.items():
            content += f"{key}: {value}\n"
        
        if metadata is None:
            metadata = {}
        metadata['type'] = 'model_performance'
        
        self.kb.add_document(content, metadata)


# Initialize default knowledge base
def initialize_default_knowledge_base(kb_path: str = "knowledge_base") -> KnowledgeBase:
    """
    Initialize knowledge base with default welding knowledge
    
    Args:
        kb_path: Path to knowledge base storage
        
    Returns:
        Initialized KnowledgeBase instance
    """
    kb = KnowledgeBase(kb_path)
    
    # Check if we need to add default knowledge
    if len(kb.documents) == 0:
        print("Initializing default welding knowledge...")
        
        # Add default welding defect information
        default_knowledge = """
        # Welding Defect Types
        
        ## Cracks (CR)
        Cracks are linear discontinuities in the weld metal or base material caused by localized stress 
        exceeding the material's strength. They can be hot cracks (occurring during solidification) or 
        cold cracks (occurring after cooling).
        
        Causes:
        - High residual stresses
        - Rapid cooling rates
        - Hydrogen embrittlement
        - Poor joint design
        - Contaminated base material
        
        Severity: CRITICAL - Cracks can propagate and lead to catastrophic failure
        
        Detection Methods:
        - Visual inspection
        - Dye penetrant testing
        - Magnetic particle testing
        - Radiographic testing
        - Ultrasonic testing
        
        ## Lack of Penetration (LP)
        Lack of penetration occurs when the weld metal fails to extend through the full thickness of 
        the joint, leaving unfused areas at the root.
        
        Causes:
        - Insufficient heat input
        - Excessive welding speed
        - Improper joint preparation
        - Electrode too large
        - Incorrect electrode angle
        
        Severity: HIGH - Significantly reduces joint strength and fatigue resistance
        
        ## Porosity (PO)
        Porosity consists of gas pockets trapped in the solidified weld metal, appearing as rounded 
        or elongated cavities.
        
        Causes:
        - Contaminated base metal or filler
        - Inadequate shielding gas coverage
        - Moisture in electrode or flux
        - Excessive welding speed
        - Improper gas flow rates
        
        Severity: MODERATE to HIGH - Depends on size, distribution, and location
        
        Types:
        - Uniform porosity: Evenly distributed small pores
        - Cluster porosity: Grouped pores
        - Linear porosity: Pores in a line
        - Piping porosity: Elongated tubular cavities
        
        ## Normal Welds (ND)
        Normal welds are defect-free joints that meet all quality standards and specifications.
        
        Characteristics:
        - Complete penetration
        - Proper bead shape and size
        - No visible defects
        - Meets specified mechanical properties
        - Uniform appearance
        
        # Weld Inspection Procedures
        
        ## Visual Inspection
        The first and most common inspection method. Check for:
        - Surface cracks
        - Incomplete fusion
        - Undercut
        - Overlap
        - Porosity
        - Spatter
        - Proper bead size and shape
        
        ## Non-Destructive Testing (NDT)
        
        ### Radiographic Testing (RT)
        Uses X-rays or gamma rays to detect internal defects.
        Advantages: Permanent record, detects internal defects
        Limitations: Expensive, safety concerns, trained operators required
        
        ### Ultrasonic Testing (UT)
        Uses high-frequency sound waves to detect defects.
        Advantages: Deep penetration, portable, no radiation
        Limitations: Requires coupling medium, operator skill dependent
        
        ### Magnetic Particle Testing (MT)
        Detects surface and near-surface defects in ferromagnetic materials.
        Advantages: Simple, fast, relatively inexpensive
        Limitations: Limited to ferromagnetic materials, surface preparation required
        
        # Quality Standards
        
        ## AWS D1.1 (Structural Welding Code - Steel)
        - Defines acceptance criteria for various defect types
        - Specifies inspection methods and procedures
        - Provides quality control requirements
        
        ## ASME Section IX
        - Covers welding procedure qualifications
        - Defines welder performance qualifications
        - Specifies acceptance criteria
        
        # AI-Based Defect Detection
        
        ## Convolutional Autoencoder (CAE)
        Unsupervised learning approach that learns normal weld patterns and detects anomalies 
        based on reconstruction error.
        
        Advantages:
        - Doesn't require labeled defect data
        - Can detect novel defect types
        - Good for imbalanced datasets
        
        Metrics:
        - Reconstruction error threshold
        - ROC-AUC for anomaly detection
        
        ## CNN Classifier
        Supervised learning approach that classifies specific defect types.
        
        Advantages:
        - High accuracy for known defect types
        - Provides specific defect classification
        - Well-understood architecture
        
        Metrics:
        - Accuracy, Precision, Recall, F1-Score
        - Confusion matrix
        - Per-class performance
        
        ## Hybrid Approach
        Combining both models provides:
        - Anomaly detection (CAE) + Specific classification (CNN)
        - Better handling of novel defects
        - More robust overall system
        """
        
        kb.add_document(default_knowledge, {
            "title": "Welding Defect Detection Guide",
            "category": "welding_knowledge",
            "source": "default"
        })
        
        print("Default knowledge base initialized")
    
    return kb
