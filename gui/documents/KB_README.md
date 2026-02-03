# LLM & Knowledge Base Integration

This module integrates Azure OpenAI with a Retrieval-Augmented Generation (RAG) system for the weld defect detection GUI.

## Features

- **Azure OpenAI Integration**: Uses GPT-4 for intelligent Q&A
- **Embedding-based Knowledge Base**: Semantic search using Azure text-embedding-ada-002
- **RAG (Retrieval-Augmented Generation)**: Combines LLM with domain-specific knowledge
- **Document Upload**: Upload welding procedures, standards, and technical documents
- **Metrics Integration**: Store and query confusion matrices and training metrics
- **Offline Management**: Command-line tools for bulk document uploads

## Setup

### 1. Environment Variables

Create or update the `.env` file in the `gui` folder:


```

### 2. Install Dependencies

```bash
cd gui
pip install -r requirements.txt
```

Required packages:
- `openai>=1.0.0` - Azure OpenAI SDK
- `python-dotenv>=0.19.0` - Environment variable management
- `numpy` - For embeddings and metrics

### 3. Initialize Knowledge Base

The knowledge base is automatically initialized with default welding knowledge when first run. It creates:
- `gui/knowledge_base/` - Storage directory
- `gui/knowledge_base/documents.json` - Document metadata
- `gui/knowledge_base/embeddings.pkl` - Vector embeddings

## Usage

### In GUI

1. **Chat Interface**:
   - Ask questions about defect detection results
   - Get explanations about welding defects
   - Query inspection procedures and standards

2. **Upload Documents**:
   - Click "Upload Document" button
   - Select `.txt`, `.md`, or `.json` files
   - Add title and category (optional)
   - Documents are automatically embedded and indexed

3. **Context-Aware Responses**:
   - LLM automatically receives current detection results
   - Combines with knowledge base for comprehensive answers

### Command-Line Management

Use `manage_kb.py` for offline document management:

```bash
# Interactive mode
python manage_kb.py

# Show statistics
python manage_kb.py stats

# Upload single document
python manage_kb.py upload document.txt "Welding Standards" "standards"

# Upload entire folder
python manage_kb.py upload-folder ./documents/procedures "welding_procedures"

# Search knowledge base
python manage_kb.py search "What causes porosity in welds?"
```

### Adding Confusion Matrix

Create a JSON file with your confusion matrix:

```json
{
  "matrix": [
    [45, 2, 1, 0],
    [1, 42, 0, 3],
    [0, 1, 48, 1],
    [2, 1, 0, 44]
  ],
  "labels": ["CR", "LP", "ND", "PO"],
  "metadata": {
    "model": "CNN Classifier",
    "date": "2024-02-01",
    "accuracy": 0.9424
  }
}
```

Then use the interactive menu (option 3) or programmatically:

```python
from knowledge_base import initialize_default_knowledge_base, MetricsKnowledgeBase
import numpy as np

kb = initialize_default_knowledge_base("knowledge_base")
metrics_kb = MetricsKnowledgeBase(kb)

# Add confusion matrix
cm = np.array([[45, 2, 1, 0], [1, 42, 0, 3], [0, 1, 48, 1], [2, 1, 0, 44]])
labels = ["CR", "LP", "ND", "PO"]
metrics_kb.add_confusion_matrix(cm, labels, {"model": "CNN", "date": "2024-02-01"})
```

### Adding Training Metrics

Create a JSON file with training metrics:

```json
{
  "metrics": {
    "train_loss": [0.856, 0.432, 0.289, 0.198],
    "val_loss": [0.623, 0.398, 0.312, 0.267],
    "train_accuracy": [0.675, 0.812, 0.878, 0.923],
    "val_accuracy": [0.742, 0.834, 0.871, 0.892]
  },
  "metadata": {
    "model": "CNN Classifier",
    "date": "2024-02-01"
  }
}
```

Use interactive menu (option 4) or programmatically.

## Document Structure

### Recommended Categories

- `welding_procedures` - Welding techniques and procedures
- `defect_info` - Detailed defect descriptions and causes
- `standards` - AWS, ASME, ISO standards
- `inspection` - NDT procedures and guidelines
- `materials` - Material properties and specifications
- `quality_control` - QA/QC procedures

### Document Chunking

Documents are automatically chunked into ~500 character segments for better retrieval:
- Preserves paragraph boundaries
- Maintains context within chunks
- Each chunk is embedded separately

## RAG System Architecture

```
User Query
    ↓
Embedding Model (text-embedding-ada-002)
    ↓
Vector Search (Cosine Similarity)
    ↓
Top-K Relevant Documents
    ↓
Combined with Detection Context
    ↓
Azure OpenAI GPT-4
    ↓
Response to User
```

## Example Queries

- "What defects were detected and how severe are they?"
- "Explain the difference between lack of penetration and lack of fusion"
- "What are the acceptance criteria for porosity in AWS D1.1?"
- "Why is my reconstruction error high?"
- "What inspection methods should I use for detected cracks?"
- "Compare the CNN and autoencoder predictions"
- "What are common causes of the detected defects?"

## Knowledge Base API

### KnowledgeBase Class

```python
from knowledge_base import KnowledgeBase

# Initialize
kb = KnowledgeBase("knowledge_base")

# Add document
kb.add_document(content, metadata={"title": "...", "category": "..."})

# Add from file
kb.add_document_from_file("document.txt", metadata={...})

# Search
results = kb.search("query", top_k=3)

# Statistics
stats = kb.get_statistics()

# Clear
kb.clear()
```

### MetricsKnowledgeBase Class

```python
from knowledge_base import MetricsKnowledgeBase

metrics_kb = MetricsKnowledgeBase(kb)

# Add confusion matrix
metrics_kb.add_confusion_matrix(cm_array, labels, metadata)

# Add training metrics
metrics_kb.add_training_metrics(metrics_dict, metadata)

# Add performance summary
metrics_kb.add_model_performance(performance_dict, metadata)
```

## Best Practices

1. **Document Organization**:
   - Use consistent categories
   - Add descriptive titles
   - Include source information in metadata

2. **Chunking**:
   - Default 500 chars works well for most documents
   - Increase for highly technical content
   - Decrease for dense information

3. **Metrics Storage**:
   - Upload confusion matrices after each training run
   - Store training metrics with clear metadata
   - Include date, model version, and dataset info

4. **Search Quality**:
   - More documents = better retrieval
   - Keep documents focused and specific
   - Remove duplicate or outdated information

## Troubleshooting

### "Azure OpenAI API error"
- Check `.env` file exists and has correct values
- Verify API key is valid
- Check deployment names match your Azure setup

### "No results found"
- Knowledge base may be empty
- Try broader search queries
- Check if documents were uploaded successfully

### "Error getting embedding"
- Check Azure OpenAI endpoint is reachable
- Verify embedding deployment name
- Check API quota and rate limits

### Slow Response
- Large knowledge base can slow embedding search
- Consider filtering by category
- Reduce top_k results

## File Structure

```
gui/
├── llm_chat.py              # Main chat widget
├── knowledge_base.py        # KB implementation
├── manage_kb.py             # CLI management tool
├── .env                     # Azure credentials
├── requirements.txt         # Dependencies
├── knowledge_base/          # Storage directory
│   ├── documents.json       # Document metadata
│   └── embeddings.pkl       # Vector embeddings
├── example_confusion_matrix.json
└── example_training_metrics.json
```

## Security Notes

- **Never commit `.env` file** - Add to `.gitignore`
- Keep API keys secure
- Rotate keys periodically
- Use Azure Key Vault for production
- Implement rate limiting for production use

## Future Enhancements

- [ ] Support for PDF documents
- [ ] Image-based knowledge (GradCAM examples)
- [ ] Multi-language support
- [ ] Advanced filtering and faceted search
- [ ] Knowledge base versioning
- [ ] Export/import functionality
- [ ] Integration with Azure Cognitive Search
