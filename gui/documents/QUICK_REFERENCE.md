# Quick Reference Guide - LLM Integration

## 🚀 Quick Start (5 minutes)

```bash
# 1. Verify environment
cd gui
python setup_llm.py

# 2. Test with sample document
python manage_kb.py upload sample_welding_procedures.txt "Welding Procedures" "procedures"

# 3. Search to verify
python manage_kb.py search "What causes cracks?"

# 4. Run GUI and try chat!
```

## 📋 Common Commands

### Knowledge Base Management
```bash
# View statistics
python manage_kb.py stats

# Upload single document
python manage_kb.py upload document.txt "Title" "category"

# Upload entire folder
python manage_kb.py upload-folder ./documents "procedures"

# Search
python manage_kb.py search "your query here"

# Interactive mode
python manage_kb.py
```

### Adding Metrics
```bash
# Add example metrics (modify script first)
python add_metrics_to_kb.py

# Add from JSON file
python add_metrics_to_kb.py metrics.json
```

## 💡 Example Queries

### About Current Detection
```
"What defects were detected?"
"How confident is the model?"
"Why is reconstruction error high?"
"Is this defect critical?"
"What should I do about this?"
```

### About Defects in General
```
"What causes cracks in welds?"
"How can I prevent porosity?"
"What's the difference between LOF and LOP?"
"Explain lack of penetration"
"What are common weld defects?"
```

### About Procedures
```
"What inspection methods should I use?"
"What are AWS D1.1 acceptance criteria for cracks?"
"How do I repair this defect?"
"What NDT methods detect cracks?"
"What documentation is required?"
```

### About Models
```
"How does the autoencoder work?"
"What's the difference between CNN and CAE?"
"What metrics indicate good performance?"
"Why use both models together?"
"What's reconstruction error?"
```

## 🔧 Programmatic Usage

### Initialize Knowledge Base
```python
from knowledge_base import initialize_default_knowledge_base

kb = initialize_default_knowledge_base("gui/knowledge_base")
```

### Add Document
```python
# From file
kb.add_document_from_file(
    "document.txt",
    metadata={"title": "...", "category": "..."}
)

# From string
kb.add_document(
    content="...",
    metadata={"title": "...", "category": "..."}
)
```

### Search Knowledge Base
```python
results = kb.search("What causes cracks?", top_k=3)

for result in results:
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Content: {result['content'][:200]}...")
```

### Add Confusion Matrix
```python
from knowledge_base import MetricsKnowledgeBase
import numpy as np

metrics_kb = MetricsKnowledgeBase(kb)

cm = np.array([[45, 2, 1, 0], [1, 42, 0, 3], [0, 1, 48, 1], [2, 1, 0, 44]])
labels = ["CR", "LP", "ND", "PO"]

metrics_kb.add_confusion_matrix(cm, labels, {"model": "CNN", "date": "2024-02-01"})
```

### Add Training Metrics
```python
metrics = {
    "train_accuracy": [0.65, 0.82, 0.91, 0.95],
    "val_accuracy": [0.70, 0.85, 0.89, 0.92],
    "train_loss": [0.85, 0.45, 0.25, 0.15],
    "val_loss": [0.75, 0.42, 0.28, 0.22]
}

metrics_kb.add_training_metrics(metrics, {"model": "CNN", "date": "2024-02-01"})
```

### Use in GUI
```python
from llm_chat import LLMChatWidget

# Create chat widget
chat_widget = LLMChatWidget(parent)

# Update with detection results
chat_widget.set_context(detection_results)

# Add metrics (after training)
chat_widget.add_metrics_to_kb(confusion_matrix, labels, metrics)
```

## 📊 Metrics JSON Format

### Confusion Matrix
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

### Training Metrics
```json
{
  "metrics": {
    "train_loss": [0.856, 0.432, 0.289],
    "val_loss": [0.623, 0.398, 0.312],
    "train_accuracy": [0.675, 0.812, 0.878],
    "val_accuracy": [0.742, 0.834, 0.871],
    "best_epoch": 10
  },
  "metadata": {
    "model": "CNN Classifier",
    "date": "2024-02-01"
  }
}
```

## 🐛 Troubleshooting

### "Azure OpenAI API error"
```bash
# Check .env file
cat .env

# Verify credentials
python setup_llm.py
```

### "No results found"
```bash
# Check KB contents
python manage_kb.py stats

# Upload documents
python manage_kb.py upload sample_welding_procedures.txt
```

### "Module not found"
```bash
# Install dependencies
pip install openai python-dotenv numpy
```

### Slow responses
- First query initializes embeddings (slower)
- Large KB increases search time
- Reduce top_k parameter in search

## 🔐 Security Checklist

- [ ] .env file NOT in git (.gitignore configured)
- [ ] API keys stored securely
- [ ] knowledge_base/ folder in .gitignore
- [ ] Regular API key rotation
- [ ] Rate limiting configured (if production)

## 📁 File Locations

```
gui/
├── llm_chat.py              ← Chat widget
├── knowledge_base.py        ← KB implementation
├── manage_kb.py             ← CLI tool
├── setup_llm.py             ← Setup verification
├── add_metrics_to_kb.py     ← Metrics helper
├── .env                     ← Credentials (DON'T COMMIT!)
└── knowledge_base/          ← Storage (auto-created)
    ├── documents.json       ← Document metadata
    └── embeddings.pkl       ← Vector embeddings
```

## 📚 Documentation Files

- `KB_README.md` - Complete documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `ARCHITECTURE.md` - System architecture
- `QUICK_REFERENCE.md` - This file

## 🎯 Recommended Workflow

### 1. Initial Setup (One-time)
```bash
cd gui
pip install openai python-dotenv
python setup_llm.py
```

### 2. Upload Your Documents
```bash
# Upload procedures
python manage_kb.py upload-folder ../documents "procedures"

# Upload standards
python manage_kb.py upload aws_d1_1.txt "AWS D1.1" "standards"
```

### 3. Add Model Metrics
```python
# After training, in your training script:
from knowledge_base import initialize_default_knowledge_base, MetricsKnowledgeBase

kb = initialize_default_knowledge_base("gui/knowledge_base")
metrics_kb = MetricsKnowledgeBase(kb)

# Add confusion matrix
metrics_kb.add_confusion_matrix(confusion_matrix, labels, metadata)

# Add training history
metrics_kb.add_training_metrics(training_metrics, metadata)
```

### 4. Use in GUI
- Run main application
- Use chat interface
- Upload additional documents as needed

### 5. Maintenance
```bash
# Periodic checks
python manage_kb.py stats

# Test search
python manage_kb.py search "test query"

# Verify setup
python setup_llm.py
```

## 💰 Cost Estimates

### Per Query (typical)
```
Embedding:  $0.0001  (query embedding)
GPT-4:      $0.002   (input + output)
Total:      ~$0.002  per query
```

### Per Document Upload
```
Embedding:  $0.001   (10 chunks @ $0.0001 each)
```

### Monthly (estimated for single user)
```
100 queries/day × 30 days = 3,000 queries
Cost: ~$6/month

10 documents/month
Cost: ~$0.01/month

Total: ~$6/month
```

## 🚨 Important Notes

1. **Embeddings are cached** - documents only embedded once
2. **KB storage is local** - no cloud storage needed
3. **Credentials in .env** - keep secure, never commit
4. **Default knowledge included** - welding defects, procedures
5. **Metrics are optional** - system works without them

## ⚡ Performance Tips

1. **Upload documents in batches** (use upload-folder)
2. **Use appropriate chunk_size** (default 500 works well)
3. **Reduce top_k for faster search** (default 3)
4. **Clear unused documents** (manage_kb.py → option 7)
5. **Use categories for organization**

## 🎓 Learning Resources

### Understanding RAG
- Retrieval-Augmented Generation combines LLMs with external knowledge
- Embeddings convert text to vectors for semantic search
- Top-K retrieval finds most relevant documents
- Context injection adds knowledge to LLM prompts

### Key Concepts
- **Embedding**: Converting text to vectors (1536 dimensions)
- **Cosine Similarity**: Measuring vector similarity (0-1)
- **Chunking**: Splitting documents for better retrieval
- **Metadata**: Additional info about documents
- **Top-K**: Retrieving K most similar results

## 🔗 Quick Links

- Azure OpenAI Docs: https://learn.microsoft.com/en-us/azure/ai-services/openai/
- OpenAI Python SDK: https://github.com/openai/openai-python
- PyQt5 Docs: https://doc.qt.io/qtforpython-5/

## ✅ Success Criteria

You've successfully set up the system when:
- [ ] `python setup_llm.py` passes all checks
- [ ] Can upload documents via GUI or CLI
- [ ] Search returns relevant results
- [ ] Chat responds with context-aware answers
- [ ] Can add and query model metrics

## 🎉 Next Steps

1. Add your organization's documents
2. Upload model training metrics
3. Test with real defect detection
4. Customize system prompts (optional)
5. Share with team (setup guide in KB_README.md)

---

**Need Help?** Check:
1. `KB_README.md` - Full documentation
2. `ARCHITECTURE.md` - System design
3. `IMPLEMENTATION_SUMMARY.md` - Implementation details
4. Run `python setup_llm.py` - Diagnose issues
