# System Architecture - LLM Integration

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RIAWELC GUI Application                      │
│                    (Weld Defect Detection System)                   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌────────────┐  ┌──────────────┐
│   Image      │  │  Inference │  │  LLM Chat    │
│   Viewer     │  │  Engine    │  │  Widget      │
└──────────────┘  └─────┬──────┘  └──────┬───────┘
                        │                │
                        │                │
        ┌───────────────┴────────┐       │
        │                        │       │
        ▼                        ▼       │
┌──────────────┐        ┌──────────────┐│
│  CNN Model   │        │  CAE Model   ││
│  (ResNet18)  │        │ (Autoencoder)││
└──────┬───────┘        └──────┬───────┘│
       │                       │        │
       └───────────┬───────────┘        │
                   │                    │
                   ▼                    │
       ┌────────────────────┐           │
       │  Detection Results │           │
       │  - Predictions     │           │
       │  - Confidence      │◄──────────┘
       │  - Metrics         │    Context
       └────────┬───────────┘
                │
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│                      LLM Chat System                          │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              User Question / Query                   │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                    │
│         ┌───────────────┴───────────────┐                   │
│         │                               │                    │
│         ▼                               ▼                    │
│  ┌──────────────┐              ┌──────────────────┐         │
│  │  Detection   │              │  Knowledge Base  │         │
│  │  Context     │              │    (RAG System)  │         │
│  │              │              │                  │         │
│  │ - Current    │              │ ┌──────────────┐ │         │
│  │   Results    │              │ │  Documents   │ │         │
│  │ - Confidence │              │ │  - Procedures│ │         │
│  │ - Metrics    │              │ │  - Standards │ │         │
│  └──────┬───────┘              │ │  - Defect    │ │         │
│         │                      │ │    Info      │ │         │
│         │                      │ └──────┬───────┘ │         │
│         │                      │        │         │         │
│         │                      │ ┌──────▼───────┐ │         │
│         │                      │ │  Embeddings  │ │         │
│         │                      │ │  (Vectors)   │ │         │
│         │                      │ └──────┬───────┘ │         │
│         │                      │        │         │         │
│         │                      │ ┌──────▼───────┐ │         │
│         │                      │ │   Metrics    │ │         │
│         │                      │ │ - Confusion  │ │         │
│         │                      │ │   Matrix     │ │         │
│         │                      │ │ - Training   │ │         │
│         │                      │ │   History    │ │         │
│         │                      │ └──────┬───────┘ │         │
│         │                      │        │         │         │
│         │                      │        ▼         │         │
│         │                      │  Semantic Search │         │
│         │                      │  (Top-K Results) │         │
│         │                      └────────┬─────────┘         │
│         │                               │                    │
│         └───────────────┬───────────────┘                    │
│                         │                                    │
│                         ▼                                    │
│              ┌────────────────────┐                          │
│              │  Combined Context  │                          │
│              │  (Results + KB)    │                          │
│              └─────────┬──────────┘                          │
│                        │                                     │
│                        ▼                                     │
│              ┌────────────────────┐                          │
│              │   Azure OpenAI     │                          │
│              │   GPT-4            │                          │
│              └─────────┬──────────┘                          │
│                        │                                     │
│                        ▼                                     │
│              ┌────────────────────┐                          │
│              │  AI Response       │                          │
│              │  - Context-aware   │                          │
│              │  - Knowledge-based │                          │
│              │  - Actionable      │                          │
│              └────────────────────┘                          │
└───────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. GUI Application Layer
- **Image Viewer**: Display weld radiographs
- **Inference Engine**: Run CNN and CAE models
- **LLM Chat Widget**: Interactive AI assistant

### 2. Model Layer
- **CNN Classifier**: ResNet18-based supervised learning
  - Classifies defect types (CR, LP, ND, PO)
  - Outputs: Prediction + Confidence + Class probabilities
  
- **Convolutional Autoencoder**: Unsupervised anomaly detection
  - Detects anomalies via reconstruction error
  - Outputs: Is_anomaly + Reconstruction error + Threshold

### 3. LLM Chat System

#### A. Knowledge Base (RAG System)
```
┌─────────────────────────────────────┐
│      Knowledge Base Storage         │
├─────────────────────────────────────┤
│                                     │
│  Documents (documents.json)         │
│  ├─ Content (text chunks)           │
│  ├─ Metadata (title, category)      │
│  └─ Source info                     │
│                                     │
│  Embeddings (embeddings.pkl)        │
│  └─ 1536-dim vectors per chunk      │
│                                     │
│  Metrics                            │
│  ├─ Confusion matrices              │
│  ├─ Training history                │
│  └─ Model performance               │
│                                     │
└─────────────────────────────────────┘
```

#### B. RAG Workflow
```
User Query
    │
    ▼
Embed Query (Azure text-embedding-ada-002)
    │
    ▼
Compute Cosine Similarity with all document embeddings
    │
    ▼
Retrieve Top-K most similar documents (default K=3)
    │
    ▼
Extract relevant context from documents
    │
    ├─ Document 1: Similarity = 0.85
    ├─ Document 2: Similarity = 0.78
    └─ Document 3: Similarity = 0.72
    │
    ▼
Combine with Detection Context
    │
    ├─ Current image analysis results
    ├─ Model predictions and confidence
    └─ Reconstruction errors and thresholds
    │
    ▼
Create LLM Prompt
    │
    ├─ System: Role + Guidelines + Context
    ├─ Context: Detection Results + KB Documents
    └─ User: Original question
    │
    ▼
Send to Azure OpenAI GPT-4
    │
    ▼
Receive & Display Response
```

### 4. Azure OpenAI Integration

```
┌───────────────────────────────────────────────────┐
│           Azure OpenAI Services                   │
├───────────────────────────────────────────────────┤
│                                                   │
│  Embedding Model (text-embedding-ada-002)        │
│  ├─ Converts text → 1536-dim vectors            │
│  ├─ Used for semantic search                    │
│  └─ Cost: ~$0.0001 per 1K tokens               │
│                                                   │
│  Chat Model (GPT-4)                              │
│  ├─ Generates conversational responses           │
│  ├─ Context window: 8K tokens                   │
│  ├─ Input cost: ~$0.03 per 1K tokens           │
│  └─ Output cost: ~$0.06 per 1K tokens          │
│                                                   │
└───────────────────────────────────────────────────┘
                     ▲
                     │
              .env credentials
                     │
        AZURE_ENDPOINT, AZURE_API_KEY,
        CHAT_DEPLOYMENT_NAME, etc.
```

## Data Flow Examples

### Example 1: Defect Detection with AI Explanation
```
1. User uploads weld image
        ↓
2. Inference engine processes image
   - CNN: Predicts "CR" with 95% confidence
   - CAE: Detects anomaly (error = 0.025 > threshold 0.018)
        ↓
3. User asks: "What defects were detected?"
        ↓
4. LLM Chat receives:
   - Detection context (CR, 95%, anomaly detected)
   - User question
        ↓
5. System searches KB for "cracks" + "defects"
   - Finds: Crack definition, causes, severity
        ↓
6. GPT-4 receives combined prompt:
   System: "You are a welding defect expert..."
   Context: "Detection: CR (95%), Anomaly: Yes"
   KB: "Cracks are linear discontinuities..."
   User: "What defects were detected?"
        ↓
7. GPT-4 responds:
   "A crack (CR) was detected with 95% confidence.
    The autoencoder also flagged this as an anomaly
    (reconstruction error: 0.025). Cracks are critical
    defects that can propagate and cause failure..."
        ↓
8. Response displayed to user
```

### Example 2: Document Upload and Knowledge Enrichment
```
1. User clicks "Upload Document"
        ↓
2. Selects "welding_procedures.txt"
        ↓
3. System chunks document (~500 chars/chunk)
   - 10 chunks created
        ↓
4. For each chunk:
   - Send to Azure embedding API
   - Receive 1536-dim vector
   - Store in embeddings.pkl
        ↓
5. Save metadata in documents.json
   - Content, title, category, source
        ↓
6. Knowledge base updated
   - New documents available for search
   - Future queries can use this knowledge
```

### Example 3: Metrics Integration
```
1. Training completes
        ↓
2. Extract metrics:
   - Confusion matrix: [[45,2,1,0], [1,42,0,3], ...]
   - Training history: losses, accuracies
        ↓
3. Format as text with calculations:
   "Confusion Matrix Results:
    CR: Precision 0.94, Recall 0.94, F1 0.94
    Overall Accuracy: 0.942"
        ↓
4. Embed and store in KB
        ↓
5. User can now ask:
   "What's the model accuracy?"
   "How well does it detect cracks?"
   "Compare precision across defect types"
        ↓
6. LLM retrieves metrics and explains
```

## File Architecture

```
RIAWELC/
├── gui/
│   ├── main.py                      # Main application
│   ├── llm_chat.py                  # Chat widget with RAG
│   ├── knowledge_base.py            # KB implementation
│   ├── manage_kb.py                 # CLI management
│   ├── setup_llm.py                 # Setup verification
│   ├── add_metrics_to_kb.py         # Metrics helper
│   │
│   ├── .env                         # Azure credentials
│   ├── requirements.txt             # Dependencies
│   │
│   ├── knowledge_base/              # KB storage
│   │   ├── documents.json           # Document metadata
│   │   └── embeddings.pkl           # Vector embeddings
│   │
│   ├── KB_README.md                 # Documentation
│   ├── IMPLEMENTATION_SUMMARY.md    # This implementation
│   └── ARCHITECTURE.md              # This file
│
├── models/
│   ├── best_model_pytorch.pth       # CNN model
│   └── ...
│
└── cae/
    └── models/
        └── best_model.pth           # CAE model
```

## Technology Stack

```
┌─────────────────────────────────────────────┐
│           Application Layer                 │
│  - PyQt5 (GUI framework)                   │
│  - Python 3.x                              │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│           ML Framework Layer                │
│  - PyTorch (model inference)               │
│  - torchvision (image processing)          │
│  - NumPy (numerical computing)             │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│           LLM & Embedding Layer             │
│  - Azure OpenAI (GPT-4)                    │
│  - text-embedding-ada-002                  │
│  - openai Python SDK                       │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│           Storage Layer                     │
│  - JSON (document metadata)                │
│  - Pickle (embedding vectors)              │
│  - File system (documents)                 │
└─────────────────────────────────────────────┘
```

## Security Architecture

```
┌──────────────────────────────────────────┐
│     Application Environment              │
│                                          │
│  ┌────────────────────────────────┐     │
│  │  .env file (NOT in git)        │     │
│  │  - AZURE_API_KEY               │     │
│  │  - AZURE_ENDPOINT              │     │
│  │  - Deployment names            │     │
│  └────────────────────────────────┘     │
│              │                           │
│              ▼                           │
│  ┌────────────────────────────────┐     │
│  │  python-dotenv                 │     │
│  │  Loads env vars at runtime     │     │
│  └────────────────────────────────┘     │
│              │                           │
│              ▼                           │
│  ┌────────────────────────────────┐     │
│  │  Azure OpenAI SDK              │     │
│  │  Handles secure communication  │     │
│  └────────────────────────────────┘     │
└──────────────────┬───────────────────────┘
                   │
                   │ HTTPS
                   │
                   ▼
         ┌──────────────────┐
         │   Azure OpenAI   │
         │   Service        │
         │   (Cloud)        │
         └──────────────────┘
```

## Scalability Considerations

### Current Implementation
- Local storage (JSON + Pickle)
- In-memory search (NumPy cosine similarity)
- Single-user application

### Future Scaling Options

```
┌─────────────────────────────────────────────┐
│        For Enterprise Deployment            │
├─────────────────────────────────────────────┤
│                                             │
│  Knowledge Base:                            │
│  └─ Azure Cognitive Search                 │
│     (Vector search at scale)               │
│                                             │
│  Storage:                                   │
│  ├─ Azure Blob Storage (documents)         │
│  └─ Azure Cosmos DB (metadata)             │
│                                             │
│  Credentials:                               │
│  └─ Azure Key Vault                        │
│                                             │
│  Caching:                                   │
│  └─ Azure Cache for Redis                  │
│                                             │
│  Multi-user:                                │
│  └─ Web API (FastAPI/Flask)                │
│     + Authentication (Azure AD)            │
│                                             │
└─────────────────────────────────────────────┘
```

## Performance Metrics

### Typical Response Times
```
Component                    Time
─────────────────────────────────────
Document Upload:
  - File read                ~10ms
  - Chunking                 ~50ms
  - Embedding (per chunk)    ~200ms
  - Save to disk             ~10ms
  Total (10 chunks):         ~2.3s

Knowledge Base Search:
  - Query embedding          ~200ms
  - Cosine similarity        ~50ms
  - Retrieve top-K           ~10ms
  Total:                     ~260ms

LLM Response:
  - Prepare prompt           ~10ms
  - GPT-4 generation         ~2-5s
  - Parse response           ~10ms
  Total:                     ~2-5s

Complete Query:              ~2.5-5.5s
```

### Storage Requirements
```
Component              Size (typical)
──────────────────────────────────────
Embedding (1 document chunk):
  - Text (500 chars)         ~500 bytes
  - Vector (1536 floats)     ~6 KB
  - Metadata                 ~200 bytes
  Total per chunk:           ~7 KB

Knowledge Base (100 documents):
  - documents.json           ~50 KB
  - embeddings.pkl           ~700 KB
  Total:                     ~750 KB

Models:
  - CNN (ResNet18)           ~45 MB
  - CAE                      ~8 MB
```

## Error Handling Flow

```
User Query
    │
    ├─ Invalid credentials? → Show error + setup guide
    ├─ No KB results?       → Use detection context only
    ├─ API timeout?         → Retry with exponential backoff
    ├─ Rate limit?          → Queue request, inform user
    └─ Success              → Display response
```

## Summary

This architecture provides:
- ✅ Modular, maintainable design
- ✅ Separation of concerns
- ✅ Scalable knowledge base
- ✅ Secure credential management
- ✅ Context-aware AI responses
- ✅ Extensible for future features
