# Weld Defect Detection GUI

Professional PyQt5 GUI application for weld defect detection using combined CNN and Autoencoder models with AI-powered assistant.

## Features

- **Dual Model Detection**: Combines unsupervised (Autoencoder) and supervised (CNN) approaches
- **AI Assistant**: Ask questions about detection results using LLM integration
- **Real-time Analysis**: Fast inference with visual feedback
- **Interactive Visualization**: View original, reconstructed, and anomaly heatmap images
- **Adjustable Thresholds**: Fine-tune detection sensitivity
- **Professional UI**: Dark theme with intuitive controls and tabbed interface
- **Comprehensive Results**: Detailed confidence scores and class probabilities

## Installation

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: For AI Assistant (choose one)
# OpenAI:
pip install openai

# Anthropic Claude:
pip install anthropic

# Local Ollama (free, no API key needed):
# Install Ollama from https://ollama.ai
# Then run: ollama pull llama3.2
```

## Usage

```bash
cd gui
python main.py
```

## How It Works

1. **Upload Image**: Load a weld image for analysis
2. **Autoencoder Analysis**: Detects anomalies by reconstruction error
3. **CNN Classification**: If defect detected, classifies type (CR/LP/PO)
4. **Results Display**: Shows reconstruction, heatmap, and classification results
5. **AI Assistant**: Ask questions about the results in natural language

## AI Assistant

The AI Assistant tab provides:
- **Contextual Help**: Automatically receives detection results
- **Quick Questions**: Pre-defined questions for common queries
- **Multiple Providers**: 
  - **Local (Ollama)**: Free, runs on your machine, no API key needed
  - **OpenAI**: GPT-4 powered, requires API key
  - **Anthropic**: Claude powered, requires API key

### Setting up AI Assistant

#### Option 1: Local Ollama (Recommended for beginners)
```bash
# Install Ollama
# Download from https://ollama.ai

# Pull a model
ollama pull llama3.2

# Start Ollama (runs in background)
ollama serve
```

#### Option 2: OpenAI
1. Get API key from https://platform.openai.com
2. In GUI, select "OpenAI" provider
3. Enter your API key

#### Option 3: Anthropic
1. Get API key from https://console.anthropic.com
2. In GUI, select "Anthropic" provider
3. Enter your API key

## Model Files Required

- CNN Model: `models/best_model_pytorch.pth`
- Autoencoder: `autoencoder_models/best_autoencoder.pth`

## Defect Classes

- **CR**: Cracks
- **LP**: Lack of Penetration  
- **ND**: No Defect (Normal)
- **PO**: Porosity

## Controls

- **Autoencoder Threshold**: Adjust anomaly sensitivity
- **CNN Confidence**: Set minimum confidence for classification
- **Gaussian Blur**: Optional noise reduction preprocessing
- **Model Toggle**: Enable/disable individual models

## Keyboard Shortcuts

- `Ctrl+O`: Open image
- `Ctrl+Shift+O`: Open folder (batch mode - coming soon)
- `Ctrl+E`: Export results
- `Ctrl+Q`: Quit application
