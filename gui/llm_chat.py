"""
LLM Chat Interface for Weld Defect Detection
Provides AI-powered assistance and explanations using Azure OpenAI
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QComboBox, QGroupBox, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor

from knowledge_base import KnowledgeBase, MetricsKnowledgeBase, initialize_default_knowledge_base

# Load environment variables
load_dotenv()


class LLMThread(QThread):
    """Background thread for LLM API calls with Azure OpenAI"""
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, prompt, context, knowledge_base=None):
        super().__init__()
        self.prompt = prompt
        self.context = context
        self.knowledge_base = knowledge_base
        
        # Azure OpenAI setup
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT")
        )
        self.deployment_name = os.getenv("CHAT_DEPLOYMENT_NAME", "gpt-4")
        
    def run(self):
        try:
            response = self._call_azure_openai()
            self.response_ready.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _call_azure_openai(self):
        """Call Azure OpenAI API with RAG"""
        try:
            # Get relevant context from knowledge base
            kb_context = ""
            if self.knowledge_base:
                results = self.knowledge_base.search(self.prompt, top_k=3)
                if results:
                    kb_context = "\n\nRelevant Knowledge Base Information:\n"
                    for i, result in enumerate(results, 1):
                        kb_context += f"\n[Source {i} - Similarity: {result['similarity']:.3f}]\n"
                        kb_context += result['content'][:500] + "...\n"
            
            messages = [
                {"role": "system", "content": self._get_system_prompt() + kb_context},
                {"role": "user", "content": self.prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Azure OpenAI API error: {str(e)}"

  
    def _get_system_prompt(self):
        """Get system prompt with context"""
        base_prompt = """You are an AI assistant specialized in welding defect detection and analysis. 
You help users understand defect detection results, explain different types of defects, and provide 
recommendations for quality control.

Common defect types:
- CR (Cracks): Linear discontinuities caused by stress or thermal effects
- LP (Lack of Penetration): Incomplete fusion at weld root
- PO (Porosity): Gas pockets trapped in weld metal
- ND (No Defect): Normal, acceptable weld

The system uses two AI models:
1. Convolutional Autoencoder (unsupervised): Detects anomalies based on reconstruction error
2. CNN Classifier (supervised): Classifies specific defect types

Keep responses concise, technical but accessible, and focused on practical implications."""

        if self.context:
            # Check if batch analysis data is present
            if self.context.get('batch_analysis') or self.context.get('total_images'):
                base_prompt += f"\n\n=== BATCH ANALYSIS RESULTS ===\n"
                base_prompt += f"Total Images Processed: {self.context.get('total_images', 0)}\n"
                base_prompt += f"Defects Found: {self.context.get('defects_found', 0)}\n"
                base_prompt += f"Overall Accuracy: {self.context.get('accuracy', 0):.2%}\n"
                base_prompt += f"CNN Accuracy: {self.context.get('cnn_accuracy', 0):.2%}\n"
                base_prompt += f"Hybrid Model Accuracy: {self.context.get('hybrid_accuracy', 0):.2%}\n"
                
                # Add per-class metrics if available and properly formatted
                cnn_metrics = self.context.get('cnn_metrics')
                if cnn_metrics and isinstance(cnn_metrics, dict):
                    precision = cnn_metrics.get('precision', {})
                    recall = cnn_metrics.get('recall', {})
                    f1 = cnn_metrics.get('f1', {})
                    
                    if isinstance(precision, dict) and isinstance(recall, dict) and isinstance(f1, dict):
                        base_prompt += f"\nPer-Class Metrics (CNN):\n"
                        for class_name in ['CR', 'LP', 'ND', 'PO']:
                            if class_name in precision and class_name in recall and class_name in f1:
                                base_prompt += f"  {class_name}: Precision={precision[class_name]:.2%}, "
                                base_prompt += f"Recall={recall[class_name]:.2%}, "
                                base_prompt += f"F1={f1[class_name]:.2%}\n"
                
                # Add class distribution if available
                class_dist = self.context.get('class_distribution')
                if class_dist and isinstance(class_dist, dict):
                    base_prompt += f"\nClass Distribution:\n"
                    for class_name, count in class_dist.items():
                        base_prompt += f"  {class_name}: {count} images\n"
            
            # Show current image if available
            if self.context.get('current_image'):
                base_prompt += f"\n\n=== CURRENT IMAGE ANALYSIS ===\n"
                base_prompt += json.dumps(self.context['current_image'], indent=2)
            elif not self.context.get('batch_analysis') and not self.context.get('total_images'):
                # Only single image analysis
                base_prompt += f"\n\n=== CURRENT IMAGE ANALYSIS ===\n"
                base_prompt += json.dumps(self.context, indent=2)
        
        return base_prompt


class LLMChatWidget(QWidget):
    """Chat widget for LLM interaction with RAG support"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_context = None
        self.batch_context = None  # Store batch results separately
        self.chat_history = []
        self.llm_thread = None
        
        # Initialize knowledge base
        kb_path = Path(__file__).parent / "knowledge_base"
        self.knowledge_base = initialize_default_knowledge_base(str(kb_path))
        self.metrics_kb = MetricsKnowledgeBase(self.knowledge_base)
        
        self.initUI()
        
    def initUI(self):
        """Initialize chat UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("AI Assistant (Azure OpenAI GPT-4)")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 10))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #0e1117;
                color: #ffffff;
                border: 1px solid #2d333b;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        # Enable rich text and word wrap
        self.chat_display.setAcceptRichText(True)
        self.chat_display.setLineWrapMode(QTextEdit.WidgetWidth)
        layout.addWidget(self.chat_display, stretch=1)
        
        # Quick questions
        quick_group = QGroupBox(" Quick Questions")
        quick_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #374151;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #111827;
            }
            QGroupBox::title {
                color: #f3f4f6;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        quick_layout = QVBoxLayout()
        
        quick_buttons = [
            (" What defects were detected?", "What defects were detected in this image and what do they mean?"),
            (" How severe is this?", "How severe are the detected defects and what should I do?"),
            (" Explain reconstruction error", "Can you explain what reconstruction error means and why it's important?"),
            (" Compare models", "What's the difference between the autoencoder and CNN classifier results?"),
            (" What causes these defects?", "What are the common causes of the detected defects?"),
            (" Inspection procedures", "What inspection procedures should be followed for these defects?")
        ]
        
        for label, question in quick_buttons:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, q=question: self.send_quick_question(q))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #1f2937;
                    color: #d1d5db;
                    border: 1px solid #374151;
                    padding: 10px 12px;
                    text-align: left;
                    border-radius: 8px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #374151;
                    color: #ffffff;
                    border: 1px solid #4b5563;
                }
                QPushButton:pressed {
                    background-color: #1f2937;
                    border: 1px solid #3b82f6;
                }
            """)
            quick_layout.addWidget(btn)
        
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask a question about the defect detection results...")
        self.input_field.returnPressed.connect(self.send_message)
        self.input_field.setStyleSheet("""
            QLineEdit {
                padding: 14px 16px;
                border: 2px solid #374151;
                border-radius: 24px;
                background-color: #1f2937;
                color: #ffffff;
                font-size: 14px;
                selection-background-color: #2563eb;
            }
            QLineEdit:focus {
                border: 2px solid #3b82f6;
                background-color: #111827;
            }
            QLineEdit::placeholder {
                color: #6b7280;
            }
        """)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #3b82f6, stop:1 #2563eb);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 24px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #2563eb, stop:1 #1d4ed8);
            }
            QPushButton:pressed {
                background: #1e40af;
            }
            QPushButton:disabled {
                background: #374151;
                color: #6b7280;
            }
        """)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_chat)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #374151;
                color: #d1d5db;
                border: 1px solid #4b5563;
                padding: 12px 20px;
                border-radius: 24px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #ef4444;
                color: white;
                border: 1px solid #dc2626;
            }
            QPushButton:pressed {
                background-color: #dc2626;
            }
        """)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        input_layout.addWidget(self.clear_btn)
        
        layout.addLayout(input_layout)
        
        # Welcome message
        self.add_system_message("Welcome! I'm your AI assistant for weld defect analysis powered by Azure OpenAI.")
        # self.add_system_message(f"Knowledge base loaded with {len(self.knowledge_base.documents)} documents.")
    
    def add_metrics_to_kb(self, confusion_matrix, labels, metrics):
        """
        Add model metrics to knowledge base
        
        Args:
            confusion_matrix: numpy array
            labels: list of class labels
            metrics: dict of training/evaluation metrics
        """
        try:
            import numpy as np
            
            # Add confusion matrix
            if confusion_matrix is not None:
                self.metrics_kb.add_confusion_matrix(
                    confusion_matrix, 
                    labels,
                    metadata={"date": str(Path(__file__).stat().st_mtime)}
                )
            
            # Add other metrics
            if metrics:
                self.metrics_kb.add_training_metrics(metrics)
            
        except Exception as e:
            print(f"Error adding metrics to KB: {e}")
    
    def set_context(self, result):
        """Update context from detection results"""
        if not result:
            return
            
        # Check if this is batch analysis results
        if result.get('batch_analysis'):
            # Store batch results separately - don't overwrite
            self.batch_context = result
            self.current_context = result
            self.add_system_message(f"Batch analysis complete: {result.get('total_images', 0)} images processed with {result.get('accuracy', 0)*100:.1f}% accuracy.")
        else:
            # Single image result - merge with batch context if exists
            single_image_context = {
                "autoencoder": {
                    "is_anomaly": result.get("is_anomaly", False),
                    "reconstruction_error": result.get("reconstruction_error", 0),
                    "threshold": result.get("ae_threshold", 0)
                },
                "cnn": {
                    "prediction": result.get("cnn_prediction", "Unknown"),
                    "confidence": result.get("cnn_confidence", 0),
                    "probabilities": result.get("cnn_probabilities", {})
                },
                "final_decision": result.get("final_decision", "Unknown"),
                "processing_time": result.get("processing_time", 0)
            }
            
            # If we have batch context, include it in system prompt
            if self.batch_context:
                self.current_context = {**self.batch_context, "current_image": single_image_context}
            else:
                self.current_context = single_image_context
    
    def send_quick_question(self, question):
        """Send a quick question"""
        self.input_field.setText(question)
        self.send_message()
    
    def send_message(self):
        """Send user message"""
        message = self.input_field.text().strip()
        if not message:
            return
        
        # Add user message to display
        self.add_user_message(message)
        self.input_field.clear()
        
        # Show typing indicator
        self.add_typing_indicator()
        
        # Disable input while processing
        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)
        self.send_btn.setText("Thinking...")
        
        # Start LLM thread with knowledge base
        self.llm_thread = LLMThread(
            message,
            self.current_context,
            self.knowledge_base
        )
        self.llm_thread.response_ready.connect(self.on_response_ready)
        self.llm_thread.error_occurred.connect(self.on_error_occurred)
        self.llm_thread.start()
    
    def on_response_ready(self, response):
        """Handle LLM response"""
        # Remove typing indicator
        self.remove_typing_indicator()
        
        self.add_assistant_message(response)
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.send_btn.setText("Send")
    
    def on_error_occurred(self, error):
        """Handle LLM error"""
        # Remove typing indicator
        self.remove_typing_indicator()
        
        self.add_system_message(f"❌ Error: {error}")
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.send_btn.setText("Send ➤")
    
    def add_typing_indicator(self):
        """Add typing indicator animation"""
        typing_html = '''
        <div id="typing-indicator" style="margin: 10px 0; text-align: left;">
            <div style="display: inline-block; padding: 12px 16px;
                        background: #1f2937; border: 1px solid #374151;
                        border-radius: 18px 18px 18px 4px;">
                <span style="color: #10b981; font-weight: 600; font-size: 11px;">AI ASSISTANT</span><br>
                <span style="color: #9ca3af; font-size: 14px;">
                    <span style="animation: blink 1.4s infinite; animation-delay: 0s;">●</span>
                    <span style="animation: blink 1.4s infinite; animation-delay: 0.2s;">●</span>
                    <span style="animation: blink 1.4s infinite; animation-delay: 0.4s;">●</span>
                </span>
            </div>
        </div>
        '''
        self.chat_display.append(typing_html)
        self.scroll_to_bottom()
    
    def remove_typing_indicator(self):
        """Remove typing indicator"""
        # Get current HTML and remove typing indicator
        html = self.chat_display.toHtml()
        if 'id="typing-indicator"' in html:
            # Remove the typing indicator div
            import re
            html = re.sub(r'<div id="typing-indicator".*?</div>\s*</div>', '', html, flags=re.DOTALL)
            self.chat_display.setHtml(html)
            self.scroll_to_bottom()
    
    def add_user_message(self, message):
        """Add user message to chat with modern bubble style"""
        self.chat_history.append({"role": "user", "content": message})
        
        # Escape HTML to prevent rendering issues
        from html import escape
        message_html = escape(message).replace('\n', '<br>')
        
        bubble_html = f'''
        <div style="margin: 10px 0; text-align: right;">
            <div style="display: inline-block; max-width: 80%; text-align: left; 
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                   stop:0 #2563eb, stop:1 #1d4ed8);
                        color: white; padding: 12px 16px; border-radius: 18px 18px 4px 18px;
                        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
                        font-size: 14px; line-height: 1.5;">
                <div style="font-weight: 600; font-size: 11px; opacity: 0.9; margin-bottom: 4px;">
                    YOU
                </div>
                <div>{message_html}</div>
            </div>
        </div>
        '''
        self.chat_display.append(bubble_html)
        self.scroll_to_bottom()
    
    def add_assistant_message(self, message):
        """Add assistant message to chat with modern bubble style"""
        self.chat_history.append({"role": "assistant", "content": message})
        
        # Escape HTML but preserve markdown-like formatting
        from html import escape
        message_html = escape(message)
        
        # Convert markdown-style formatting
        message_html = self._format_markdown(message_html)
        
        bubble_html = f'''
        <div style="margin: 10px 0; text-align: left;">
            <div style="display: inline-block; max-width: 85%; text-align: left;
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                   stop:0 #1f2937, stop:1 #111827);
                        color: #e5e7eb; padding: 12px 16px; border-radius: 18px 18px 18px 4px;
                        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
                        border: 1px solid #374151;
                        font-size: 14px; line-height: 1.6;">
                <div style="font-weight: 600; font-size: 11px; color: #10b981; margin-bottom: 4px;">
                    AI ASSISTANT
                </div>
                <div>{message_html}</div>
            </div>
        </div>
        '''
        self.chat_display.append(bubble_html)
        self.scroll_to_bottom()
    
    def add_system_message(self, message):
        """Add system message to chat"""
        from html import escape
        message_html = escape(message).replace('\n', '<br>')
        
        system_html = f'''
        <div style="margin: 8px 0; text-align: center;">
            <div style="display: inline-block; padding: 6px 12px; 
                        background: rgba(107, 114, 128, 0.15);
                        color: #9ca3af; font-size: 12px; border-radius: 12px;
                        border: 1px solid rgba(107, 114, 128, 0.2);">
                <i>{message_html}</i>
            </div>
        </div>
        '''
        self.chat_display.append(system_html)
        self.scroll_to_bottom()
    
    def _format_markdown(self, text):
        """Convert basic markdown formatting to HTML"""
        import re
        
        # Bold: **text** or __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
        
        # Italic: *text* or _text_
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
        text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<i>\1</i>', text)
        
        # Inline code: `code`
        text = re.sub(r'`([^`]+)`', r'<code style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 13px;">\1</code>', text)
        
        # Line breaks
        text = text.replace('\n', '<br>')
        
        # Lists (basic support)
        text = re.sub(r'<br>[-•]\s', r'<br>&nbsp;&nbsp;• ', text)
        
        # Highlight numbers and percentages
        text = re.sub(r'\b(\d+\.?\d*%)\b', r'<span style="color: #fbbf24; font-weight: 600;">\1</span>', text)
        text = re.sub(r'\b(\d+\.\d+)\b', r'<span style="color: #60a5fa;">\1</span>', text)
        
        return text
    
    def scroll_to_bottom(self):
        """Scroll chat to bottom"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)
    
    def clear_chat(self):
        """Clear chat history"""
        self.chat_display.clear()
        self.chat_history = []
        self.add_system_message("Chat cleared. How can I help you?")
