"""
LLM Chat Interface for Weld Defect Detection
Provides AI-powered assistance and explanations
"""

import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QComboBox, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor


class LLMThread(QThread):
    """Background thread for LLM API calls"""
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, provider, api_key, prompt, context):
        super().__init__()
        self.provider = provider
        self.api_key = api_key
        self.prompt = prompt
        self.context = context
        
    def run(self):
        try:
            response = self._call_openai()
            self.response_ready.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _call_openai(self):
        """Call OpenAI API"""
        try:
            import openai
            openai.api_key = self.api_key
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self.prompt}
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except ImportError:
            return "OpenAI package not installed. Run: pip install openai"
        except Exception as e:
            return f"OpenAI API error: {str(e)}"
  
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
            base_prompt += f"\n\nCurrent analysis context:\n{json.dumps(self.context, indent=2)}"
        
        return base_prompt


class LLMChatWidget(QWidget):
    """Chat widget for LLM interaction"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_context = None
        self.chat_history = []
        self.llm_thread = None
        self.api_key = ""
        
        self.initUI()
        
    def initUI(self):
        """Initialize chat UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("AI Assistant (OpenAI GPT-4)")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Courier New", 10))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.chat_display, stretch=1)
        
        # Quick questions
        quick_group = QGroupBox("Quick Questions")
        quick_layout = QVBoxLayout()
        
        quick_buttons = [
            ("What defects were detected?", "What defects were detected in this image and what do they mean?"),
            ("How severe is this?", "How severe are the detected defects and what should I do?"),
            ("Explain reconstruction error", "Can you explain what reconstruction error means and why it's important?"),
            ("Compare models", "What's the difference between the autoencoder and CNN classifier results?")
        ]
        
        for label, question in quick_buttons:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, q=question: self.send_quick_question(q))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3a3a3a;
                    border: none;
                    padding: 8px;
                    text-align: left;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
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
                padding: 10px;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                background-color: #2a2a2a;
                color: #ffffff;
            }
        """)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066cc;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0052a3;
            }
            QPushButton:disabled {
                background-color: #3a3a3a;
                color: #666666;
            }
        """)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_chat)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
        """)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        input_layout.addWidget(self.clear_btn)
        
        layout.addLayout(input_layout)
        
        # Welcome message
        self.add_system_message("Welcome! I'm your AI assistant for weld defect analysis. Ask me anything about the detection results!")
    
    def set_context(self, result):
        """Update context from detection results"""
        if not result:
            self.current_context = None
            return
            
        # Extract relevant information
        self.current_context = {
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
        
        self.add_system_message("Context updated with latest detection results.")
    
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
        
        # Disable input while processing
        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)
        
        # Start LLM thread
        self.llm_thread = LLMThread(
            "OpenAI",
            self.api_key,
            message,
            self.current_context
        )
        self.llm_thread.response_ready.connect(self.on_response_ready)
        self.llm_thread.error_occurred.connect(self.on_error_occurred)
        self.llm_thread.start()
    
    def on_response_ready(self, response):
        """Handle LLM response"""
        self.add_assistant_message(response)
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
    
    def on_error_occurred(self, error):
        """Handle LLM error"""
        self.add_system_message(f"Error: {error}")
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
    
    def add_user_message(self, message):
        """Add user message to chat"""
        self.chat_history.append({"role": "user", "content": message})
        self.chat_display.append(f'<p style="color: #00aaff;"><b>You:</b> {message}</p>')
        self.scroll_to_bottom()
    
    def add_assistant_message(self, message):
        """Add assistant message to chat"""
        self.chat_history.append({"role": "assistant", "content": message})
        self.chat_display.append(f'<p style="color: #44ff44;"><b>AI Assistant:</b> {message}</p>')
        self.scroll_to_bottom()
    
    def add_system_message(self, message):
        """Add system message to chat"""
        self.chat_display.append(f'<p style="color: #888888;"><i>{message}</i></p>')
        self.scroll_to_bottom()
    
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
