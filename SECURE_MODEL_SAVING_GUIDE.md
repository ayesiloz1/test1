# Secure Model Saving Guide

## ✅ Automatic Security (New Method)

Your trained models **now automatically produce secure .pth files** with integrity verification!

### What Changed?

**Before (Manual - 2 steps):**
```python
# Step 1: Save model
torch.save(model.state_dict(), 'model.pth')

# Step 2: Generate hash manually
verifier = ModelIntegrityVerifier()
verifier.save_model_with_verification('model.pth')
```

**After (Automatic - 1 step):**
```python
# Single call does both - saves model + generates hash
verifier = ModelIntegrityVerifier()
verifier.save_pytorch_model(model.state_dict(), 'model.pth')
```

---

## 🎯 How It Works Now

### 1. CNN Training (`train_pytorch.py`)

When you train your CNN model, it **automatically**:
- ✅ Saves the best model during training
- ✅ Generates SHA-256 integrity hash
- ✅ Creates `.pth.hash` file alongside `.pth`

**Output:**
```
Epoch 10: Val Acc: 96.5%
✓ Best model saved securely! (Val Acc: 96.5%)
✓ Model saved: models/best_model_pytorch.pth
✓ Integrity hash generated: models/best_model_pytorch.pth.hash
```

### 2. CAE Training (`cae/src/train.py`)

Same automatic behavior:
```
Validation AUC: 0.9845
✓ Model saved securely: cae/models/best_model.pth
✓ Integrity hash: cae/models/best_model.pth.hash
```

---

## 📖 New API Reference

### `save_pytorch_model(model_or_state_dict, model_path, is_state_dict=True)`

**Recommended method for all PyTorch model saving.**

**Parameters:**
- `model_or_state_dict`: Model state_dict (recommended) or full model
- `model_path`: Path where model will be saved
- `is_state_dict`: True if saving state_dict, False if saving full model

**Returns:**
- `True` if successful (model + hash saved)
- `False` if failed

**Examples:**

```python
from gui.security_utils import ModelIntegrityVerifier

verifier = ModelIntegrityVerifier()

# Example 1: Save state_dict (recommended)
verifier.save_pytorch_model(model.state_dict(), 'my_model.pth')

# Example 2: Save full model
verifier.save_pytorch_model(model, 'my_model.pth', is_state_dict=False)

# Example 3: Save checkpoint with metadata
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss
}
verifier.save_pytorch_model(checkpoint, 'checkpoint.pth', is_state_dict=False)
```

---

## 🔒 What Gets Generated?

### Model File (`.pth`)
Your standard PyTorch model file - unchanged format.

### Hash File (`.pth.hash`)
JSON file containing integrity information:

```json
{
  "file": "best_model_pytorch.pth",
  "hash": "a3f8b9c2d1e4f5a6b7c8d9e0f1a2b3c4...",
  "algorithm": "sha256"
}
```

**This hash is automatically verified when you load the model in production!**

---

## ✅ Verification (Automatic)

When you load a model using the inference engine:

```python
# gui/inference_engine.py automatically verifies before loading
model = self._load_cnn()  # Internally checks hash before loading
```

**Output:**
```
✓ Model integrity verified: models/best_model_pytorch.pth
Model loaded successfully
```

---

## 🚨 Security Benefits

### 1. **Tampering Detection**
If someone modifies the `.pth` file (malicious injection, corruption), loading will fail:
```
✗ Model integrity verification failed!
  Expected: a3f8b9c2...
  Got: b4e9c0d3...
  File may be corrupted or tampered with!
```

### 2. **No Manual Steps**
- Every model save automatically generates hash
- Impossible to forget - it's built-in
- Production-ready from day 1

### 3. **Audit Trail**
- Hash files serve as audit records
- Can verify model integrity anytime
- Compliance-ready (ISO, SOC2, etc.)

---

## 📝 Migration Guide

### If You Have Existing Training Scripts

**Option A: Update to new API (Recommended)**

```python
# Old code
torch.save(model.state_dict(), 'model.pth')

# New code  
from gui.security_utils import ModelIntegrityVerifier
verifier = ModelIntegrityVerifier()
verifier.save_pytorch_model(model.state_dict(), 'model.pth')
```

**Option B: Generate hashes for existing models**

```bash
# Run the hash generator utility
python generate_model_hashes.py
```

---

## 🧪 Testing

Verify that your models have integrity hashes:

```bash
# Run security verification
python verify_security.py
```

**Expected output:**
```
Running Security Verification...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Model Integrity Hashes
✓ Secure Model Loading
✓ File Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All security checks passed! ✓
```

---

## 📊 Where This Applies

### ✅ Updated Files (Automatic Hash Generation):
1. ✅ `train_pytorch.py` - CNN training
2. ✅ `cae/src/train.py` - Autoencoder training
3. ✅ `gui/inference_engine.py` - Model loading with verification
4. ✅ `gui/security_utils.py` - New `save_pytorch_model()` function

### Files Using Verification:
- `gui/main.py` - GUI application
- `gui/inference_engine.py` - AI inference
- Any future training scripts

---

## 💡 Best Practices

### ✅ DO:
- Use `save_pytorch_model()` for all new code
- Keep `.pth` and `.pth.hash` files together
- Commit hash files to version control
- Verify models in CI/CD pipelines

### ❌ DON'T:
- Delete `.hash` files (breaks verification)
- Use plain `torch.save()` for production models
- Share models without hash files
- Modify `.pth` files manually

---

## 🎓 Example: Complete Training Loop

```python
import torch
import torch.nn as nn
from gui.security_utils import ModelIntegrityVerifier

# Initialize
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
verifier = ModelIntegrityVerifier()

# Training loop
for epoch in range(num_epochs):
    # Train model
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_acc = validate(model, val_loader)
    
    # Save best model securely (automatic hash generation)
    if val_acc > best_acc:
        best_acc = val_acc
        success = verifier.save_pytorch_model(
            model.state_dict(), 
            'models/best_model.pth'
        )
        if success:
            print(f"✓ Best model saved with integrity verification")
        else:
            print(f"✗ Failed to save model securely")

# Save final model
verifier.save_pytorch_model(
    {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc
    },
    'models/final_model.pth',
    is_state_dict=False  # Saving dict with metadata
)

print("Training complete - all models secured with integrity hashes!")
```

---

## 🔍 Troubleshooting

### "Security utils not available"
**Problem:** `security_utils.py` not found  
**Solution:** Ensure `gui/security_utils.py` exists, or add to Python path:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'gui'))
from security_utils import ModelIntegrityVerifier
```

### "Model integrity verification failed"
**Problem:** Hash mismatch  
**Solutions:**
1. Regenerate hash: `python generate_model_hashes.py`
2. Check for file corruption
3. Verify correct model file

### "No .hash file found"
**Problem:** Old model without hash  
**Solution:** Generate hash for existing models:

```bash
python generate_model_hashes.py
```

---

## 📚 Related Documentation

- [SECURITY_FIXES_APPLIED.md](SECURITY_FIXES_APPLIED.md) - Phase 1 security fixes
- [SECURITY_IMPROVEMENTS_PHASE2.md](SECURITY_IMPROVEMENTS_PHASE2.md) - Additional protections
- [verify_security.py](verify_security.py) - Automated security testing

---

## ✨ Summary

**You don't need to manually convert models anymore!**

Every time you train a model:
1. Model saves automatically → `.pth` file
2. Hash generates automatically → `.pth.hash` file  
3. Verification happens automatically on load

**Your workflow is now secure by default. Just train and deploy!** 🎉
