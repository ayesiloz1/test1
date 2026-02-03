# Chat Style Improvements - Visual Reference

## Modern Message Bubbles

### User Messages (Right-aligned, Blue Gradient)
```
                    ┌─────────────────────────┐
                    │ YOU                     │
                    │                         │
                    │ What is the accuracy    │
                    │ for this batch?         │
                    └─────────────────────────┘
                              Blue gradient
                              Right-aligned
                              Rounded: 18px 18px 4px 18px
```

### AI Messages (Left-aligned, Dark Gradient)
```
┌──────────────────────────────────────┐
│ 🤖 AI ASSISTANT                      │
│                                      │
│ Your batch achieved **86.9%**        │
│ accuracy across `497` images.        │
│                                      │
│ Model Performance:                   │
│   • CNN: 86.92%                      │
│   • Hybrid: 85.51%                   │
└──────────────────────────────────────┘
  Dark gradient with border
  Left-aligned
  Rounded: 18px 18px 18px 4px
```

### System Messages (Centered, Subtle)
```
        ┌─────────────────────────┐
        │ 🤖 Welcome! AI ready    │
        └─────────────────────────┘
              Subtle gray
              Centered
              Italic text
```

## Color Palette

### Background
- Main: `#0e1117` (Very dark blue-gray)
- Bubbles: `#1f2937` → `#111827` (Gradient)
- Borders: `#374151`, `#4b5563`

### Accents
- User Bubble: `#2563eb` → `#1d4ed8` (Blue gradient)
- AI Header: `#10b981` (Green)
- Buttons: `#3b82f6` (Blue)
- Success: `#10b981` (Green)
- Error: `#ef4444` (Red)

### Text
- Primary: `#ffffff` (White)
- Secondary: `#e5e7eb`, `#d1d5db` (Light gray)
- Tertiary: `#9ca3af`, `#6b7280` (Medium gray)
- Muted: `#4b5563` (Dark gray)

### Highlights
- Percentages: `#fbbf24` (Yellow/gold)
- Numbers: `#60a5fa` (Light blue)
- Code: `rgba(255,255,255,0.1)` background

## Typography

### Fonts
- Main: `Segoe UI, 10pt`
- Code: `monospace, 13px`
- Labels: `11px` (small caps style)
- Input: `14px`

### Line Height
- Chat: `1.6`
- Bubbles: `1.5`
- Buttons: `1.0`

## Spacing

### Message Bubbles
```
Padding: 12px 16px
Margin: 10px 0
Border-radius: 18px (with small corner variant)
Max-width: 80-85%
```

### Input Field
```
Padding: 14px 16px
Border-radius: 24px (pill shape)
Border: 2px
```

### Buttons
```
Padding: 12px 24px (Send)
Padding: 8px 16px (Upload)
Border-radius: 24px (pill shape)
```

## Text Formatting Examples

### Input Text
```
Your batch achieved **86.9% accuracy** with `497` images.

The CNN model performed well on:
- Cracks (CR): 91% precision
- Normal welds (ND): 95% precision

Reconstruction error was 0.025 (threshold: 0.018)
```

### Rendered Output
```
Your batch achieved 𝗮𝗰𝗰𝘂𝗿𝗮𝗰𝘆 with 497 images.
                     ↑           ↑        ↑
                   bold      yellow    blue

The CNN model performed well on:
  • Cracks (CR): 91% precision
                 ↑
              yellow

Reconstruction error was 0.025 (threshold: 0.018)
                           ↑                 ↑
                         blue              blue
```

## Interactive Elements

### Buttons

#### Send Button
```
┌──────────────┐
│  Send ➤      │  ← Normal (blue gradient)
└──────────────┘

┌──────────────┐
│  Send ➤      │  ← Hover (darker blue)
└──────────────┘

┌──────────────┐
│ Thinking ⏳  │  ← Processing (disabled)
└──────────────┘
```

#### Quick Question Buttons
```
┌────────────────────────────────┐
│ 🔍 What defects were detected? │  ← Normal
└────────────────────────────────┘

┌────────────────────────────────┐
│ 🔍 What defects were detected? │  ← Hover (lighter)
└────────────────────────────────┘
```

### Input Field
```
┌─────────────────────────────────────────┐
│ 💬 Ask a question about the results...  │  ← Placeholder
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ What is the accuracy?|                   │  ← Typing
└─────────────────────────────────────────┘
      Blue border when focused
```

## Animation States

### Typing Indicator
```
Frame 1:  ● ○ ○
Frame 2:  ○ ● ○
Frame 3:  ○ ○ ●
Frame 4:  ● ○ ○
...
```

### Button Transitions
- Hover: 200ms ease
- Press: Instant
- Disable: 150ms fade

### Message Appearance
- Fade in: 150ms
- Slide in: 200ms ease-out
- Auto-scroll: Smooth

## Layout Structure

```
┌──────────────────────────────────────────────┐
│ 🤖 AI Assistant (Azure OpenAI GPT-4)        │  ← Title
├──────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────┐ │
│ │ 📚 Knowledge Base                        │ │  ← KB Controls
│ │  [📤 Upload] 📄 42 documents             │ │
│ └──────────────────────────────────────────┘ │
├──────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────┐ │
│ │                                          │ │
│ │  System: 🤖 Welcome...                   │ │
│ │                                          │ │  ← Chat Display
│ │                  ┌────────────┐          │ │
│ │                  │ User msg   │          │ │
│ │                  └────────────┘          │ │
│ │                                          │ │
│ │  ┌────────────────┐                     │ │
│ │  │ AI response    │                     │ │
│ │  └────────────────┘                     │ │
│ │                                          │ │
│ └──────────────────────────────────────────┘ │
├──────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────┐ │
│ │ 💡 Quick Questions                       │ │  ← Quick Buttons
│ │  [🔍 What defects were detected?]        │ │
│ │  [⚠️ How severe is this?]                │ │
│ │  [📊 Explain reconstruction error]       │ │
│ │  [⚖️ Compare models]                     │ │
│ │  [🔧 What causes these defects?]         │ │
│ │  [📋 Inspection procedures]              │ │
│ └──────────────────────────────────────────┘ │
├──────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────┐ │
│ │ 💬 Ask a question... [Send ➤] [Clear 🗑️]│ │  ← Input Area
│ └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

## Accessibility

### Contrast Ratios
- White on dark blue: 14.5:1 ✓
- Light gray on dark: 8.2:1 ✓
- Yellow on dark: 9.5:1 ✓
- Blue on dark: 7.8:1 ✓

### Keyboard Navigation
- Tab through buttons
- Enter to send
- Clear shortcuts available

### Screen Readers
- Emojis have alt text
- Buttons labeled clearly
- Message roles defined

## Responsive Behavior

### Width Adaptation
- Small: Messages 90% width
- Medium: Messages 80-85% width
- Large: Messages maintain max-width

### Text Wrapping
- Word wrap enabled
- Line breaks preserved
- Long URLs handled

## Best Practices Applied

✅ Modern gradient designs
✅ Consistent spacing (8px grid)
✅ Professional typography
✅ Clear visual hierarchy
✅ Accessible color contrast
✅ Smooth animations
✅ Responsive layout
✅ Icon usage throughout
✅ Semantic HTML structure
✅ Proper state management

## Quick Reference

| Element | Style |
|---------|-------|
| User Bubble | Blue gradient, right, rounded corners |
| AI Bubble | Dark gradient, left, with border |
| System Msg | Gray, centered, subtle |
| Numbers | Blue highlight |
| Percentages | Yellow/gold highlight |
| Code | Gray background, monospace |
| Bold | **text** → font-weight: bold |
| Italic | *text* → font-style: italic |
| Buttons | Pill-shaped, gradient |
| Input | Pill-shaped, focused border |
| Icons | Emojis throughout |

---

**Result:** A modern, professional chat interface that looks like it belongs in 2026! 🎨✨
