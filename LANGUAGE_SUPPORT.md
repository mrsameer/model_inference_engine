# Multi-Language Support for Gemini VLM

This document describes the Telugu (తెలుగు) language support implementation for the Gemini-based pest detection system.

## Overview

The system now supports returning pest detection results, descriptions, and remedies in multiple languages:
- **English** (`en`) - Default
- **Telugu** (`te`) - తెలుగు

## Supported Pests/Diseases

All 8 pest/disease types are supported in both languages:

### Maize Pests
- Fall Army Worm (ఫాల్ ఆర్మీ వార్మ్)

### Paddy/Rice Pests & Diseases
- Sheath Blight (షీత్ బ్లైట్)
- Brown Plant Hopper (బ్రౌన్ ప్లాంట్ హాపర్)
- Paddy Smut (వరి స్మట్)
- Rice Leaf Roller (వరి ఆకు రోలర్)
- Bacterial Leaf Blight (బాక్టీరియల్ లీఫ్ బ్లైట్)

### Cotton Pests
- Pink Boll Worm (పింక్ బోల్ వార్మ్)
- White Fly (వైట్ ఫ్లై)

## API Usage

### Request Format

Add the `language` parameter to your inference request:

```json
{
  "model_id": "vlm_ss",
  "user_id": "farmer_123",
  "crop": "maize",
  "language": "te",
  "image_url": "https://example.com/image.jpg"
}
```

### Parameters

- `language` (optional): Language code for response
  - `"en"` - English (default)
  - `"te"` - Telugu
  - Falls back to English if invalid language provided

## Response Format

### English Response Example

```json
{
  "detections": [
    {
      "label": "fall_army_worm",
      "confidence": 0.98,
      "box": { "x_min": 146, "y_min": 194, "x_max": 372, "y_max": 337 }
    }
  ],
  "answers": [
    {
      "answer": "**Fall Army Worm**\n\nFall Army Worm is a destructive pest that attacks maize crops. Larvae feed on leaves, creating characteristic ragged holes and windows in the foliage. The caterpillars have an inverted Y-shaped marking on their head capsule.\n\n**Remedies:**\nApply neem-based bio-pesticides or chemical insecticides like Emamectin Benzoate. Use pheromone traps for early detection. Practice crop rotation and intercropping with non-host plants. Remove and destroy infested plants.",
      "confidence": 1.0
    }
  ]
}
```

### Telugu Response Example

```json
{
  "detections": [
    {
      "label": "fall_army_worm",
      "confidence": 0.98,
      "box": { "x_min": 146, "y_min": 194, "x_max": 372, "y_max": 337 }
    }
  ],
  "answers": [
    {
      "answer": "**ఫాల్ ఆర్మీ వార్మ్**\n\nఫాల్ ఆర్మీ వార్మ్ మొక్కజొన్న పంటలపై దాడి చేసే విధ్వంసక తెగులు. లార్వా ఆకులను తింటుంది, ఆకులలో చిరిగిన రంధ్రాలు మరియు కిటికీల వంటి లక్షణాలను సృష్టిస్తుంది. గొంగళి పురుగుల తల క్యాప్సూల్‌పై విలోమ Y-ఆకారపు గుర్తు ఉంటుంది.\n\n**నివారణలు:**\nవేప ఆధారిత జీవ-పురుగుమందులు లేదా ఎమామెక్టిన్ బెంజోయేట్ వంటి రసాయన పురుగుమందులను వర్తింపజేయండి. ముందస్తు గుర్తింపు కోసం ఫెరోమోన్ ట్రాప్‌లను ఉపయోగించండి. పంట మార్పిడి మరియు అతిధేయేతర మొక్కలతో మధ్యవర్తి సాగు చేయండి. సోకిన మొక్కలను తొలగించి నాశనం చేయండి.",
      "confidence": 1.0
    }
  ]
}
```

## Implementation Details

### Data Structure

Pest information is stored in a nested dictionary:

```python
PEST_INFO = {
    "en": {
        "fall_army_worm": {
            "name": "Fall Army Worm",
            "description": "...",
            "remedies": "..."
        },
        # ... other pests
    },
    "te": {
        "fall_army_worm": {
            "name": "ఫాల్ ఆర్మీ వార్మ్",
            "description": "...",
            "remedies": "..."
        },
        # ... other pests
    }
}
```

### Language Selection

1. API receives `language` parameter (default: `"en"`)
2. System validates language exists in `PEST_INFO`
3. Falls back to English if invalid language provided
4. Pest information retrieved from appropriate language dictionary

### Detection Labels

- Detection labels remain in English (e.g., `fall_army_worm`)
- This ensures consistency across language versions
- Only names, descriptions, and remedies are translated

## Testing

Run the Telugu language test:

```bash
python test_telugu_gemini.py
```

Expected output:
```
✓ TELUGU LANGUAGE SUPPORT TEST PASSED

Testing English (language=en)
  Name: Fall Army Worm
  Description: Fall Army Worm is a destructive pest...

Testing Telugu (language=te)
  Name: ఫాల్ ఆర్మీ వార్మ్
  Description: ఫాల్ ఆర్మీ వార్మ్ మొక్కజొన్న పంటలపై...
```

## Adding New Languages

To add a new language (e.g., Hindi - `hi`):

1. Add new language dictionary to `PEST_INFO` in `main.py`:
```python
PEST_INFO = {
    "en": { ... },
    "te": { ... },
    "hi": {  # New language
        "fall_army_worm": {
            "name": "फॉल आर्मी वर्म",
            "description": "...",
            "remedies": "..."
        },
        # ... translate all pests
    }
}
```

2. Update the remedies label logic if needed:
```python
remedies_label = {
    "en": "Remedies",
    "te": "నివారణలు",
    "hi": "उपाय"
}[language]
```

3. Test the new language with existing test scripts

## Notes

- Language parameter is optional (defaults to English)
- Invalid language codes gracefully fall back to English
- System logs language selection for debugging
- All languages use the same Gemini VLM for detection
- Only response text is translated, not the detection logic

## Future Enhancements

Potential improvements:
1. Add more languages (Hindi, Kannada, Tamil, etc.)
2. Support language-specific crop names
3. Database logging of language preferences
4. Auto-detect language from user profile
5. Support mixed-language responses for multilingual users
