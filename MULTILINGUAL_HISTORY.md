# Multilingual History Retrieval

## Overview

The history API now supports retrieving past inferences in **any language**, regardless of the original request language. This allows farmers to:
- View their detection history in their preferred language
- Switch between languages without losing data
- Share history with others who speak different languages

## How It Works

### Storage Strategy

1. **Detections are stored once** (language-agnostic)
   - Labels like `fall_army_worm`, `sheath_blight` remain in English
   - Bounding boxes and confidence scores are universal

2. **Original language is tracked** for analytics
   - Database stores which language was requested
   - Useful for understanding user preferences

3. **Answers are regenerated on-the-fly** when retrieving
   - Pest names, descriptions, and remedies generated in requested language
   - Uses the same `PEST_INFO` dictionary as real-time inference

## API Usage

### Endpoint
```
GET /history/{user_id}?page=1&page_size=10&language={language_code}
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `user_id` | Yes | - | User identifier |
| `page` | No | 1 | Page number for pagination |
| `page_size` | No | 10 | Items per page (max: 100) |
| `language` | No | original | Language code: `en` or `te` |

### Examples

#### 1. Retrieve in Original Languages (Default)
No language parameter - returns answers in the language originally requested:

```bash
GET /history/farmer123?page=1&page_size=10
```

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "language": "en",  // Original request language
      "detections": [...],
      "answers": [{"answer": "**Fall Army Worm**\n\n..."}]  // English
    },
    {
      "id": 2,
      "language": "te",  // Original request language
      "detections": [...],
      "answers": [{"answer": "**షీత్ బ్లైట్**\n\n..."}]  // Telugu
    }
  ]
}
```

#### 2. Force All Results to English
Add `language=en` parameter:

```bash
GET /history/farmer123?page=1&language=en
```

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "language": "en",  // Original: English
      "answers": [{"answer": "**Fall Army Worm**\n\n..."}]  // English
    },
    {
      "id": 2,
      "language": "te",  // Original: Telugu, converted to English
      "answers": [{"answer": "**Sheath Blight**\n\n..."}]  // English
    }
  ]
}
```

#### 3. Force All Results to Telugu
Add `language=te` parameter:

```bash
GET /history/farmer123?page=1&language=te
```

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "language": "en",  // Original: English, converted to Telugu
      "answers": [{"answer": "**ఫాల్ ఆర్మీ వార్మ్**\n\n..."}]  // Telugu
    },
    {
      "id": 2,
      "language": "te",  // Original: Telugu
      "answers": [{"answer": "**షీత్ బ్లైట్**\n\n..."}]  // Telugu
    }
  ]
}
```

## Use Cases

### 1. Farmer Switches Language Preference
A farmer originally used the app in English but now prefers Telugu:
```bash
# Retrieve all past detections in Telugu
GET /history/farmer123?language=te
```
All previous English detections are now shown in Telugu!

### 2. Extension Worker Reviews Farmer's History
An extension worker who speaks English reviews a Telugu farmer's detection history:
```bash
# View Telugu farmer's history in English
GET /history/telugu_farmer456?language=en
```

### 3. Multi-language Reports
Generate reports in multiple languages for the same data:
```bash
# English report
GET /history/farmer123?language=en

# Telugu report
GET /history/farmer123?language=te
```

## Technical Details

### Answer Regeneration Process

When `language` parameter differs from stored language:

1. **Load detections** from database (stored once)
2. **Extract pest types** from detection labels
3. **Look up pest info** in `PEST_INFO[target_language]`
4. **Generate new answers** with localized:
   - Pest names (e.g., "Fall Army Worm" → "ఫాల్ ఆర్మీ వార్మ్")
   - Descriptions (full text translation)
   - Remedies (full text translation)
   - Section labels ("Remedies" → "నివారణలు")

### Performance

- **No extra database queries** - answers generated in memory
- **Same response time** - minimal overhead for translation
- **Efficient** - only regenerates when language differs

### Data Integrity

- **Original language preserved** in `language` field
- **Detections unchanged** - bounding boxes remain accurate
- **Confidence scores unchanged** - detection quality preserved
- **Timestamps unchanged** - historical record intact

## Benefits

✅ **Flexibility**: View same data in any language
✅ **No Data Loss**: Original requests tracked for analytics
✅ **Efficient Storage**: Detections stored once, not per language
✅ **Easy Migration**: Farmers can switch languages anytime
✅ **Shareable**: Same detection viewable by users of different languages
✅ **Backward Compatible**: Works with old records without language field

## Validation

The system validates language codes and falls back gracefully:
- Invalid language → uses original language
- Missing language in old records → defaults to English
- Unsupported pest in target language → skips that pest

## Logging

When answers are regenerated, the system logs:
```
INFO: Regenerated answers in 'te' for history item 123 (original: 'en')
```

This helps track language conversion activity and debug any issues.
