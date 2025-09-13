# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Teams Meeting RAG (Retrieval-Augmented Generation) system** that processes meeting transcriptions and provides intelligent question-answering capabilities. The system uses **Azure AI Search exclusively** for both vector storage and metadata management, with OpenAI for embeddings and response generation.

## Core Architecture

### Azure-Only Single-File Architecture
The entire system is implemented in `meeting_rag_processor.py` containing:
- **TranscriptParser**: Multi-format parser (DOCX, TXT/VTT, PDF) with flexible regex patterns
- **ChunkingService**: Contextual Conversation Chunking (CCC) strategy
- **SearchService**: Azure AI Search with comprehensive metadata fields
- **MeetingProcessor**: Main orchestration class

### Simplified Data Flow
```
Meeting Files → TranscriptParser → ChunkingService → Azure AI Search → Hybrid Search → LLM → Response
```

### Chunking Strategy (CCC)
The system uses **Contextual Conversation Chunking** which:
- Creates 500-token target chunks with natural conversation boundaries
- Maintains 1-2 speaker turns of context overlap
- Uses pause-based natural boundaries (30+ second gaps)
- Preserves meeting context without information loss

### Series Analysis Architecture
The system provides advanced **multi-meeting series analysis**:
- **Automatic Series Detection**: LLM detects series queries (e.g., "Document Fulfillment meetings")
- **Meeting Aggregation**: Groups chunks by meeting_id, chronologically orders meetings
- **Context Prioritization**: Summary chunks prioritized, then regular chunks from each meeting
- **Cross-Meeting Insights**: Tracks themes, decisions, and action items across entire series
- **Scalable Design**: Handles 100+ meetings per series with efficient token management
- **Progressive Analysis**: Shows evolution of topics from first meeting to last meeting

### LLM-Driven Query Intelligence
The system uses **dual LLM architecture** for optimal performance:
- **Query Analyzer LLM** (gpt-4o-mini): Fast query intent analysis and filter generation
- **Response Generator LLM** (gpt-4o): Comprehensive response generation with 60k token context
- **No hardcoded logic**: All query analysis and decision-making is LLM-driven
- **Case-insensitive fuzzy matching**: All searches use Azure AI Search's native case-insensitive matching
- **Dynamic calendar awareness**: Query analyzer receives current date context for relative date queries

### Summary Generation (60k Token Enhanced)
- Processes **entire meeting** in batches of 100 entries (increased from 35)
- Uses 60k token context window for comprehensive analysis
- Creates **structured fields** for optimized retrieval:
  - `meeting_purpose`: Brief description of meeting objective
  - `key_outcomes`: Main accomplishments and results
  - `main_topics`: Primary discussion areas with context
  - `decisions_made`: Decisions with attribution and reasoning
  - `action_items`: Tasks with owners and deadlines
  - `past_events`: References to previous meetings/projects
  - `future_actions`: Upcoming meetings and commitments
  - `detailed_narrative`: Comprehensive meeting flow with verbatim quotes
- **Dynamic Field Parsing**: Simple section-based extraction handles various LLM response formats

## Environment Setup

### Required Services
- **Azure AI Search**: Vector storage and hybrid search (requires S1+ for semantic search)
- **OpenAI**: text-embedding-3-small (1536 dimensions) + dual LLM setup (gpt-4o + gpt-4o-mini)
- **PostgreSQL**: Optional - only needed for database_exporter.py utility

### Environment Variables (.env)
```
# Required
OPENAI_API_KEY=your_openai_key
AZURE_AI_SEARCH_KEY=your_search_key  
AZURE_AI_SEARCH_ENDPOINT=https://your-service.search.windows.net
AZURE_AI_SEARCH_INDEX=your_index_name

# Optional - for database export utility only
AZURE_POSTGRES_CONNECTION_STRING=Host=host;Database=db;Username=user;Password=pass

# Optional - for enterprise Azure authentication
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
```

### Windows WSL Environment
This codebase runs in Windows WSL with Python venv:
```bash
# Activate environment (Windows venv in WSL)
venv/Scripts/python.exe script.py

# NOT: source venv/bin/activate (Linux style)
```

## Core Commands

### Development Setup
```bash
# Install dependencies
venv/Scripts/python.exe -m pip install -r requirements.txt

# Test all connections
venv/Scripts/python.exe meeting_rag_processor.py --test
```

### Processing Meeting Files
```bash
# Process a meeting transcript
venv/Scripts/python.exe meeting_rag_processor.py --file "temp/meeting.docx"

# Supported formats: .docx, .txt/.vtt, .pdf
# Files should be placed in temp/ directory
```

### Querying the RAG System
```bash
# Ask questions (returns LLM-generated responses)
venv/Scripts/python.exe meeting_rag_processor.py --search "What did Sandeep demonstrate?"

# Filter by specific meeting
venv/Scripts/python.exe meeting_rag_processor.py --search "action items" --meeting-id "uuid-here"
```

### Database Export Utility
```bash
# Export PostgreSQL data to CSV files
venv/Scripts/python.exe database_exporter.py

# Exports data to exported_data/ directory
```

### Advanced Query Testing Examples
```bash
# Series analysis queries
venv/Scripts/python.exe meeting_rag_processor.py --search "summarize AIML series meetings"
venv/Scripts/python.exe meeting_rag_processor.py --search "what topics were discussed in fulfillment meetings?"

# Temporal queries with calendar awareness
venv/Scripts/python.exe meeting_rag_processor.py --search "meetings from last week"
venv/Scripts/python.exe meeting_rag_processor.py --search "what happened in July 2025?"

# Case-insensitive speaker queries
venv/Scripts/python.exe meeting_rag_processor.py --search "what did SANDEEP contribute to aiml meetings?"
venv/Scripts/python.exe meeting_rag_processor.py --search "show me action items from Joseph and Michael"

# Intent-specific queries
venv/Scripts/python.exe meeting_rag_processor.py --search "what decisions were made across all Document Fulfillment meetings?"
```

## Database Schema

### PostgreSQL Tables
- **meetings**: Meeting metadata, participants, series tracking
- **chunks**: Chunk metadata with speaker info and token counts  
- **meeting_relationships**: Series and cross-meeting relationships

### Azure AI Search Index
- **Hybrid fields**: content (text) + content_vector (1536d embeddings)
- **Metadata fields**: meeting_id, speakers, chunk_type, meeting_date, meeting_time, duration, metadata_json
- **Structured summary fields**: meeting_purpose, key_outcomes, main_topics, decisions_made, action_items, past_events, future_actions, detailed_narrative
- **Vector configuration**: HNSW algorithm with cosine similarity

## Key Implementation Details

### Multi-Format Transcript Parsing
- **DOCX**: Pattern `Speaker [MM:SS] text` with metadata in first paragraph
- **VTT**: WebVTT format `<v Speaker>text</v>` with timestamps
- **PDF**: Attempts both DOCX and VTT parsing strategies
- **Metadata Extraction**: Title, date, time, duration from document headers

### LLM-Driven RAG Pipeline
1. **Query Analysis**: GPT-4o-mini analyzes user intent and generates search strategy with current date context
2. **Dynamic Filter Generation**: Creates case-insensitive Azure AI Search filters (date, speaker, content, series)
3. **Smart Field Selection**: Automatically selects optimal fields + ensures meeting_id inclusion for series queries
4. **Series-Aware Context Building**: Prioritizes summary chunks, groups by meeting_id, chronologically orders
5. **Context Assembly**: Builds 50k token context from retrieved fields
6. **Response Generation**: GPT-4o generates comprehensive answers with source attribution

### Intelligent Query Analysis Examples
- **Temporal queries**: "last week meetings", "current month" → Dynamic calendar-based date range filters
- **Series queries**: "fulfillment meetings", "aiml series" → Fuzzy series title matching with multi-meeting aggregation
- **Speaker queries**: "What did john say?" → Case-insensitive speaker matching + conversation content retrieval
- **Intent-based queries**: "Action items" → Retrieves `action_items` and `future_actions` fields only
- **Complex queries**: "Compare Q1 vs Q2 outcomes" → Multi-temporal analysis with comprehensive context
- **Casual queries**: "WHAT DID SANDEEP DO?" → Normalized to case-insensitive matching

### Performance Considerations
- **Summary processing**: 100-entry batches utilizing 60k token context window
- **Search context**: 50k token context assembly with intelligent field selection
- **LLM Cost Optimization**: Query analyzer (gpt-4o-mini) + targeted field retrieval reduces costs by 60-80%
- **Dynamic Model Selection**: Easy model switching via `AVAILABLE_MODELS` configuration
- **Zero Hardcoded Logic**: All decisions are LLM-driven for maximum flexibility

### Model Configuration and Flexibility
The system uses a flexible model configuration in `AVAILABLE_MODELS`:
- **gpt-5**: Maps to gpt-4o for main response generation (16k tokens)
- **gpt-4.1**: Maps to gpt-4o-mini for balanced performance (16k tokens)  
- **query-analyzer**: Maps to gpt-4o-mini for fast query analysis (2k tokens)

Global LLM initialization pattern:
```python
try:
    access_token = get_access_token()
    embedding_model = get_embedding_model(access_token)
    llm = get_llm(access_token)
    query_analyzer_llm = get_query_analyzer_llm(access_token)
except Exception as e:
    # All models set to None on failure
```

### Error Handling and User Experience
- **PostgreSQL**: Uses `gen_random_uuid()` (not uuid-ossp extension)
- **Azure Search**: Empty chunk uploads skipped to avoid errors
- **LLM Graceful Degradation**: Basic search when query analyzer unavailable
- **Contextual No-Results Responses**: 
  - Speaker queries: Lists actual meeting participants with suggestions
  - Date queries: Provides date range guidance and alternatives
  - Series queries: Explains available meeting series
  - Intent-specific: Tailored messages for decisions, action items, etc.

## Testing Meeting Formats

Sample meeting formats in temp/ directory:
- Document Fulfillment AIML meetings (DOCX format)
- VTT transcript format with precise timestamps
- Test queries include speaker-specific, topic-based, and cross-meeting analysis

The system handles meeting series detection via filename patterns and maintains relationships between related meetings automatically.

## Troubleshooting

### Common Issues
- **Index creation fails**: Ensure Azure AI Search service is S1+ tier
- **Missing dependencies**: Install all requirements with `venv/Scripts/python.exe -m pip install -r requirements.txt`
- **Authentication errors**: Verify OpenAI API key and Azure Search credentials in .env
- **Query analysis fails**: System gracefully degrades to basic search when query_analyzer_llm unavailable
- **Empty structured fields**: Check LLM response parsing - uses simple section-based extraction
- **Missing metadata fields**: Ensure document format matches expected patterns (title, date, time in first paragraph)
- **Series queries showing duplicate dates**: meeting_id field automatically included in all queries for proper grouping
- **Case-sensitive matching issues**: All search.ismatch() operations are inherently case-insensitive in Azure AI Search

## Architecture Principles

### Direct Function Modification Approach
- **No backup functions**: All enhancements modify existing functions directly
- **No duplicate methods**: Prevents code drift and maintenance issues
- **Human-readable code**: Implementations look natural, not AI-generated
- **Single source of truth**: Each function has one clear purpose and implementation

### LLM Integration Pattern
When adding new LLM functionality:
1. Add model config to `AVAILABLE_MODELS` 
2. Create `get_*_llm()` function following existing pattern
3. Add to global initialization block
4. Modify existing methods directly (no new functions unless absolutely necessary)