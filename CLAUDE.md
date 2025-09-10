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
- Detects topic shifts, Q&A pairs, and speaker changes
- Preserves meeting context without information loss

### Summary Generation (Enhanced)
- Processes **entire meeting** in batches of 35 entries
- Creates **structured fields** for optimized retrieval:
  - `meeting_purpose`: Brief description of meeting objective
  - `key_outcomes`: Main accomplishments and results
  - `main_topics`: Primary discussion areas with context
  - `decisions_made`: Decisions with attribution and reasoning
  - `action_items`: Tasks with owners and deadlines
  - `past_events`: References to previous meetings/projects
  - `future_actions`: Upcoming meetings and commitments
  - `detailed_narrative`: Comprehensive meeting flow with verbatim quotes
- **Flexible LLM Response Parsing**: Regex-based extraction handles multiple formats (markdown headers, bold text, plain text)

## Environment Setup

### Required Services
- **Azure AI Search**: Vector storage and hybrid search (requires S1+ for semantic search)
- **OpenAI**: text-embedding-3-small (1536 dimensions) + GPT-4 for generation
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

### RAG Pipeline with Query Intent Detection
1. **Query Analysis**: Detects intent (action items, decisions, topics, etc.)
2. **Smart Retrieval**: 
   - Targeted queries retrieve only relevant structured fields
   - Complex queries retrieve full context
   - Automatic field projection reduces token usage by 60-80%
3. **Context Assembly**: Builds context from available fields dynamically
4. **Generation**: LLM prompt with optimized context
5. **Response**: Structured answer with source attribution

### Query Optimization Patterns
- **Action items queries** → Retrieves only: action_items, future_actions fields
- **Decision queries** → Retrieves only: decisions_made field
- **Topic queries** → Retrieves only: main_topics, meeting_purpose fields
- **Outcome queries** → Retrieves only: key_outcomes field
- **Complex queries** → Retrieves: detailed_narrative or full content

### Performance Considerations
- Summary processing: 35-entry batches to balance completeness and token limits
- Search context: 12,000 token limit with intelligent truncation
- Azure Search index auto-creation with schema validation
- Field-specific retrieval reduces LLM costs by 60-80% for targeted queries
- Structured fields enable faster response times for common questions

### Error Handling
- PostgreSQL uses `gen_random_uuid()` (not uuid-ossp extension)
- Empty chunk uploads are skipped to avoid Azure Search errors
- Token counting prevents LLM context overflow

## Testing Meeting Formats

Sample meeting formats in temp/ directory:
- Document Fulfillment AIML meetings (DOCX format)
- VTT transcript format with precise timestamps
- Test queries include speaker-specific, topic-based, and cross-meeting analysis

The system handles meeting series detection via filename patterns and maintains relationships between related meetings automatically.

## Troubleshooting

### Common Issues
- **Index creation fails**: Ensure Azure AI Search service is S1+ tier
- **Token limit errors**: System automatically truncates context at 12,000 tokens
- **Missing dependencies**: Install all requirements with `venv/Scripts/python.exe -m pip install -r requirements.txt`
- **Authentication errors**: Verify OpenAI API key and Azure Search credentials in .env
- **Empty structured fields**: LLM response parsing uses flexible regex that handles markdown (###), bold (**), and plain text formats
- **Query not using optimized fields**: Verify query keywords match intent detection patterns
- **Missing metadata fields**: Ensure document format matches expected patterns (title, date, time in first paragraph)