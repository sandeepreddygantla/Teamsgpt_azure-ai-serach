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

### Summary Generation
- Processes **entire meeting** in batches of 35 entries
- Generates comprehensive summaries with structured extraction:
  - Main topics discussed
  - Decisions made and action items
  - Past events referenced and future actions
  - Participant contributions

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
- **Metadata fields**: meeting_id, speakers, chunk_type, meeting_date
- **Vector configuration**: HNSW algorithm with cosine similarity

## Key Implementation Details

### Multi-Format Transcript Parsing
- **DOCX**: Pattern `Speaker [MM:SS] text` 
- **VTT**: WebVTT format `<v Speaker>text</v>` with timestamps
- **PDF**: Attempts both DOCX and VTT parsing strategies

### RAG Pipeline
1. **Retrieval**: Hybrid search (vector + keyword) via Azure AI Search
2. **Context Assembly**: Top-K chunks with meeting metadata
3. **Generation**: LLM prompt with retrieved context + user question
4. **Response**: Structured answer with source attribution

### Performance Considerations
- Summary processing: 35-entry batches to balance completeness and token limits
- Search context: 12,000 token limit with intelligent truncation
- Azure Search index auto-creation with schema validation

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