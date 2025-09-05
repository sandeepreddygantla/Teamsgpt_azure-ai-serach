#!/usr/bin/env python3
"""
Teams Meeting RAG Processor
Single file implementation with CCC chunking strategy
"""

import os
import re
import json
import uuid
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Core imports
from dotenv import load_dotenv
import tiktoken
import httpx

# OpenAI and LangChain
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Azure AI Search
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SearchFieldDataType
)
from azure.core.credentials import AzureKeyCredential

# PostgreSQL
import psycopg2
from psycopg2.extras import RealDictCursor

# Document processing
from docx import Document
import PyPDF2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Configure tiktoken cache
tiktoken_cache_dir = os.path.abspath("tiktoken_cache")
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# Global variables from your existing code
project_id = "openai-meeting-processor"
current_model_name = "gpt-4o"

# Available models configuration
AVAILABLE_MODELS = {
    "gpt-5": {
        "name": "GPT-5",
        "model": "gpt-4o",
        "temperature": 0,
        "max_tokens": 16000,
        "description": "Most advanced model for complex reasoning"
    },
    "gpt-4.1": {
        "name": "GPT-4.1",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 16000,
        "description": "Enhanced GPT-4 model for balanced performance"
    }
}

def get_access_token():
    """Get access token - keeping your existing function structure"""
    try:
        azure_client_id = os.getenv("AZURE_CLIENT_ID")
        azure_client_secret = os.getenv("AZURE_CLIENT_SECRET")
        
        if azure_client_id and azure_client_secret:
            logger.info("Azure environment detected - getting Azure AD token")
            auth = "https://api.uhg.com/oauth2/token"
            scope = "https://api.uhg.com/.default"
            grant_type = "client_credentials"
            
            with httpx.Client() as client:
                body = {
                    "grant_type": grant_type,
                    "scope": scope,
                    "client_id": azure_client_id,
                    "client_secret": azure_client_secret
                }
                headers = {"Content-Type": "application/x-www-form-urlencoded"}
                response = client.post(auth, headers=headers, data=body, timeout=60)
                response.raise_for_status()
                return response.json()["access_token"]
        else:
            logger.info("OpenAI environment detected - using API key authentication")
            return None
            
    except Exception as e:
        logger.error(f"Error getting access token: {e}")
        return None

def get_llm(access_token: str = None, model_name: str = None):
    """Get OpenAI LLM client - using your existing function"""
    try:
        current_api_key = os.getenv("OPENAI_API_KEY")
        if not current_api_key:
            logger.warning("OPENAI_API_KEY environment variable not set")
            return None
        
        model_config = AVAILABLE_MODELS.get(model_name or current_model_name, AVAILABLE_MODELS["gpt-5"])
        
        return ChatOpenAI(
            model=model_config["model"],
            openai_api_key=current_api_key,
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            request_timeout=120
        )
    except Exception as e:
        logger.error(f"Error creating LLM client: {e}")
        return None

def get_embedding_model(access_token: str = None):
    """Get OpenAI embedding model - modified to use text-embedding-3-small"""
    try:
        current_api_key = os.getenv("OPENAI_API_KEY")
        if not current_api_key:
            logger.warning("OPENAI_API_KEY environment variable not set")
            return None
        
        return OpenAIEmbeddings(
            model="text-embedding-3-small",  # Using text-embedding-3-small as requested
            openai_api_key=current_api_key,
            dimensions=1536  # text-embedding-3-small dimension
        )
    except Exception as e:
        logger.error(f"Error creating embedding model: {e}")
        return None

# Initialize global variables using your existing structure
try:
    access_token = get_access_token()
    embedding_model = get_embedding_model(access_token)
    llm = get_llm(access_token)
except Exception as e:
    logger.error(f"Error initializing models: {e}")
    embedding_model = None
    llm = None

@dataclass
class TranscriptEntry:
    speaker: str
    timestamp: str
    text: str
    timestamp_seconds: float

@dataclass
class MeetingChunk:
    chunk_id: str
    meeting_id: str
    chunk_type: str
    content: str
    speakers: List[str]
    start_timestamp: str
    end_timestamp: str
    chunk_index: int
    token_count: int
    has_context: bool
    metadata: Dict

class TranscriptParser:
    """Handles parsing of different transcript formats"""
    
    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def parse_file(self, file_path: str) -> Tuple[List[TranscriptEntry], Dict]:
        """Parse transcript file based on format"""
        file_ext = file_path.split('.')[-1].lower()
        
        if file_ext == 'docx':
            return self._parse_docx(file_path)
        elif file_ext in ['txt', 'vtt']:
            return self._parse_txt_vtt(file_path)
        elif file_ext == 'pdf':
            return self._parse_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _parse_docx(self, file_path: str) -> Tuple[List[TranscriptEntry], Dict]:
        """Parse DOCX format: Speaker [MM:SS] text"""
        doc = Document(file_path)
        content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        entries = []
        metadata = self._extract_metadata_from_docx(content, file_path)
        
        # Pattern for: Speaker Name [MM:SS] or [H:MM:SS]
        pattern = r'([^[]+?)\s*\[(\d+:\d+(?::\d+)?)\]\s*([^\n]+(?:\n(?![^[]+\[)[^\n]*)*)'
        matches = re.findall(pattern, content, re.MULTILINE)
        
        for match in matches:
            speaker = match[0].strip()
            timestamp = match[1]
            text = match[2].strip()
            
            timestamp_seconds = self._timestamp_to_seconds(timestamp)
            
            entries.append(TranscriptEntry(
                speaker=speaker,
                timestamp=timestamp,
                text=text,
                timestamp_seconds=timestamp_seconds
            ))
            
            if speaker not in metadata['participants']:
                metadata['participants'].append(speaker)
        
        return entries, metadata
    
    def _parse_txt_vtt(self, file_path: str) -> Tuple[List[TranscriptEntry], Dict]:
        """Parse VTT format: timestamp --> timestamp <v Speaker>text</v>"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        entries = []
        metadata = {'participants': [], 'file_name': os.path.basename(file_path)}
        
        # Pattern for VTT format
        pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n<v\s+([^>]+)>([^<]+)</v>'
        matches = re.findall(pattern, content, re.MULTILINE)
        
        for match in matches:
            start_time = match[0]
            end_time = match[1]
            speaker = match[2].strip()
            text = match[3].strip()
            
            timestamp_seconds = self._vtt_timestamp_to_seconds(start_time)
            
            entries.append(TranscriptEntry(
                speaker=speaker,
                timestamp=start_time,
                text=text,
                timestamp_seconds=timestamp_seconds
            ))
            
            if speaker not in metadata['participants']:
                metadata['participants'].append(speaker)
        
        if entries:
            last_seconds = self._vtt_timestamp_to_seconds(matches[-1][1])
            metadata['duration'] = f"{int(last_seconds // 60)}m {int(last_seconds % 60)}s"
        
        return entries, metadata
    
    def _parse_pdf(self, file_path: str) -> Tuple[List[TranscriptEntry], Dict]:
        """Parse PDF format"""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text()
        
        # Try to parse as DOCX format first
        try:
            entries, metadata = self._parse_docx_content(content, file_path)
            return entries, metadata
        except:
            # If that fails, try VTT format
            try:
                entries, metadata = self._parse_vtt_content(content, file_path)
                return entries, metadata
            except:
                raise ValueError("Could not parse PDF content in any known format")
    
    def _extract_metadata_from_docx(self, content: str, filename: str) -> Dict:
        """Extract meeting metadata from DOCX content"""
        metadata = {'participants': [], 'file_name': os.path.basename(filename)}
        
        lines = content.split('\n')[:10]
        
        for line in lines:
            # Extract meeting title and ID
            title_pattern = r'^(.*?)-(\d{8}_\d{6})-Meeting Recording'
            if title_match := re.search(title_pattern, line):
                metadata['meeting_title'] = title_match.group(1).strip()
                metadata['series_id'] = title_match.group(1).strip()
                
            # Extract date and time
            date_pattern = r'(\w+\s+\d{1,2},\s+\d{4}),\s+(\d{1,2}:\d{2}\s*[AP]M)'
            if date_match := re.search(date_pattern, line):
                metadata['meeting_date'] = date_match.group(1)
                metadata['meeting_time'] = date_match.group(2)
                
            # Extract duration
            duration_pattern = r'(\d+m\s+\d+s|\d+h\s+\d+m\s+\d+s)'
            if duration_match := re.search(duration_pattern, line):
                metadata['duration'] = duration_match.group(0)
        
        return metadata
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert MM:SS or H:MM:SS to seconds"""
        parts = timestamp.split(':')
        if len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # H:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0
    
    def _vtt_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert HH:MM:SS.mmm to seconds"""
        match = re.match(r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})', timestamp)
        if match:
            h, m, s, ms = match.groups()
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        return 0

class ChunkingService:
    """Implements Contextual Conversation Chunking (CCC) strategy"""
    
    def __init__(self, target_tokens=500, min_tokens=200, max_tokens=800, context_turns=1):
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.context_turns = context_turns
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def create_chunks(self, entries: List[TranscriptEntry], meeting_id: str) -> List[MeetingChunk]:
        """Create chunks using CCC strategy"""
        chunks = []
        
        # Create conversation chunks
        conv_chunks = self._create_conversation_chunks(entries, meeting_id)
        chunks.extend(conv_chunks)
        
        # Create summary chunk
        summary_chunk = self._create_summary_chunk(entries, meeting_id)
        if summary_chunk:
            chunks.append(summary_chunk)
        
        return chunks
    
    def _create_conversation_chunks(self, entries: List[TranscriptEntry], meeting_id: str) -> List[MeetingChunk]:
        """Create overlapping conversation chunks with natural boundaries"""
        if not entries:
            return []
        
        chunks = []
        current_segment = []
        chunk_index = 0
        
        for i, entry in enumerate(entries):
            current_segment.append(entry)
            
            # Check if we should create a chunk
            should_chunk = self._should_create_chunk(current_segment, entries, i)
            
            if should_chunk or i == len(entries) - 1:  # Last entry
                # Add context from previous turns
                context_start = max(0, i - len(current_segment) - self.context_turns)
                context_before = entries[context_start:i - len(current_segment) + 1]
                
                # Add context from next turns
                context_end = min(len(entries), i + self.context_turns + 1)
                context_after = entries[i + 1:context_end]
                
                # Create chunk with context
                full_segment = context_before + current_segment + context_after
                content = self._format_segment_content(full_segment)
                
                # Calculate token count
                token_count = len(self.encoder.encode(content))
                
                # Truncate if too large
                if token_count > self.max_tokens:
                    content = self._truncate_content(content, self.max_tokens)
                    token_count = self.max_tokens
                
                chunk = MeetingChunk(
                    chunk_id=f"{meeting_id}_conv_{chunk_index}",
                    meeting_id=meeting_id,
                    chunk_type="conversation",
                    content=content,
                    speakers=list(set(entry.speaker for entry in current_segment)),
                    start_timestamp=current_segment[0].timestamp,
                    end_timestamp=current_segment[-1].timestamp,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    has_context=bool(context_before or context_after),
                    metadata={
                        'entry_count': len(current_segment),
                        'has_context_before': bool(context_before),
                        'has_context_after': bool(context_after)
                    }
                )
                
                chunks.append(chunk)
                chunk_index += 1
                
                # Keep some overlap for next chunk
                if i < len(entries) - 1:
                    overlap_size = min(2, len(current_segment))
                    current_segment = current_segment[-overlap_size:]
                else:
                    current_segment = []
        
        return chunks
    
    def _should_create_chunk(self, current_segment: List[TranscriptEntry], all_entries: List[TranscriptEntry], current_index: int) -> bool:
        """Determine if we should create a chunk at this point"""
        if not current_segment:
            return False
        
        # Check token count
        content = self._format_segment_content(current_segment)
        token_count = len(self.encoder.encode(content))
        
        # Force chunk if we hit max tokens
        if token_count >= self.max_tokens:
            return True
        
        # Don't chunk if below minimum
        if token_count < self.min_tokens:
            return False
        
        # Check for natural boundaries
        if current_index < len(all_entries) - 1:
            current_entry = all_entries[current_index]
            next_entry = all_entries[current_index + 1]
            
            # Topic shift detection
            if self._detect_topic_shift(current_entry, next_entry):
                return True
            
            # Long pause detection (more than 30 seconds)
            if next_entry.timestamp_seconds - current_entry.timestamp_seconds > 30:
                return True
        
        # Default: chunk if we're at target size
        return token_count >= self.target_tokens
    
    def _detect_topic_shift(self, current_entry: TranscriptEntry, next_entry: TranscriptEntry) -> bool:
        """Detect potential topic shifts"""
        topic_shift_phrases = [
            "moving on", "next topic", "let's discuss", "switching to",
            "another thing", "also", "by the way", "speaking of"
        ]
        
        next_text_lower = next_entry.text.lower()
        return any(phrase in next_text_lower for phrase in topic_shift_phrases)
    
    def _format_segment_content(self, segment: List[TranscriptEntry]) -> str:
        """Format segment entries into readable content"""
        lines = []
        for entry in segment:
            lines.append(f"{entry.speaker} [{entry.timestamp}]: {entry.text}")
        return "\n\n".join(lines)
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token limit"""
        tokens = self.encoder.encode(content)
        if len(tokens) <= max_tokens:
            return content
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoder.decode(truncated_tokens)
    
    def _create_summary_chunk(self, entries: List[TranscriptEntry], meeting_id: str) -> Optional[MeetingChunk]:
        """Create a summary chunk for the entire meeting"""
        if not entries or not llm:
            return None
        
        try:
            # Get first 20 entries for summary
            summary_entries = entries[:20]
            content = self._format_segment_content(summary_entries)
            
            # Generate summary using LLM
            prompt = f"""Summarize this meeting transcript. Include:
1. Main topics discussed
2. Key decisions made
3. Action items mentioned
4. Participants involved

Transcript:
{content[:2000]}"""
            
            summary_response = llm.invoke(prompt)
            summary_text = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
            
            return MeetingChunk(
                chunk_id=f"{meeting_id}_summary",
                meeting_id=meeting_id,
                chunk_type="summary",
                content=summary_text,
                speakers=list(set(entry.speaker for entry in entries)),
                start_timestamp=entries[0].timestamp,
                end_timestamp=entries[-1].timestamp,
                chunk_index=9999,  # High index for summary
                token_count=len(self.encoder.encode(summary_text)),
                has_context=False,
                metadata={'type': 'ai_generated_summary'}
            )
        except Exception as e:
            logger.error(f"Error creating summary chunk: {e}")
            return None

class DatabaseService:
    """Handles PostgreSQL operations"""
    
    def __init__(self):
        self.connection_string = os.getenv('AZURE_POSTGRES_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("AZURE_POSTGRES_CONNECTION_STRING not found in environment")
        
        # Parse connection string
        self.connection_params = self._parse_connection_string(self.connection_string)
        self._create_tables()
    
    def _parse_connection_string(self, conn_string: str) -> Dict:
        """Parse PostgreSQL connection string"""
        params = {}
        parts = conn_string.split(';')
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip().lower()
                
                if key == 'host':
                    params['host'] = value
                elif key == 'database':
                    params['database'] = value
                elif key == 'username':
                    params['user'] = value
                elif key == 'password':
                    params['password'] = value
                elif key == 'port':
                    params['port'] = int(value)
        
        # Set defaults
        params.setdefault('port', 5432)
        params.setdefault('database', 'postgres')
        
        logger.info(f"Connecting to PostgreSQL: host={params.get('host')}, database={params.get('database')}, user={params.get('user')}")
        
        return params
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.connection_params)
    
    def _create_database(self):
        """Create the database if it doesn't exist"""
        # Connect to postgres database to create our target database
        temp_params = self.connection_params.copy()
        target_database = temp_params['database']
        temp_params['database'] = 'postgres'  # Connect to default postgres db
        
        try:
            conn = psycopg2.connect(**temp_params)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE \"{target_database}\"")
                logger.info(f"Created database: {target_database}")
            conn.close()
        except psycopg2.errors.DuplicateDatabase:
            logger.info(f"Database {target_database} already exists")
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
    
    def _create_tables(self):
        """Create required tables"""
        try:
            # First, try to connect and create extension
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if we can connect
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]
                    logger.info(f"Connected to PostgreSQL: {version}")
        except psycopg2.OperationalError as e:
            if "does not exist" in str(e):
                # Try to create the database
                logger.warning(f"Database does not exist, trying to create it: {e}")
                self._create_database()
            else:
                raise
        
        create_tables_sql = """        
        CREATE TABLE IF NOT EXISTS meetings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title VARCHAR(255),
            meeting_date DATE,
            series_id VARCHAR(255),
            file_path TEXT,
            status VARCHAR(50) DEFAULT 'pending',
            participants TEXT[],
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_meetings_series_id ON meetings(series_id);
        CREATE INDEX IF NOT EXISTS idx_meetings_date ON meetings(meeting_date);
        
        CREATE TABLE IF NOT EXISTS chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
            chunk_id VARCHAR(255) UNIQUE,
            chunk_type VARCHAR(50),
            chunk_index INTEGER,
            speakers TEXT[],
            token_count INTEGER,
            azure_search_id VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_chunks_meeting_id ON chunks(meeting_id);
        
        CREATE TABLE IF NOT EXISTS meeting_relationships (
            id SERIAL PRIMARY KEY,
            meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
            related_meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
            relationship_type VARCHAR(50),
            UNIQUE(meeting_id, related_meeting_id, relationship_type)
        );
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_tables_sql)
                conn.commit()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def create_meeting(self, metadata: Dict) -> str:
        """Create a new meeting record"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO meetings (title, meeting_date, series_id, file_path, participants, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    metadata.get('meeting_title'),
                    metadata.get('meeting_date'),
                    metadata.get('series_id'),
                    metadata.get('file_name'),
                    metadata.get('participants', []),
                    json.dumps(metadata)
                ))
                meeting_id = cur.fetchone()[0]
            conn.commit()
        
        return str(meeting_id)
    
    def create_chunks(self, chunks: List[MeetingChunk]):
        """Create chunk records"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    cur.execute("""
                        INSERT INTO chunks (meeting_id, chunk_id, chunk_type, chunk_index, 
                                          speakers, token_count, azure_search_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        chunk.meeting_id,
                        chunk.chunk_id,
                        chunk.chunk_type,
                        chunk.chunk_index,
                        chunk.speakers,
                        chunk.token_count,
                        chunk.chunk_id
                    ))
            conn.commit()
        
        logger.info(f"Created {len(chunks)} chunk records in database")
    
    def update_meeting_status(self, meeting_id: str, status: str):
        """Update meeting status"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE meetings SET status = %s WHERE id = %s
                """, (status, meeting_id))
            conn.commit()

class SearchService:
    """Handles Azure AI Search operations"""
    
    def __init__(self):
        self.endpoint = os.getenv('AZURE_AI_SEARCH_ENDPOINT')
        self.key = os.getenv('AZURE_AI_SEARCH_KEY')
        self.index_name = os.getenv('AZURE_AI_SEARCH_INDEX', 'meetings-index')
        
        if not all([self.endpoint, self.key]):
            raise ValueError("Azure AI Search credentials not found in environment")
        
        credential = AzureKeyCredential(self.key)
        self.index_client = SearchIndexClient(self.endpoint, credential)
        self.search_client = SearchClient(self.endpoint, self.index_name, credential)
        
        self._create_index()
    
    def _create_index(self):
        """Create or update the search index"""
        # Check if index already exists
        try:
            existing_index = self.index_client.get_index(self.index_name)
            logger.info(f"Index '{self.index_name}' already exists, using existing index")
            return existing_index
        except Exception:
            logger.info(f"Index '{self.index_name}' doesn't exist, creating new index")
        
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="my-vector-profile"
            ),
            SimpleField(name="meeting_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="chunk_type", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="speakers", collection=True, type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="meeting_date", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, sortable=True),
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="my-hnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="my-vector-profile",
                    algorithm_configuration_name="my-hnsw"
                )
            ]
        )
        
        try:
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            result = self.index_client.create_or_update_index(index)
            logger.info(f"Search index '{self.index_name}' created/updated successfully")
        except Exception as e:
            logger.error(f"Error creating search index: {e}")
            raise
    
    def upload_documents(self, chunks: List[MeetingChunk]):
        """Upload chunks to Azure AI Search"""
        if not embedding_model:
            raise ValueError("Embedding model not initialized")
        
        documents = []
        
        # Generate embeddings for all chunks
        contents = [chunk.content for chunk in chunks]
        embeddings = embedding_model.embed_documents(contents)
        
        # Create search documents
        for chunk, embedding in zip(chunks, embeddings):
            doc = {
                "id": chunk.chunk_id,
                "content": chunk.content,
                "content_vector": embedding,
                "meeting_id": chunk.meeting_id,
                "chunk_type": chunk.chunk_type,
                "speakers": chunk.speakers,
                "meeting_date": datetime.now().isoformat(),  # You may want to extract from metadata
                "chunk_index": chunk.chunk_index
            }
            documents.append(doc)
        
        try:
            result = self.search_client.upload_documents(documents)
            logger.info(f"Uploaded {len(documents)} documents to search index")
            return result
        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            raise
    
    def search(self, query: str, filter_expr: str = None, top: int = 5) -> List[Dict]:
        """Perform hybrid search"""
        if not embedding_model:
            raise ValueError("Embedding model not initialized")
        
        try:
            # Generate query embedding
            query_embedding = embedding_model.embed_query(query)
            
            # Import VectorizedQuery for proper vector search
            from azure.search.documents.models import VectorizedQuery
            
            # Perform search
            results = self.search_client.search(
                search_text=query,
                vector_queries=[VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=top,
                    fields="content_vector"
                )],
                filter=filter_expr,
                top=top,
                include_total_count=True
            )
            
            search_results = []
            for result in results:
                search_results.append({
                    "id": result["id"],
                    "content": result["content"],
                    "meeting_id": result["meeting_id"],
                    "chunk_type": result["chunk_type"],
                    "speakers": result["speakers"],
                    "score": result["@search.score"]
                })
            
            return search_results
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise

class MeetingProcessor:
    """Main processor class that orchestrates all operations"""
    
    def __init__(self):
        self.parser = TranscriptParser()
        self.chunker = ChunkingService()
        self.db_service = DatabaseService()
        self.search_service = SearchService()
    
    def process_meeting_file(self, file_path: str) -> str:
        """Process a meeting transcript file end-to-end"""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Step 1: Parse transcript
            entries, metadata = self.parser.parse_file(file_path)
            logger.info(f"Parsed {len(entries)} transcript entries")
            
            # Step 2: Create meeting record
            meeting_id = self.db_service.create_meeting(metadata)
            logger.info(f"Created meeting record: {meeting_id}")
            
            # Step 3: Create chunks
            chunks = self.chunker.create_chunks(entries, meeting_id)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 4: Store chunks in database
            self.db_service.create_chunks(chunks)
            
            # Step 5: Upload to search index (only if we have chunks)
            if chunks:
                self.search_service.upload_documents(chunks)
            else:
                logger.warning("No chunks created, skipping search index upload")
            
            # Step 6: Update meeting status
            self.db_service.update_meeting_status(meeting_id, 'completed')
            
            logger.info(f"Successfully processed meeting: {meeting_id}")
            return meeting_id
            
        except Exception as e:
            logger.error(f"Error processing meeting file: {e}")
            raise
    
    def search_meetings(self, query: str, filter_meeting_id: str = None) -> List[Dict]:
        """Search across meeting transcripts"""
        try:
            filter_expr = None
            if filter_meeting_id:
                filter_expr = f"meeting_id eq '{filter_meeting_id}'"
            
            results = self.search_service.search(query, filter_expr)
            logger.info(f"Found {len(results)} results for query: {query}")
            
            return results
        except Exception as e:
            logger.error(f"Error searching meetings: {e}")
            raise

def test_connections():
    """Test all service connections"""
    logger.info("Testing connections...")
    
    # Test OpenAI
    try:
        if embedding_model:
            test_embedding = embedding_model.embed_query("test")
            logger.info(f"✅ OpenAI Embeddings: {len(test_embedding)} dimensions")
        else:
            logger.error("❌ OpenAI Embeddings: Not initialized")
    except Exception as e:
        logger.error(f"❌ OpenAI Embeddings: {e}")
    
    try:
        if llm:
            response = llm.invoke("Say 'Hello'")
            logger.info(f"✅ OpenAI LLM: {response.content[:50] if hasattr(response, 'content') else str(response)[:50]}")
        else:
            logger.error("❌ OpenAI LLM: Not initialized")
    except Exception as e:
        logger.error(f"❌ OpenAI LLM: {e}")
    
    # Test PostgreSQL
    try:
        db_service = DatabaseService()
        with db_service.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
        logger.info("✅ PostgreSQL: Connected successfully")
    except Exception as e:
        logger.error(f"❌ PostgreSQL: {e}")
    
    # Test Azure AI Search
    try:
        search_service = SearchService()
        logger.info("✅ Azure AI Search: Index created successfully")
    except Exception as e:
        logger.error(f"❌ Azure AI Search: {e}")

def main():
    parser = argparse.ArgumentParser(description='Teams Meeting RAG Processor')
    parser.add_argument('--test', action='store_true', help='Test connections')
    parser.add_argument('--file', type=str, help='Process a meeting file')
    parser.add_argument('--search', type=str, help='Search query')
    parser.add_argument('--meeting-id', type=str, help='Filter by meeting ID')
    
    args = parser.parse_args()
    
    if args.test:
        test_connections()
        return
    
    processor = MeetingProcessor()
    
    if args.file:
        if os.path.exists(args.file):
            meeting_id = processor.process_meeting_file(args.file)
            print(f"Processed meeting: {meeting_id}")
        else:
            print(f"File not found: {args.file}")
            return
    
    if args.search:
        results = processor.search_meetings(args.search, args.meeting_id)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Meeting: {result['meeting_id']}")
            print(f"   Speakers: {', '.join(result['speakers'])}")
            print(f"   Content: {result['content'][:200]}...")

if __name__ == "__main__":
    main()