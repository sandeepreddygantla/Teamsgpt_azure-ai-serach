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
    },
    "query-analyzer": {
        "name": "Query Analyzer",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 2000,
        "description": "Query analysis for search optimization"
    }
}

def get_access_token():
    """Get access token - dummy function for LLM initialization pattern"""
    try:
        azure_client_id = os.getenv("AZURE_CLIENT_ID")
        azure_client_secret = os.getenv("AZURE_CLIENT_SECRET")
        
        if azure_client_id and azure_client_secret:
            logger.info("Azure environment detected - but using dummy token (no API call)")
            # Return dummy token instead of making API call
            return "dummy_azure_token"
        else:
            logger.info("OpenAI environment detected - using API key authentication")
            return None
            
    except Exception as e:
        logger.error(f"Error in access token function: {e}")
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

def get_query_analyzer_llm(access_token: str = None):
    """Get query analyzer LLM client"""
    try:
        current_api_key = os.getenv("OPENAI_API_KEY")
        if not current_api_key:
            logger.warning("OPENAI_API_KEY environment variable not set")
            return None
        
        model_config = AVAILABLE_MODELS.get("query-analyzer")
        
        return ChatOpenAI(
            model=model_config["model"],
            openai_api_key=current_api_key,
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            request_timeout=60
        )
    except Exception as e:
        logger.error(f"Error creating query analyzer LLM: {e}")
        return None

# Initialize global variables using your existing structure
try:
    access_token = get_access_token()
    embedding_model = get_embedding_model(access_token)
    llm = get_llm(access_token)
    query_analyzer_llm = get_query_analyzer_llm(access_token)
except Exception as e:
    logger.error(f"Error initializing models: {e}")
    embedding_model = None
    llm = None
    query_analyzer_llm = None

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
        
        # Pattern for: Speaker Name MM:SS (with or without brackets)
        # This handles both "Speaker [MM:SS]" and "Speaker MM:SS" formats
        pattern = r'([A-Za-z, ]+?)\s+(?:\[)?(\d+:\d+(?::\d+)?)(?:\])?\s*\n(.*?)(?=\n[A-Za-z, ]+\s+(?:\[)?\d+:\d+|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
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
        
        # For PDFs, we'll try to parse the extracted text as if it were DOCX format
        try:
            entries = []
            metadata = {'participants': [], 'file_name': os.path.basename(file_path)}
            
            # Pattern for: Speaker Name MM:SS (with or without brackets)
            # This handles both "Speaker [MM:SS]" and "Speaker MM:SS" formats
            pattern = r'([A-Za-z, ]+?)\s+(?:\[)?(\d+:\d+(?::\d+)?)(?:\])?\s*\n(.*?)(?=\n[A-Za-z, ]+\s+(?:\[)?\d+:\d+|\Z)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            if not matches:
                raise ValueError("No transcript pattern found in PDF content")
            
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
        except Exception as e:
            raise ValueError(f"Could not parse PDF content: {e}")
    
    def _extract_metadata_from_docx(self, content: str, filename: str) -> Dict:
        """Extract meeting metadata from DOCX content"""
        metadata = {'participants': [], 'file_name': os.path.basename(filename)}
        
        # Get first few lines which may contain metadata
        lines = content.split('\n')[:10]
        
        # First line often contains title, date, time, and duration all together
        # Format: "Meeting Title-YYYYMMDD_HHMMSS-Meeting Recording\nMonth DD, YYYY, H:MMPM\nXXm XXs"
        if lines:
            first_line = lines[0]
            
            # Extract meeting title and ID from first line or filename
            title_pattern = r'(.*?)-(\d{8}_\d{6})-Meeting Recording'
            if title_match := re.search(title_pattern, first_line):
                metadata['meeting_title'] = title_match.group(1).strip()
                metadata['series_id'] = title_match.group(1).strip()
            elif title_match := re.search(title_pattern, filename):
                metadata['meeting_title'] = title_match.group(1).strip()
                metadata['series_id'] = title_match.group(1).strip()
        
        # Check all lines for date, time, and duration
        full_text = '\n'.join(lines)
        
        # Extract date and time (might be on same or different lines)
        date_pattern = r'(\w+\s+\d{1,2},\s+\d{4})'
        time_pattern = r'(\d{1,2}:\d{2}\s*[AP]M)'
        
        if date_match := re.search(date_pattern, full_text):
            metadata['meeting_date'] = date_match.group(1)
        
        if time_match := re.search(time_pattern, full_text):
            metadata['meeting_time'] = time_match.group(1)
        
        # Extract duration
        duration_pattern = r'(\d+h\s+\d+m\s+\d+s|\d+m\s+\d+s|\d+h\s+\d+m)'
        if duration_match := re.search(duration_pattern, full_text):
            metadata['duration'] = duration_match.group(1)
        
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
            
            # Long pause detection (more than 30 seconds)
            if next_entry.timestamp_seconds - current_entry.timestamp_seconds > 30:
                return True
        
        # Default: chunk if we're at target size
        return token_count >= self.target_tokens
    
    
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
        """Create a comprehensive summary chunk using 60k token context"""
        if not entries or not llm:
            return None
        
        try:
            # Use larger batches with 60k context
            total_entries = len(entries)
            batch_size = 100  # Increased from 35
            partial_summaries = []
            
            for i in range(0, total_entries, batch_size):
                batch_entries = entries[i:min(i + batch_size, total_entries)]
                batch_content = self._format_segment_content(batch_entries)
                
                batch_prompt = f"""Analyze this meeting segment and extract key information:

{batch_content}

Extract:
1. Main discussion points and topics
2. Decisions made with context
3. Action items with owners
4. Important quotes and statements
5. Technical details mentioned
6. References to other meetings or projects"""
                
                batch_response = llm.invoke(batch_prompt)
                partial_summaries.append(batch_response.content if hasattr(batch_response, 'content') else str(batch_response))
            
            # Final comprehensive summary with all new fields
            combined_summary_prompt = f"""Create a comprehensive meeting summary from these segments:

{chr(10).join(f"Segment {i+1}: {summary}" for i, summary in enumerate(partial_summaries))}

Generate these specific fields:

MEETING_PURPOSE:
Brief description of meeting objective and why it was held.

KEY_OUTCOMES:
Main accomplishments and results achieved.

MAIN_TOPICS:
Primary discussion areas with context.

DECISIONS_MADE:
Specific decisions with who made them and reasoning.

ACTION_ITEMS:
Tasks with assigned owners and deadlines.

PAST_EVENTS:
References to previous meetings, projects, or events.

FUTURE_ACTIONS:
Upcoming meetings, planned activities, and commitments.

DETAILED_NARRATIVE:
Comprehensive flow capturing who said what, technical specifics, decision context, and all searchable details."""
            
            final_response = llm.invoke(combined_summary_prompt)
            response_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
            
            # Extract structured sections using flexible regex pattern
            # This pattern handles multiple LLM response formats:
            # - Markdown headers: ### SECTION_NAME or ## SECTION_NAME
            # - Bold markdown: **SECTION_NAME:** or **SECTION_NAME**
            # - Plain text: SECTION_NAME: or SECTION_NAME
            # - With or without colons, with or without spaces
            sections = {}
            
            # Define expected sections and their normalized keys
            section_mappings = {
                'MEETING_PURPOSE': 'meeting_purpose',
                'KEY_OUTCOMES': 'key_outcomes',
                'MAIN_TOPICS': 'main_topics',
                'DECISIONS_MADE': 'decisions_made',
                'ACTION_ITEMS': 'action_items',
                'PAST_EVENTS': 'past_events',
                'FUTURE_ACTIONS': 'future_actions',
                'DETAILED_NARRATIVE': 'detailed_narrative'
            }
            
            # Regex pattern to match section headers and content
            # Group 1: Optional markdown formatting (###, ##, #, **)
            # Group 2: Section name (MEETING_PURPOSE, KEY_OUTCOMES, etc.)
            # Group 3: Optional colon and formatting
            # Group 4: Content until next section or end of text
            pattern = r'(?:^|\n)\s*(?:#{1,3}\s*|\*{2}\s*)?([A-Z_]+)(?:\s*:?\s*\*{2}|\s*:|\s*$)\s*\n?(.*?)(?=\n\s*(?:#{1,3}\s*|\*{2}\s*)?[A-Z_]+(?:\s*:?\s*\*{2}|\s*:|\s*$)|\Z)'
            
            # Find all matches in the response text
            matches = re.findall(pattern, response_text, re.MULTILINE | re.DOTALL)
            
            # Process each match and store in sections dictionary
            for section_header, content in matches:
                section_header = section_header.strip()
                
                # Map the section header to normalized key
                if section_header in section_mappings:
                    normalized_key = section_mappings[section_header]
                    # Clean up the content: remove leading/trailing whitespace and empty lines
                    cleaned_content = '\n'.join(
                        line for line in content.strip().split('\n')
                        if line.strip()
                    )
                    sections[normalized_key] = cleaned_content
            
            # Log extraction results
            if sections:
                logger.info(f"Successfully extracted {len(sections)} structured fields from summary")
                logger.debug(f"Extracted sections: {list(sections.keys())}")
            else:
                logger.warning("No structured sections found in LLM response")
                logger.debug(f"LLM response first 500 chars: {response_text[:500]}")
            
            summary_text = response_text
            
            metadata = {
                'type': 'comprehensive_summary',
                'segments_processed': len(partial_summaries),
                'total_entries': total_entries,
                'participants': list(set(entry.speaker for entry in entries)),
                'structured_fields': sections
            }
            
            return MeetingChunk(
                chunk_id=f"{meeting_id}_summary",
                meeting_id=meeting_id,
                chunk_type="summary",
                content=summary_text,
                speakers=list(set(entry.speaker for entry in entries)),
                start_timestamp=entries[0].timestamp,
                end_timestamp=entries[-1].timestamp,
                chunk_index=9999,
                token_count=len(self.encoder.encode(summary_text)),
                has_context=False,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error creating summary chunk: {e}")
            return None


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
            logger.info(f"Index '{self.index_name}' already exists, checking if update is needed")
            
            # Check if the existing index has our new fields
            existing_field_names = {field.name for field in existing_index.fields}
            required_fields = {'document_type', 'meeting_title', 'series_id', 'participants'}
            
            if not required_fields.issubset(existing_field_names):
                logger.info(f"Index missing required fields, recreating index")
                # Delete the existing index and recreate with new schema
                self.index_client.delete_index(self.index_name)
                logger.info(f"Deleted existing index '{self.index_name}'")
            else:
                logger.info(f"Index has all required fields, using existing index")
                return existing_index
        except Exception:
            logger.info(f"Index '{self.index_name}' doesn't exist, creating new index")
        
        fields = [
            # Primary Keys
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True),
            
            # Content Fields
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="my-vector-profile"
            ),
            
            # Meeting Metadata
            SimpleField(name="meeting_id", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="meeting_title", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="series_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="meeting_date", type=SearchFieldDataType.DateTimeOffset, 
                       filterable=True, sortable=True),
            SimpleField(name="file_path", type=SearchFieldDataType.String),
            SimpleField(name="file_name", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="status", type=SearchFieldDataType.String, filterable=True),
            
            # Chunk Metadata  
            SimpleField(name="chunk_type", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, sortable=True),
            SimpleField(name="token_count", type=SearchFieldDataType.Int32, filterable=True),
            
            # Participants & Speakers
            SearchableField(name="speakers", collection=True, 
                           type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="participants", collection=True, 
                           type=SearchFieldDataType.String, filterable=True),
            
            # Timestamps
            SimpleField(name="start_timestamp", type=SearchFieldDataType.String),
            SimpleField(name="end_timestamp", type=SearchFieldDataType.String),
            SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, 
                       filterable=True, sortable=True),
            
            # Additional Context
            SimpleField(name="duration", type=SearchFieldDataType.String),
            SimpleField(name="meeting_time", type=SearchFieldDataType.String),
            SimpleField(name="metadata_json", type=SearchFieldDataType.String),
            
            # Structured Summary Fields
            SearchableField(name="meeting_purpose", type=SearchFieldDataType.String),
            SearchableField(name="key_outcomes", type=SearchFieldDataType.String),
            SearchableField(name="main_topics", type=SearchFieldDataType.String),
            SearchableField(name="decisions_made", type=SearchFieldDataType.String),
            SearchableField(name="action_items", type=SearchFieldDataType.String),
            SearchableField(name="past_events", type=SearchFieldDataType.String),
            SearchableField(name="future_actions", type=SearchFieldDataType.String),
            SearchableField(name="detailed_narrative", type=SearchFieldDataType.String),
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
    
    def _parse_meeting_date(self, date_string: str) -> str:
        """Parse meeting date string to ISO format with timezone"""
        if not date_string:
            return datetime.now().isoformat() + 'Z'
            
        try:
            # Try different date formats
            for fmt in ["%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y"]:
                try:
                    parsed_date = datetime.strptime(date_string, fmt)
                    # Azure AI Search expects DateTimeOffset with timezone
                    return parsed_date.isoformat() + 'Z'
                except ValueError:
                    continue
            
            # If parsing fails, return current time
            logger.warning(f"Could not parse date '{date_string}', using current time")
            return datetime.now().isoformat() + 'Z'
        except Exception as e:
            logger.error(f"Error parsing date '{date_string}': {e}")
            return datetime.now().isoformat() + 'Z'

    def upload_meeting_and_chunks(self, chunks: List[MeetingChunk], metadata: Dict) -> str:
        """Upload meeting metadata and chunks to Azure AI Search"""
        if not embedding_model:
            raise ValueError("Embedding model not initialized")
        
        documents = []
        meeting_id = chunks[0].meeting_id if chunks else str(uuid.uuid4())
        
        # Create meeting-level document
        meeting_doc = {
            "id": f"meeting_{meeting_id}",
            "document_type": "meeting",
            "meeting_id": meeting_id,
            "meeting_title": metadata.get('meeting_title', 'Unknown Meeting'),
            "series_id": metadata.get('series_id', ''),
            "meeting_date": self._parse_meeting_date(metadata.get('meeting_date')),
            "file_path": metadata.get('file_path', ''),
            "file_name": metadata.get('file_name', ''),
            "status": "completed",
            "participants": metadata.get('participants', []),
            "duration": metadata.get('duration', ''),
            "meeting_time": metadata.get('meeting_time', ''),
            "created_at": datetime.now().isoformat() + 'Z',
            "metadata_json": json.dumps(metadata),
            "content": f"Meeting: {metadata.get('meeting_title', '')} on {metadata.get('meeting_date', '')}. Participants: {', '.join(metadata.get('participants', []))}",
            "content_vector": embedding_model.embed_query(f"Meeting {metadata.get('meeting_title', '')} {metadata.get('series_id', '')}")
        }
        documents.append(meeting_doc)
        
        # Process chunks
        if chunks:
            contents = [chunk.content for chunk in chunks]
            embeddings = embedding_model.embed_documents(contents)
            
            for chunk, embedding in zip(chunks, embeddings):
                doc = {
                    "id": chunk.chunk_id,
                    "document_type": "chunk",
                    "meeting_id": chunk.meeting_id,
                    "meeting_title": metadata.get('meeting_title', ''),
                    "series_id": metadata.get('series_id', ''),
                    "meeting_date": self._parse_meeting_date(metadata.get('meeting_date')),
                    "file_name": metadata.get('file_name', ''),
                    "content": chunk.content,
                    "content_vector": embedding,
                    "chunk_type": chunk.chunk_type,
                    "chunk_index": chunk.chunk_index,
                    "speakers": chunk.speakers,
                    "start_timestamp": chunk.start_timestamp,
                    "end_timestamp": chunk.end_timestamp,
                    "token_count": chunk.token_count,
                    "created_at": datetime.now().isoformat() + 'Z',
                    "status": "completed"
                }
                
                if chunk.chunk_type == "summary" and chunk.metadata.get('structured_fields'):
                    fields = chunk.metadata['structured_fields']
                    doc["meeting_purpose"] = fields.get('meeting_purpose', '')
                    doc["key_outcomes"] = fields.get('key_outcomes', '')
                    doc["main_topics"] = fields.get('main_topics', '')
                    doc["decisions_made"] = fields.get('decisions_made', '')
                    doc["action_items"] = fields.get('action_items', '')
                    doc["past_events"] = fields.get('past_events', '')
                    doc["future_actions"] = fields.get('future_actions', '')
                    doc["detailed_narrative"] = fields.get('detailed_narrative', '')
                
                documents.append(doc)
        
        try:
            result = self.search_client.upload_documents(documents)
            logger.info(f"Uploaded {len(documents)} documents to search index (1 meeting + {len(chunks)} chunks)")
            return meeting_id
        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            raise

    
    def search(self, query: str, filter_expr: str = None, top: int = 5, document_types: List[str] = None, select_fields: List[str] = None) -> List[Dict]:
        """Perform hybrid search with optional field projection"""
        if not embedding_model:
            raise ValueError("Embedding model not initialized")
        
        try:
            query_embedding = embedding_model.embed_query(query)
            
            filters = []
            
            if document_types is None:
                document_types = ["chunk"]
            
            if document_types:
                if len(document_types) == 1:
                    filters.append(f"document_type eq '{document_types[0]}'")
                else:
                    type_filters = [f"document_type eq '{dt}'" for dt in document_types]
                    filters.append(f"({' or '.join(type_filters)})")
            
            if filter_expr:
                filters.append(filter_expr)
            
            final_filter = " and ".join(filters) if filters else None
            
            from azure.search.documents.models import VectorizedQuery
            
            search_params = {
                "search_text": query,
                "vector_queries": [VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=top,
                    fields="content_vector"
                )],
                "filter": final_filter,
                "top": top,
                "include_total_count": True
            }
            
            if select_fields:
                search_params["select"] = select_fields
            
            results = self.search_client.search(**search_params)
            
            search_results = []
            for result in results:
                result_dict = {
                    "id": result.get("id", ""),
                    "meeting_id": result.get("meeting_id", ""),
                    "chunk_type": result.get("chunk_type", ""),
                    "speakers": result.get("speakers", []),
                    "meeting_title": result.get("meeting_title", ""),
                    "series_id": result.get("series_id", ""),
                    "document_type": result.get("document_type", ""),
                    "meeting_date": result.get("meeting_date", ""),
                    "meeting_time": result.get("meeting_time", ""),
                    "duration": result.get("duration", ""),
                    "metadata_json": result.get("metadata_json", ""),
                    "score": result.get("@search.score", 0)
                }
                
                if "content" in result:
                    result_dict["content"] = result["content"]
                if "meeting_purpose" in result:
                    result_dict["meeting_purpose"] = result["meeting_purpose"]
                if "key_outcomes" in result:
                    result_dict["key_outcomes"] = result["key_outcomes"]
                if "main_topics" in result:
                    result_dict["main_topics"] = result["main_topics"]
                if "decisions_made" in result:
                    result_dict["decisions_made"] = result["decisions_made"]
                if "action_items" in result:
                    result_dict["action_items"] = result["action_items"]
                if "past_events" in result:
                    result_dict["past_events"] = result["past_events"]
                if "future_actions" in result:
                    result_dict["future_actions"] = result["future_actions"]
                if "detailed_narrative" in result:
                    result_dict["detailed_narrative"] = result["detailed_narrative"]
                
                search_results.append(result_dict)
            
            return search_results
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise

class MeetingProcessor:
    """Main processor class that orchestrates all operations"""
    
    def __init__(self):
        self.parser = TranscriptParser()
        self.chunker = ChunkingService()
        self.search_service = SearchService()
    
    def process_meeting_file(self, file_path: str) -> str:
        """Process a meeting transcript file end-to-end"""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Step 1: Parse transcript
            entries, metadata = self.parser.parse_file(file_path)
            logger.info(f"Parsed {len(entries)} transcript entries")
            
            # Step 2: Generate meeting ID
            meeting_id = str(uuid.uuid4())
            metadata['file_path'] = file_path  # Add file path to metadata
            logger.info(f"Generated meeting ID: {meeting_id}")
            
            # Step 3: Create chunks
            chunks = self.chunker.create_chunks(entries, meeting_id)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 4: Upload meeting and chunks to AI Search
            if chunks or metadata:  # Upload even if no chunks, for meeting metadata
                meeting_id = self.search_service.upload_meeting_and_chunks(chunks, metadata)
                logger.info(f"Uploaded meeting and chunks to search index")
            else:
                logger.warning("No chunks or metadata to upload")
            
            logger.info(f"Successfully processed meeting: {meeting_id}")
            return meeting_id
            
        except Exception as e:
            logger.error(f"Error processing meeting file: {e}")
            raise
    
    def search_meetings(self, query: str, filter_meeting_id: str = None) -> str:
        """Search across meeting transcripts and generate RAG response"""
        try:
            filter_expr = None
            if filter_meeting_id:
                filter_expr = f"meeting_id eq '{filter_meeting_id}'"
            
            # Analyze query intent using LLM
            if query_analyzer_llm:
                query_analysis = self._analyze_query_with_llm(query)
                
                # Build filters from analysis
                if query_analysis.get('date_filter'):
                    date_filter = query_analysis['date_filter']
                    filter_expr = f"{filter_expr} and {date_filter}" if filter_expr else date_filter
                
                if query_analysis.get('speaker_filter'):
                    speaker_filter = query_analysis['speaker_filter']
                    filter_expr = f"{filter_expr} and {speaker_filter}" if filter_expr else speaker_filter
                
                if query_analysis.get('content_filter'):
                    content_filter = query_analysis['content_filter']
                    filter_expr = f"{filter_expr} and {content_filter}" if filter_expr else content_filter
                
                select_fields = query_analysis.get('optimal_fields')
                # Ensure meeting_id is always included for proper grouping
                if select_fields and 'meeting_id' not in select_fields:
                    select_fields = select_fields + ['meeting_id']
                use_summary_only = query_analysis.get('use_summary_chunks', False)
            else:
                # Basic strategy when analyzer unavailable
                select_fields = None
                use_summary_only = False
            
            search_filter = filter_expr
            if use_summary_only:
                summary_filter = "chunk_type eq 'summary'"
                search_filter = f"{filter_expr} and {summary_filter}" if filter_expr else summary_filter
            
            results = self.search_service.search(query, search_filter, select_fields=select_fields)
            logger.info(f"Found {len(results)} results for query: {query}")
            
            if not results:
                return self._generate_no_results_response(query, query_analysis)
            
            # Check if this is a series query
            is_series_query = query_analysis.get('is_series_query', False)
            intent = query_analysis.get('intent', 'general')
            
            # Build context optimized for 60k tokens with series-aware prioritization
            context_chunks = []
            total_tokens = 0
            max_context_tokens = 50000
            
            if is_series_query or intent == 'series_analysis':
                context_chunks, total_tokens = self._build_series_context(results, max_context_tokens)
                context = "\n---\n".join(context_chunks)
                logger.info(f"Series context built: {len(context_chunks)} chunks, {total_tokens} tokens")
            else:
                # Original single-meeting context building
                for result in results:
                    chunk_parts = [f"Meeting: {result.get('meeting_title', result['meeting_id'])}"]
                    
                    if result.get('meeting_date'):
                        chunk_parts.append(f"Date: {result['meeting_date'][:10]}")
                    
                    if 'speakers' in result and result['speakers']:
                        chunk_parts.append(f"Speakers: {', '.join(result['speakers'])}")
                    
                    # Add relevant content based on available fields
                    for field in ['action_items', 'decisions_made', 'main_topics', 'key_outcomes', 
                                 'past_events', 'future_actions', 'meeting_purpose', 'detailed_narrative', 'content']:
                        if result.get(field):
                            field_name = field.replace('_', ' ').title()
                            chunk_parts.append(f"{field_name}: {result[field]}")
                    
                    chunk_content = '\n'.join(chunk_parts) + '\n'
                    chunk_tokens = len(self.chunker.encoder.encode(chunk_content))
                    
                    if total_tokens + chunk_tokens > max_context_tokens:
                        break
                    
                    context_chunks.append(chunk_content)
                    total_tokens += chunk_tokens
                
                context = "\n---\n".join(context_chunks)
            
            if not llm:
                raise ValueError("LLM not available for response generation")
            
            # Generate series-aware prompt
            meeting_ids = list(set([r['meeting_id'] for r in results]))
            is_multi_meeting = len(meeting_ids) > 1
            
            if (is_series_query or intent == 'series_analysis') and is_multi_meeting:
                prompt = f"""Based on the following meeting series transcripts, answer the user's question with comprehensive series-level analysis.

User Question: {query}

Meeting Series Transcripts ({len(meeting_ids)} meetings):
{context}

Instructions for Series Analysis:
- Analyze information across ALL meetings in the series chronologically
- Identify patterns, trends, and progression across meetings
- Summarize key themes and developments throughout the series
- Highlight important decisions and action items from the entire series
- Show how topics evolved or were resolved across meetings
- Organize your response to show the comprehensive series perspective
- Include specific meeting dates and speakers when relevant
- If information is missing from certain meetings, note it clearly

Answer:"""
            else:
                prompt = f"""Based on the following meeting transcripts, answer the user's question accurately and comprehensively.

User Question: {query}

Meeting Transcripts:
{context}

Instructions:
- Provide a clear, comprehensive answer based on the meeting content
- Include specific details from the transcripts when relevant
- Mention which speakers said what when appropriate
- If the information spans multiple meetings, organize your response clearly
- If you cannot find specific information, say so clearly

Answer:"""

            response = llm.invoke(prompt)
            generated_answer = response.content if hasattr(response, 'content') else str(response)
            
            meeting_ids = list(set([r['meeting_id'] for r in results]))
            source_info = f"\n\n**Sources**: Based on {len(context_chunks)} relevant excerpts from {len(meeting_ids)} meeting(s)."
            
            return generated_answer + source_info
            
        except Exception as e:
            logger.error(f"Error searching meetings: {e}")
            raise
    
    def _analyze_query_with_llm(self, query: str) -> dict:
        """Analyze query using LLM to determine search strategy"""
        try:
            logger.info(f"Analyzing query: \"{query}\"")
            
            from datetime import datetime, timedelta
            now = datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            
            # Calculate proper calendar weeks (Sunday to Saturday)
            days_since_sunday = now.weekday() + 1 if now.weekday() != 6 else 0  # Sunday = 0 in weekday()
            current_week_start = (now - timedelta(days=days_since_sunday)).strftime("%Y-%m-%d")
            current_week_end = (now - timedelta(days=days_since_sunday) + timedelta(days=6)).strftime("%Y-%m-%d")
            last_week_start = (now - timedelta(days=days_since_sunday + 7)).strftime("%Y-%m-%d")
            last_week_end = (now - timedelta(days=days_since_sunday + 1)).strftime("%Y-%m-%d")
            
            # Calculate proper calendar months
            import calendar
            current_month_start = now.replace(day=1).strftime("%Y-%m-%d")
            last_day_current_month = calendar.monthrange(now.year, now.month)[1]
            current_month_end = now.replace(day=last_day_current_month).strftime("%Y-%m-%d")
            
            if now.month == 1:
                last_month = now.replace(year=now.year-1, month=12, day=1)
            else:
                last_month = now.replace(month=now.month-1, day=1)
            last_day_last_month = calendar.monthrange(last_month.year, last_month.month)[1]
            last_month_start = last_month.strftime("%Y-%m-%d")
            last_month_end = last_month.replace(day=last_day_last_month).strftime("%Y-%m-%d")
            
            analysis_prompt = f"""Analyze this meeting search query and return JSON with search strategy:

CURRENT DATE: {current_date}

CALENDAR CONTEXT:
WEEKS (Sunday to Saturday):
- Current week: {current_week_start} to {current_week_end}
- Last week: {last_week_start} to {last_week_end}

MONTHS (1st to last day):
- Current month: {current_month_start} to {current_month_end}
- Last month: {last_month_start} to {last_month_end}

Use these exact date ranges for relative date queries like "this week", "last month", "current month", etc.

Query: "{query}"

Available fields: meeting_date, speakers (collection), participants (collection), series_id, meeting_title, content, meeting_purpose, key_outcomes, main_topics, decisions_made, action_items, past_events, future_actions, detailed_narrative

IMPORTANT - Azure AI Search Filter Syntax:

For Speaker Names (case-insensitive fuzzy matching):
- search.ismatch('name', 'speakers') (case-insensitive substring match)
- Use lowercase for consistency: 'sandeep' matches 'Sandeep Reddy'
- Examples: search.ismatch('michael', 'speakers'), search.ismatch('sandeep', 'speakers')

For Dates (DateTimeOffset comparison):
- meeting_date ge 2025-07-01T00:00:00Z (after July 1st 2025)
- meeting_date eq 2025-07-14T00:00:00Z (exactly July 14th 2025)  
- meeting_date ge 2025-07-01T00:00:00Z and meeting_date le 2025-07-07T23:59:59Z (date range)

For Series/Meeting Title (case-insensitive fuzzy matching - PREFERRED):
- search.ismatch('keyword', 'meeting_title') (case-insensitive substring match)
- meeting_title eq 'Exact Title' (case-sensitive exact match)
- series_id eq 'Exact Series Name' (case-sensitive exact series match)

IMPORTANT: Extract key terms from user queries for flexible matching:
- Users use informal names: "fulfillment meetings", "doc fulfillment", "AIML series"
- search.ismatch() is case-insensitive: 'fulfillment' matches 'Document Fulfillment AIML'
- Use lowercase for consistency: 'fulfillment', 'aiml', 'document'
- search.ismatch() works without wildcards (wildcards don't work reliably)

- Examples:
  - "July 14" → meeting_date eq 2025-07-14T00:00:00Z
  - "last week" → meeting_date ge {last_week_start}T00:00:00Z and meeting_date le {last_week_end}T23:59:59Z
  - "current week" → meeting_date ge {current_week_start}T00:00:00Z and meeting_date le {current_week_end}T23:59:59Z
  - "last month" → meeting_date ge {last_month_start}T00:00:00Z and meeting_date le {last_month_end}T23:59:59Z
  - "current month" → meeting_date ge {current_month_start}T00:00:00Z and meeting_date le {current_month_end}T23:59:59Z
  - "December 2025" → search.ismatch('2025-12', 'meeting_date')
  - "Document Fulfillment meetings" → search.ismatch('fulfillment', 'meeting_title')
  - "AIML meetings" → search.ismatch('aiml', 'meeting_title')  
  - "fulfillment meetings" → search.ismatch('fulfillment', 'meeting_title')
  - "doc fulfillment" → search.ismatch('document', 'meeting_title')
  - "aiml series" → search.ismatch('aiml', 'meeting_title')

Return JSON:
{{
    "intent": "summary|decisions|action_items|speaker_analysis|series_analysis|technical|general",
    "date_filter": "meeting_date filter expression or empty string",
    "speaker_filter": "search.ismatch('name*', 'speakers') format or empty string", 
    "content_filter": "series_id or meeting_title filter or empty string",
    "optimal_fields": ["list of best fields for this query"],
    "use_summary_chunks": true/false,
    "is_series_query": true/false
}}

Return only JSON, no explanation."""

            response = query_analyzer_llm.invoke(analysis_prompt)
            raw_response = response.content.strip()
            logger.info(f"Query Analyzer Response: {raw_response}")
            
            # Clean markdown code blocks if present
            clean_response = raw_response
            if clean_response.startswith('```'):
                parts = clean_response.split('```')
                if len(parts) >= 3:
                    clean_response = parts[1]
                    if clean_response.startswith('json'):
                        clean_response = clean_response[4:]
                    clean_response = clean_response.strip()
            
            import json
            parsed_strategy = json.loads(clean_response)
            
            # Log key components of the strategy
            logger.info(f"Detected intent: {parsed_strategy.get('intent', 'unknown')}")
            if parsed_strategy.get('date_filter'):
                logger.info(f"Generated date filter: {parsed_strategy['date_filter']}")
            if parsed_strategy.get('speaker_filter'):
                logger.info(f"Generated speaker filter: {parsed_strategy['speaker_filter']}")
            if parsed_strategy.get('content_filter'):
                logger.info(f"Generated content filter: {parsed_strategy['content_filter']}")
            if parsed_strategy.get('optimal_fields'):
                logger.info(f"Selected fields: {parsed_strategy['optimal_fields']}")
            logger.info(f"Use summary chunks: {parsed_strategy.get('use_summary_chunks', False)}")
            
            return parsed_strategy
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}")
            return {}
    
    def _build_series_context(self, results, max_context_tokens):
        """Build context for series queries with prioritized aggregation"""
        context_chunks = []
        total_tokens = 0
        
        # Group results by meeting_id and meeting_date for chronological ordering
        meetings = {}
        for result in results:
            meeting_id = result.get('meeting_id', 'unknown')
            meeting_date = result.get('meeting_date', 'unknown')
            meeting_title = result.get('meeting_title', meeting_id)
            
            if meeting_id not in meetings:
                meetings[meeting_id] = {
                    'date': meeting_date,
                    'title': meeting_title,
                    'chunks': [],
                    'summary_chunks': [],
                    'regular_chunks': []
                }
            
            # Prioritize summary chunks for series analysis
            chunk_type = result.get('chunk_type', 'regular')
            if chunk_type == 'summary':
                meetings[meeting_id]['summary_chunks'].append(result)
            else:
                meetings[meeting_id]['regular_chunks'].append(result)
        
        # Sort meetings chronologically
        sorted_meetings = sorted(meetings.items(), key=lambda x: x[1]['date'])
        
        # Build context with priority: summary chunks first, then regular chunks
        for meeting_id, meeting_data in sorted_meetings:
            meeting_title = meeting_data['title']
            meeting_date = meeting_data['date'][:10] if meeting_data['date'] != 'unknown' else 'Unknown Date'
            
            # Add meeting header
            meeting_header = f"\n=== MEETING: {meeting_title} ({meeting_date}) ===\n"
            header_tokens = len(self.chunker.encoder.encode(meeting_header))
            
            if total_tokens + header_tokens > max_context_tokens:
                break
                
            # Process summary chunks first (high priority for series analysis)
            for result in meeting_data['summary_chunks']:
                chunk_content = self._build_single_chunk(result, include_meeting_header=False)
                chunk_tokens = len(self.chunker.encoder.encode(chunk_content))
                
                if total_tokens + header_tokens + chunk_tokens > max_context_tokens:
                    break
                    
                if not context_chunks or not context_chunks[-1].startswith(f"\n=== MEETING: {meeting_title}"):
                    context_chunks.append(meeting_header.strip())
                    total_tokens += header_tokens
                    header_tokens = 0  # Don't count again
                
                context_chunks.append(chunk_content)
                total_tokens += chunk_tokens
            
            # Add regular chunks if there's remaining space
            for result in meeting_data['regular_chunks']:
                chunk_content = self._build_single_chunk(result, include_meeting_header=False)
                chunk_tokens = len(self.chunker.encoder.encode(chunk_content))
                
                if total_tokens + header_tokens + chunk_tokens > max_context_tokens:
                    break
                    
                if not context_chunks or not context_chunks[-1].startswith(f"\n=== MEETING: {meeting_title}"):
                    context_chunks.append(meeting_header.strip())
                    total_tokens += header_tokens
                    header_tokens = 0
                
                context_chunks.append(chunk_content)
                total_tokens += chunk_tokens
        
        return context_chunks, total_tokens
    
    def _build_single_chunk(self, result, include_meeting_header=True):
        """Build a single chunk with consistent formatting"""
        chunk_parts = []
        
        if include_meeting_header:
            chunk_parts.append(f"Meeting: {result.get('meeting_title', result.get('meeting_id', 'Unknown'))}")
            if result.get('meeting_date'):
                chunk_parts.append(f"Date: {result['meeting_date'][:10]}")
        
        if 'speakers' in result and result['speakers']:
            chunk_parts.append(f"Speakers: {', '.join(result['speakers'])}")
        
        # Add relevant content based on available fields
        for field in ['action_items', 'decisions_made', 'main_topics', 'key_outcomes', 
                     'past_events', 'future_actions', 'meeting_purpose', 'detailed_narrative', 'content']:
            if result.get(field):
                field_name = field.replace('_', ' ').title()
                chunk_parts.append(f"{field_name}: {result[field]}")
        
        return '\n'.join(chunk_parts) + '\n'
    
    def _generate_no_results_response(self, query: str, query_analysis: dict) -> str:
        """Generate contextual response when no results are found"""
        intent = query_analysis.get('intent', 'general')
        speaker_filter = query_analysis.get('speaker_filter', '')
        date_filter = query_analysis.get('date_filter', '')
        
        # Extract speaker name from filter if present
        speaker_name = None
        if speaker_filter and 'search.ismatch' in speaker_filter:
            import re
            match = re.search(r"search\.ismatch\('([^*']+)\*?'", speaker_filter)
            if match:
                speaker_name = match.group(1)
        
        # Extract date info from filter if present
        date_info = None
        if date_filter:
            date_info = "the specified date"
        
        # Generate specific responses based on intent and filters
        if intent == 'speaker_analysis' and speaker_name:
            # Get list of actual speakers to provide helpful feedback
            try:
                all_results = self.search_service.search('*', 'document_type eq \'chunk\'', select_fields=['speakers'], top=50)
                all_speakers = set()
                for result in all_results:
                    if result.get('speakers'):
                        all_speakers.update(result['speakers'])
                
                if all_speakers:
                    speaker_list = sorted(list(all_speakers))[:5]  # Show first 5
                    speakers_text = ', '.join(speaker_list)
                    if len(all_speakers) > 5:
                        speakers_text += f" and {len(all_speakers)-5} others"
                    
                    return f"I couldn't find any speaker named '{speaker_name}' in the meeting transcripts.\n\nActual participants in the meetings include: {speakers_text}.\n\nTry asking about one of these participants instead."
                else:
                    return f"I couldn't find any speaker named '{speaker_name}' in the meeting transcripts."
            except:
                return f"I couldn't find any speaker named '{speaker_name}' in the meeting transcripts."
        
        elif intent == 'summary' and date_info:
            return f"No meetings were found for {date_info}. Please check the date or try a broader date range."
        
        elif intent == 'decisions':
            return "No specific decisions were found in the meeting transcripts for your query. The meetings may not have reached formal decisions on this topic."
        
        elif intent == 'action_items':
            return "No action items were found matching your query. The meetings may not have generated specific action items on this topic."
        
        elif date_info:
            return f"No meetings were found for {date_info}. Please verify the date or try searching for a different time period."
        
        else:
            # Generic but more helpful response
            return "I couldn't find any relevant information in the meeting transcripts for your query. Please try:\n• Using different keywords\n• Asking about specific participants or topics\n• Checking if the information might be in a different meeting"

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
        response = processor.search_meetings(args.search, args.meeting_id)
        print("\n" + "="*80)
        print(f"QUESTION: {args.search}")
        print("="*80)
        print(response)
        print("="*80)

if __name__ == "__main__":
    main()