import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import quote
import re

import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai
from groq import AsyncGroq
from dotenv import load_dotenv
import fitz  # PyMuPDF


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_scout.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    args: Dict
    start_time: float
    end_time: Optional[float] = None
    outcome: Optional[str] = None
    error: Optional[str] = None

    @property
    def latency(self) -> float:
        return (self.end_time or time.time()) - self.start_time


class LLMClient:
    """Model-agnostic LLM client supporting OpenAI, Anthropic, and Gemini"""
    
    def __init__(self):
        self.provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        self.model = os.getenv('LLM_MODEL', self._get_default_model())
        self.client = self._initialize_client()
        
    def _get_default_model(self) -> str:
        defaults = {
            'openai': 'gpt-3.5-turbo',
            'anthropic': 'claude-3-sonnet-20240229',
            'gemini': 'gemini-pro',
            'groq': 'llama3-8b-8192'
        }
        return defaults.get(self.provider, 'gpt-3.5-turbo')
    
    def _initialize_client(self):
        if self.provider == 'openai':
            return AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif self.provider == 'anthropic':
            return AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        elif self.provider == 'gemini':
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            return genai.GenerativeModel(self.model)
        elif self.provider == 'groq':
            return AsyncGroq(api_key=os.getenv('GROQ_API_KEY'))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def chat_completion(self, messages: List[Dict], tools: List[Dict] = None) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion"""
        try:
            if self.provider == 'openai':
                async for chunk in self._openai_stream(messages, tools):
                    yield chunk
            elif self.provider == 'anthropic':
                async for chunk in self._anthropic_stream(messages, tools):
                    yield chunk
            elif self.provider == 'gemini':
                async for chunk in self._gemini_stream(messages):
                    yield chunk
            elif self.provider == 'groq':
                async for chunk in self._groq_stream(messages, tools):
                    yield chunk
        except Exception as e:
            logger.error(f"LLM completion error: {e}")
            yield f"Error: {e}"
    
    async def _openai_stream(self, messages: List[Dict], tools: List[Dict] = None):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            
        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                elif hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                    tool_call = chunk.choices[0].delta.tool_calls[0]
                    if tool_call.function:
                        yield f"\n🔧 Calling {tool_call.function.name}...\n"
    
    async def _anthropic_stream(self, messages: List[Dict], tools: List[Dict] = None):
        # Convert OpenAI format to Anthropic format
        system_msg = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)
        
        kwargs = {
            "model": self.model,
            "messages": user_messages,
            "max_tokens": 1000,
            "stream": True
        }
        if system_msg:
            kwargs["system"] = system_msg
        if tools:
            kwargs["tools"] = [{"name": t["function"]["name"], 
                             "description": t["function"]["description"],
                             "input_schema": t["function"]["parameters"]} for t in tools]
        
        stream = await self.client.messages.create(**kwargs)
        async for chunk in stream:
            if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text') and chunk.delta.text:
                yield chunk.delta.text
    
    async def _gemini_stream(self, messages: List[Dict]):
        # Convert to Gemini format
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        response = await self.client.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000,
            ),
            stream=True
        )
        
        async for chunk in response:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
    
    async def _groq_stream(self, messages: List[Dict], tools: List[Dict] = None):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            
        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                elif hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                    tool_call = chunk.choices[0].delta.tool_calls[0]
                    if tool_call.function:
                        yield f"\n🔧 Calling {tool_call.function.name}...\n"


class PaperSearchServer:
    """MCP server for searching arXiv papers"""
    
    async def search_papers(self, query: str, max_results: int = 5) -> Dict:
        """Search arXiv for papers matching the query"""
        try:
            encoded_query = quote(query)
            url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                namespace = {'atom': 'http://www.w3.org/2005/Atom'}
                
                papers = []
                for entry in root.findall('atom:entry', namespace):
                    title_elem = entry.find('atom:title', namespace)
                    summary_elem = entry.find('atom:summary', namespace)
                    authors = entry.findall('atom:author', namespace)
                    published_elem = entry.find('atom:published', namespace)
                    pdf_link = None
                    
                    # Find PDF link
                    for link in entry.findall('atom:link', namespace):
                        if link.get('title') == 'pdf':
                            pdf_link = link.get('href')
                            break
                    
                    author_names = []
                    for author in authors:
                        name_elem = author.find('atom:name', namespace)
                        if name_elem is not None:
                            author_names.append(name_elem.text)
                    
                    papers.append({
                        'title': title_elem.text.strip() if title_elem is not None else "Unknown Title",
                        'summary': summary_elem.text.strip() if summary_elem is not None else "No summary available",
                        'authors': author_names,
                        'published': published_elem.text if published_elem is not None else "Unknown date",
                        'pdf_url': pdf_link
                    })
                
                return {
                    'papers': papers,
                    'total_found': len(papers)
                }
                
        except Exception as e:
            logger.error(f"Paper search error: {e}")
            return {'error': str(e), 'papers': []}


class PDFSummarizeServer:
    """MCP server for downloading and summarizing PDFs"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    async def summarize_pdf(self, pdf_url: str) -> Dict:
        """Download PDF and generate summary"""
        try:
            # Download PDF
            async with httpx.AsyncClient() as client:
                response = await client.get(pdf_url, timeout=60.0)
                response.raise_for_status()
                
                # Extract text from PDF
                doc = fitz.open(stream=response.content, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                
                if not text.strip():
                    return {'error': 'Could not extract text from PDF'}
                
                # Truncate text if too long (keep first 4000 chars)
                if len(text) > 4000:
                    text = text[:4000] + "..."
                
                # Generate summary
                messages = [
                    {"role": "system", "content": "You are an expert at summarizing scientific papers. Provide a concise but comprehensive summary covering: main findings, methodology, and significance."},
                    {"role": "user", "content": f"Please summarize this research paper:\n\n{text}"}
                ]
                
                summary = ""
                async for chunk in self.llm_client.chat_completion(messages):
                    summary += chunk
                
                return {'summary': summary.strip()}
                
        except Exception as e:
            logger.error(f"PDF summarization error: {e}")
            return {'error': str(e)}

    async def summarize_multiple_pdfs(self, pdf_urls: List[str]) -> Dict:
        """Summarize multiple PDFs and combine into one comprehensive summary"""
        try:
            individual_summaries = []
            failed_urls = []
            
            for i, url in enumerate(pdf_urls, 1):
                print(f"📄 Processing PDF {i}/{len(pdf_urls)}...")
                result = await self.summarize_pdf(url)
                
                if 'error' in result:
                    failed_urls.append(url)
                    logger.warning(f"Failed to summarize PDF {i}: {result['error']}")
                else:
                    individual_summaries.append({
                        'url': url,
                        'summary': result['summary']
                    })
            
            if not individual_summaries:
                return {'error': 'Failed to summarize any of the provided PDFs'}
            
            # Create combined summary
            combined_text = "Here are the individual paper summaries:\n\n"
            for i, summary_data in enumerate(individual_summaries, 1):
                combined_text += f"**Paper {i}:**\n{summary_data['summary']}\n\n---\n\n"
            
            # Generate meta-summary
            messages = [
                {"role": "system", "content": "You are an expert research analyst. Create a comprehensive meta-analysis summary that identifies common themes, contrasting findings, and overall insights across multiple research papers."},
                {"role": "user", "content": f"Please create a comprehensive analysis of these research papers, highlighting key themes, methodologies, and findings:\n\n{combined_text}"}
            ]
            
            meta_summary = ""
            async for chunk in self.llm_client.chat_completion(messages):
                meta_summary += chunk
            
            return {
                'individual_summaries': individual_summaries,
                'meta_summary': meta_summary.strip(),
                'failed_urls': failed_urls,
                'total_processed': len(individual_summaries),
                'total_failed': len(failed_urls)
            }
            
        except Exception as e:
            logger.error(f"Multiple PDF summarization error: {e}")
            return {'error': str(e)}


class SummaryManager:
    """Manages saving and retrieving summaries"""
    
    def __init__(self):
        self.summaries_dir = "summaries"
        if not os.path.exists(self.summaries_dir):
            os.makedirs(self.summaries_dir)
    
    def save_summary(self, content: str, filename: str = None) -> str:
        """Save summary to text file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{timestamp}.txt"
        
        filepath = os.path.join(self.summaries_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def save_multiple_summaries(self, summary_data: Dict) -> str:
        """Save multiple PDF summaries to organized text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multiple_summaries_{timestamp}.txt"
        filepath = os.path.join(self.summaries_dir, filename)
        
        content = f"Multiple Paper Analysis Summary\n"
        content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"Total Papers Processed: {summary_data.get('total_processed', 0)}\n"
        content += f"Total Papers Failed: {summary_data.get('total_failed', 0)}\n"
        content += "=" * 80 + "\n\n"
        
        # Add meta-summary
        if 'meta_summary' in summary_data:
            content += "COMPREHENSIVE ANALYSIS\n"
            content += "=" * 30 + "\n"
            content += summary_data['meta_summary'] + "\n\n"
            content += "=" * 80 + "\n\n"
        
        # Add individual summaries
        content += "INDIVIDUAL PAPER SUMMARIES\n"
        content += "=" * 35 + "\n\n"
        
        for i, summary_info in enumerate(summary_data.get('individual_summaries', []), 1):
            content += f"PAPER {i}\n"
            content += f"URL: {summary_info.get('url', 'Unknown')}\n"
            content += "-" * 50 + "\n"
            content += summary_info.get('summary', '') + "\n\n"
            content += "=" * 80 + "\n\n"
        
        # Add failed URLs if any
        if summary_data.get('failed_urls'):
            content += "FAILED TO PROCESS\n"
            content += "=" * 20 + "\n"
            for url in summary_data['failed_urls']:
                content += f"• {url}\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath


class AgentHost:
    """Main agent orchestrator"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.paper_search = PaperSearchServer()
        self.pdf_summarize = PDFSummarizeServer(self.llm_client)
        self.summary_manager = SummaryManager()
        self.tool_calls: List[ToolCall] = []
        self.last_search_results: List[Dict] = []
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_papers",
                    "description": "Search for research papers on arXiv",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query for papers"},
                            "max_results": {"type": "integer", "description": "Maximum number of results (default: 5)"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "summarize_pdf",
                    "description": "Download and summarize a research paper PDF",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pdf_url": {"type": "string", "description": "URL of the PDF to summarize"}
                        },
                        "required": ["pdf_url"]
                    }
                }
            }
        ]
    
    def parse_smart_command(self, user_input: str) -> Dict:
        """Parse smart commands like 'summarize paper 1,2,3' or 'summarize all papers'"""
        user_lower = user_input.lower().strip()
        
        # Check for summarize commands
        if 'summarize' in user_lower:
            if 'all papers' in user_lower or 'all' in user_lower:
                return {
                    'action': 'summarize_multiple',
                    'indices': list(range(len(self.last_search_results)))
                }
            
            # Look for paper numbers
            paper_pattern = r'paper[s]?\s+(\d+(?:\s*,\s*\d+)*|\d+(?:\s*-\s*\d+)?)'
            match = re.search(paper_pattern, user_lower)
            
            if match:
                numbers_str = match.group(1)
                indices = []
                
                # Handle ranges like "1-3"
                if '-' in numbers_str:
                    start, end = map(int, numbers_str.split('-'))
                    indices = list(range(start-1, end))  # Convert to 0-based indexing
                else:
                    # Handle comma-separated numbers like "1,2,3"
                    numbers = [int(x.strip()) for x in numbers_str.split(',')]
                    indices = [n-1 for n in numbers]  # Convert to 0-based indexing
                
                return {
                    'action': 'summarize_multiple',
                    'indices': indices
                }
        
        return {'action': 'normal'}
    
    async def call_tool(self, tool_name: str, args: Dict) -> Dict:
        """Execute a tool call and log the results"""
        tool_call = ToolCall(name=tool_name, args=args, start_time=time.time())
        self.tool_calls.append(tool_call)
        
        print(f"🔧 Calling {tool_name} with args: {json.dumps(args, indent=2)}")
        
        try:
            if tool_name == "search_papers":
                result = await self.paper_search.search_papers(
                    args.get("query", ""), 
                    args.get("max_results", 5)
                )
            elif tool_name == "summarize_pdf":
                result = await self.pdf_summarize.summarize_pdf(args.get("pdf_url", ""))
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
            
            tool_call.end_time = time.time()
            tool_call.outcome = "success" if "error" not in result else "error"
            
            print(f"✅ Tool completed in {tool_call.latency:.2f}s")
            logger.info(f"Tool call: {tool_name}, Args: {args}, Latency: {tool_call.latency:.2f}s, Outcome: {tool_call.outcome}")
            
            return result
            
        except Exception as e:
            tool_call.end_time = time.time()
            tool_call.error = str(e)
            tool_call.outcome = "error"
            
            print(f"❌ Tool failed in {tool_call.latency:.2f}s: {e}")
            logger.error(f"Tool call failed: {tool_name}, Args: {args}, Error: {e}")
            
            return {"error": str(e)}
    
    async def process_message(self, user_message: str) -> AsyncGenerator[str, None]:
        """Process user message and stream response"""
        # Parse smart commands first
        command_info = self.parse_smart_command(user_message)
        
        if command_info['action'] == 'summarize_multiple':
            if not self.last_search_results:
                yield "No papers available. Please search for papers first."
                return
            
            indices = command_info['indices']
            valid_indices = [i for i in indices if 0 <= i < len(self.last_search_results)]
            
            if not valid_indices:
                yield f"Invalid paper numbers. Available papers: 1-{len(self.last_search_results)}"
                return
            
            # Get PDF URLs for selected papers
            pdf_urls = []
            selected_papers = []
            for i in valid_indices:
                paper = self.last_search_results[i]
                if paper.get('pdf_url'):
                    pdf_urls.append(paper['pdf_url'])
                    selected_papers.append(f"Paper {i+1}: {paper['title']}")
            
            if not pdf_urls:
                yield "No PDF URLs available for the selected papers."
                return
            
            yield f"📄 Summarizing {len(pdf_urls)} papers:\n"
            for paper_info in selected_papers:
                yield f"• {paper_info}\n"
            yield "\n"
            
            # Summarize multiple PDFs
            result = await self.pdf_summarize.summarize_multiple_pdfs(pdf_urls)
            
            if 'error' in result:
                yield f"Sorry, I encountered an error: {result['error']}"
                return
            
            # Save summary to file
            filepath = self.summary_manager.save_multiple_summaries(result)
            
            # Display results
            yield "📊 **COMPREHENSIVE ANALYSIS**\n"
            yield "=" * 50 + "\n"
            yield result['meta_summary'] + "\n\n"
            
            yield "📋 **INDIVIDUAL SUMMARIES**\n"
            yield "=" * 30 + "\n"
            for i, summary_info in enumerate(result['individual_summaries'], 1):
                yield f"**Paper {i}:**\n"
                yield summary_info['summary'] + "\n\n---\n\n"
            
            yield f"💾 **Summary saved to:** {filepath}\n"
            
            if result['failed_urls']:
                yield f"⚠️ **Failed to process {len(result['failed_urls'])} papers**\n"
            
            return
        
        # Original message processing logic
        messages = [
            {
                "role": "system",
                "content": "You are a Scientific Paper Scout AI assistant. Help users discover and summarize recent research papers. Use the search_papers tool to find relevant papers, and the summarize_pdf tool to provide detailed summaries of specific papers. Always be helpful and provide comprehensive information about the research."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        # Check if user is asking for paper search or PDF summary
        user_lower = user_message.lower()
        
        if any(keyword in user_lower for keyword in ['search', 'find', 'papers', 'research']):
            # Extract search query
            query = user_message
            if 'search for' in user_lower:
                query = user_message.split('search for', 1)[1].strip()
            elif 'find' in user_lower and 'papers' in user_lower:
                query = user_message.replace('find', '').replace('papers', '').strip()
            
            result = await self.call_tool("search_papers", {"query": query, "max_results": 5})
            
            if 'error' in result:
                yield f"Sorry, I encountered an error searching for papers: {result['error']}"
            else:
                papers = result.get('papers', [])
                if not papers:
                    yield "No papers found for your query."
                    self.last_search_results = []
                else:
                    self.last_search_results = papers  # Store for smart commands
                    yield f"Found {len(papers)} recent papers:\n\n"
                    for i, paper in enumerate(papers, 1):
                        yield f"**{i}. {paper['title']}**\n"
                        yield f"Authors: {', '.join(paper['authors'][:3])}\n"
                        yield f"Published: {paper['published'][:10]}\n"
                        yield f"Summary: {paper['summary'][:200]}...\n"
                        if paper['pdf_url']:
                            yield f"PDF: {paper['pdf_url']}\n"
                        yield "\n---\n\n"
                    
                    yield "💡 **Smart Commands Available:**\n"
                    yield "• `summarize paper 1` - Summarize a specific paper\n"
                    yield "• `summarize papers 1,2,3` - Summarize multiple papers\n"
                    yield "• `summarize all papers` - Summarize all found papers\n\n"
        
        elif 'summarize' in user_lower and 'http' in user_message:
            # Extract PDF URL
            words = user_message.split()
            pdf_url = next((word for word in words if word.startswith('http')), None)
            
            if pdf_url:
                result = await self.call_tool("summarize_pdf", {"pdf_url": pdf_url})
                
                if 'error' in result:
                    yield f"Sorry, I couldn't summarize the PDF: {result['error']}"
                else:
                    summary_content = f"📄 **Paper Summary**\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nURL: {pdf_url}\n\n{result['summary']}"
                    
                    # Save summary
                    filepath = self.summary_manager.save_summary(summary_content)
                    
                    yield "📄 **Paper Summary:**\n\n"
                    yield result['summary']
                    yield f"\n\n💾 **Summary saved to:** {filepath}\n"
            else:
                yield "Please provide a valid PDF URL to summarize."
        
        else:
            # General conversation
            async for chunk in self.llm_client.chat_completion(messages, self.tools):
                yield chunk


async def main():
    """Main CLI interface"""
    print("🔬 Scientific Paper Scout - Enhanced Edition")
    print("=" * 60)
    print("Commands:")
    print("• Search: 'search for quantum computing papers'")
    print("• Summarize URL: 'summarize https://arxiv.org/pdf/2301.00001.pdf'")
    print("• Smart Commands (after searching):")
    print("  - 'summarize paper 1' - Summarize specific paper")
    print("  - 'summarize papers 1,2,3' - Summarize multiple papers")
    print("  - 'summarize all papers' - Summarize all found papers")
    print("• Type 'quit' to exit")
    print("=" * 60)
    
    agent = AgentHost()
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not user_input:
                continue
            
            print("🤖 Assistant: ", end="", flush=True)
            
            async for chunk in agent.process_message(user_input):
                print(chunk, end="", flush=True)
            
            print()  # New line after response
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Main loop error: {e}")


if __name__ == "__main__":
    asyncio.run(main())