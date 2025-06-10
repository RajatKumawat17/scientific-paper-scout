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
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            elif chunk.choices[0].delta.tool_calls:
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
            if chunk.type == "content_block_delta" and chunk.delta.text:
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
            if chunk.text:
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
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            elif chunk.choices[0].delta.tool_calls:
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
                    title = entry.find('atom:title', namespace)
                    summary = entry.find('atom:summary', namespace)
                    authors = entry.findall('atom:author', namespace)
                    published = entry.find('atom:published', namespace)
                    pdf_link = None
                    
                    # Find PDF link
                    for link in entry.findall('atom:link', namespace):
                        if link.get('title') == 'pdf':
                            pdf_link = link.get('href')
                            break
                    
                    author_names = [author.find('atom:name', namespace).text 
                                  for author in authors if author.find('atom:name', namespace) is not None]
                    
                    papers.append({
                        'title': title.text.strip() if title is not None else "Unknown Title",
                        'summary': summary.text.strip() if summary is not None else "No summary available",
                        'authors': author_names,
                        'published': published.text if published is not None else "Unknown date",
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


class AgentHost:
    """Main agent orchestrator"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.paper_search = PaperSearchServer()
        self.pdf_summarize = PDFSummarizeServer(self.llm_client)
        self.tool_calls: List[ToolCall] = []
        
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
                else:
                    yield f"Found {len(papers)} recent papers:\n\n"
                    for i, paper in enumerate(papers, 1):
                        yield f"**{i}. {paper['title']}**\n"
                        yield f"Authors: {', '.join(paper['authors'][:3])}\n"
                        yield f"Published: {paper['published'][:10]}\n"
                        yield f"Summary: {paper['summary'][:200]}...\n"
                        if paper['pdf_url']:
                            yield f"PDF: {paper['pdf_url']}\n"
                        yield "\n---\n\n"
        
        elif 'summarize' in user_lower and 'http' in user_message:
            # Extract PDF URL
            words = user_message.split()
            pdf_url = next((word for word in words if word.startswith('http')), None)
            
            if pdf_url:
                result = await self.call_tool("summarize_pdf", {"pdf_url": pdf_url})
                
                if 'error' in result:
                    yield f"Sorry, I couldn't summarize the PDF: {result['error']}"
                else:
                    yield "📄 **Paper Summary:**\n\n"
                    yield result['summary']
            else:
                yield "Please provide a valid PDF URL to summarize."
        
        else:
            # General conversation
            async for chunk in self.llm_client.chat_completion(messages, self.tools):
                yield chunk


async def main():
    """Main CLI interface"""
    print("🔬 Scientific Paper Scout")
    print("=" * 50)
    print("Commands:")
    print("• Search: 'search for quantum computing papers'")
    print("• Summarize: 'summarize https://arxiv.org/pdf/2301.00001.pdf'")
    print("• Type 'quit' to exit")
    print("=" * 50)
    
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