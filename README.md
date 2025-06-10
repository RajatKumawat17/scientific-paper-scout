# Scientific Paper Scout - Enhanced Edition

An AI agent that helps discover and summarize recent research papers from arXiv with advanced batch processing and smart command capabilities.

## 🆕 New Features

- **Smart Commands**: Reference papers by number after searching (e.g., "summarize paper 1,2,3")
- **Batch Processing**: Summarize multiple papers in one command with meta-analysis
- **Auto-Save**: All summaries automatically saved to organized text files
- **Enhanced Error Handling**: Robust streaming with comprehensive error recovery

## Features

- **Model-agnostic**: Supports OpenAI, Anthropic, Gemini, and Groq LLMs
- **Paper Search**: Query arXiv for recent research papers
- **PDF Summarization**: Download and summarize research papers
- **Multiple PDF Processing**: Summarize multiple papers with comprehensive analysis
- **Smart CLI Commands**: Natural language commands for efficient workflow
- **Auto-Save Summaries**: All summaries saved to timestamped files
- **Real-time Streaming**: See responses as they're generated
- **Tool Call Logging**: Complete visibility into agent operations

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and set your configuration:
```bash
# Choose your LLM provider
LLM_PROVIDER=openai  # or anthropic, gemini, groq

# Set the appropriate API key
OPENAI_API_KEY=your_key_here
# OR
ANTHROPIC_API_KEY=your_key_here  
# OR
GEMINI_API_KEY=your_key_here
# OR
GROQ_API_KEY=your_key_here
```

### 3. Run the Application

```bash
python main.py
```

## Usage Examples

### Search for Papers
```
💬 You: search for quantum computing papers
```

After searching, you'll see numbered results and available smart commands.

### Smart Commands (After Searching)
```
💬 You: summarize paper 1
💬 You: summarize papers 1,2,3
💬 You: summarize papers 1-5
💬 You: summarize all papers
```

### Traditional URL Summarization
```
💬 You: summarize https://arxiv.org/pdf/2301.00001.pdf
```

### General Questions
```
💬 You: what are the latest trends in machine learning?
```

## Smart Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `summarize paper N` | Summarize a specific paper | `summarize paper 1` |
| `summarize papers N,M,P` | Summarize multiple papers | `summarize papers 1,3,5` |
| `summarize papers N-M` | Summarize range of papers | `summarize papers 1-3` |
| `summarize all papers` | Summarize all found papers | `summarize all papers` |

## Enhanced Workflow

1. **🔍 Search** for papers on your research topic
2. **📋 Review** the numbered list of results with paper details
3. **🤖 Summarize** using smart commands (no URL copying needed!)
4. **💾 Access** automatically saved summaries in `summaries/` folder

## File Organization

All summaries are automatically saved with organized structure:

```
summaries/
├── summary_20250611_143022.txt           # Single paper summary
├── multiple_summaries_20250611_143105.txt # Multiple papers analysis
└── multiple_summaries_20250611_150230.txt # Another batch analysis
```

### Multiple Paper Summary Structure
- **Comprehensive Analysis**: Meta-analysis identifying themes and insights
- **Individual Summaries**: Detailed summary of each paper
- **Processing Statistics**: Success/failure counts and failed URLs
- **Metadata**: Timestamps, URLs, and paper information

## Supported LLM Providers

| Provider | Models | Environment Variables |
|----------|--------|----------------------|
| OpenAI | gpt-3.5-turbo, gpt-4, etc. | `OPENAI_API_KEY` |
| Anthropic | claude-3-sonnet-20240229, etc. | `ANTHROPIC_API_KEY` |
| Google | gemini-pro, etc. | `GEMINI_API_KEY` |
| Groq | llama3-8b-8192, mixtral-8x7b-32768, etc. | `GROQ_API_KEY` |

## Architecture

The application consists of four main components:

1. **Agent Host**: Model-agnostic LLM client with tool orchestration and smart command parsing
2. **Paper Search Server**: Queries arXiv API for research papers
3. **PDF Summarize Server**: Downloads and summarizes PDFs using LLM (single and batch)
4. **Summary Manager**: Handles organized saving and file management

All tool calls are logged with timestamps, arguments, and latency metrics.

## Advanced Features

### Batch Processing
- Process multiple papers simultaneously
- Generate comprehensive meta-analysis across papers
- Identify common themes, contrasting findings, and research trends
- Robust error handling with partial success reporting

### Smart Command Parsing
- Natural language command interpretation
- Paper indexing from search results
- Range and list support for paper selection
- Automatic validation and error correction

### Enhanced Error Handling
- Comprehensive null safety checks
- Graceful degradation for partial failures
- Detailed error logging and user feedback
- Stream processing stability improvements

## Logs and Output

### Console Output
- Real-time streaming responses
- Tool call status and timing
- Progress indicators for batch processing
- Smart command hints and availability

### Log Files
- `paper_scout.log` - Comprehensive application logs
- `summaries/` - Organized summary files with timestamps

### Summary File Contents
- **Single Papers**: Summary, metadata, and source URL
- **Multiple Papers**: Meta-analysis, individual summaries, statistics
- **Timestamps**: Generation time and processing details
- **Error Reports**: Failed URLs and processing issues

## Error Resolution

The enhanced version includes fixes for common issues:

- **Fixed**: `NoneType` unpacking errors in streaming responses
- **Fixed**: XML parsing failures with missing elements
- **Enhanced**: Stream processing stability across all LLM providers
- **Improved**: Error messages and user guidance

## Example Session

```bash
🔬 Scientific Paper Scout - Enhanced Edition
============================================================
💬 You: search for transformer neural networks

🤖 Assistant: Found 5 recent papers:

**1. Attention Is All You Need: Revisiting Transformer Architecture**
Authors: Vaswani, Shazeer, Parmar
Published: 2024-01-15
Summary: This paper introduces improvements to the original transformer...
PDF: https://arxiv.org/pdf/2401.12345.pdf

[... more papers ...]

💡 Smart Commands Available:
• summarize paper 1 - Summarize a specific paper
• summarize papers 1,2,3 - Summarize multiple papers  
• summarize all papers - Summarize all found papers

💬 You: summarize papers 1,3,5

🤖 Assistant: 📄 Summarizing 3 papers:
• Paper 1: Attention Is All You Need: Revisiting Transformer Architecture
• Paper 3: Efficient Transformers: A Survey
• Paper 5: BERT: Pre-training of Deep Bidirectional Transformers

📊 COMPREHENSIVE ANALYSIS
==================================================
The three papers collectively demonstrate the evolution of transformer 
architectures from foundational concepts to practical optimizations...

💾 Summary saved to: summaries/multiple_summaries_20250611_143022.txt
```

This enhanced version transforms the paper research workflow from manual URL handling to an intuitive, numbered reference system while providing comprehensive analysis capabilities for literature reviews and research projects.