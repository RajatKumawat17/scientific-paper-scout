# Scientific Paper Scout

An AI agent that helps discover and summarize recent research papers from arXiv.

## Features

- **Model-agnostic**: Supports OpenAI, Anthropic, Gemini, and Groq LLMs
- **Paper Search**: Query arXiv for recent research papers
- **PDF Summarization**: Download and summarize research papers
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

### Summarize a Paper
```
💬 You: summarize https://arxiv.org/pdf/2301.00001.pdf
```

### General Questions
```
💬 You: what are the latest trends in machine learning?
```

## Supported LLM Providers

| Provider | Models | Environment Variables |
|----------|--------|----------------------|
| OpenAI | gpt-3.5-turbo, gpt-4, etc. | `OPENAI_API_KEY` |
| Anthropic | claude-3-sonnet-20240229, etc. | `ANTHROPIC_API_KEY` |
| Google | gemini-pro, etc. | `GEMINI_API_KEY` |
| Groq | llama3-8b-8192, mixtral-8x7b-32768, etc. | `GROQ_API_KEY` |

## Architecture

The application consists of three main components:

1. **Agent Host**: Model-agnostic LLM client with tool orchestration
2. **Paper Search Server**: Queries arXiv API for research papers
3. **PDF Summarize Server**: Downloads and summarizes PDFs using LLM

All tool calls are logged with timestamps, arguments, and latency metrics.

## Logs

Tool calls and errors are logged to:
- Console output (real-time)
- `paper_scout.log` file