# Blog Writing Agent

An intelligent, agentic workflow for planning, researching, and writing technical blog posts. Built with [LangGraph](https://langchain-ai.github.io/langgraph/), [LangChain](https://python.langchain.com/), and [Ollama](https://ollama.com/).

This agent is designed to produce high-quality, structured technical content by mimicking a professional writer's workflow: routing, researching, planning, drafting, and refining.

## 🚀 Features

- **Intelligent Routing**: Determines if a topic requires external research (Open Book) or can be written from internal knowledge (Closed Book).
- **Automated Research**: Uses [Tavily](https://tavily.com/) to gather up-to-date information, facts, and citations for "Hybrid" or "Open Book" topics.
- **Structured Planning**: The "Orchestrator" node creates a comprehensive outline with specific goals, word counts, and key points for each section.
- **Parallel Drafting**: "Worker" nodes write each section of the blog post in parallel, ensuring focus and efficiency.
- **Image Generation**: Automatically decides where images are needed, suggests prompts, and generates them using Ollama (e.g., diagrams, visualizations).
- **Content Assembly**: Merges all sections into a polished Markdown file, embedding generated images.

## 🛠️ Prerequisites

- **Python 3.12+**
- **Ollama**: Must be installed and running locally.
  - Recommended model for text: `qwen2.5:7b-instruct` (or similar capable instruction-tuned model)
  - Recommended model for images: `x/z-image-turbo:latest` (or your preferred image model)
- **Tavily API Key**: For web research capabilities.

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/blog-writing-agent.git
    cd blog-writing-agent
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables**:
    Create a `.env` file in the root directory and add your Tavily API key:
    ```env
    TAVILY_API_KEY=tvly-xxxxxxxxxxxx
    ```

## 🏃 Usage

The core logic is implemented in the Jupyter Notebook `basic_blog.ipynb`. 

1.  **Start Ollama**:
    Ensure your Ollama server is running:
    ```bash
    ollama serve
    ```
    And pull the necessary models:
    ```bash
    ollama pull qwen2.5:7b-instruct
    # If using image generation:
    ollama pull x/z-image-turbo:latest
    ```

2.  **Run the Agent**:
    Open `basic_blog.ipynb` in Jupyter Lab or VS Code.
    Run the cells to define the graph and invoke the agent with your desired topic:

    ```python
    output = app.invoke({
        "topic": "The Future of Quantum Computing",
        "mode": "hybrid" # or let the router decide
    })
    ```

3.  **Output**:
    The agent will generate a Markdown file (e.g., `The_Future_of_Quantum_Computing.md`) in the project directory, complete with citations and images (if applicable).

## � Local LLMs & Configuration

This agent is optimized for local execution using **Ollama**, ensuring privacy and cost-efficiency.

### Text Generation Models
- **Primary Model**: `qwen2.5:7b-instruct`
- **Why**: Selected for its strong instruction-following capabilities and balance between speed and quality. It handles structured outputs (JSON/Pydantic models) effectively, which is crucial for the Router and Orchestrator nodes.
- **Context Window**: Configured with `num_ctx=8192` to handle large research contexts and long blog posts.
- **Temperature**: Set to `0.2` for factual, structured consistency in planning and `0.7` (default) could be used for creativity in drafting if modified in `basic_blog.ipynb`.

### Image Generation Models
- **Model**: `x/z-image-turbo:latest`
- **Why**: A fast SDXL-turbo based model that generates images in seconds, perfect for iterative drafting.
- **Integration**: The agent prompts this model via Ollama's API to generate visualizations based on the blog content.

### Customization
You can change these models in `basic_blog.ipynb`:
```python
llm = ChatOllama(
    model="your-preferred-model", # e.g., llama3, mistral
    temperature=0.2,
    num_ctx=8192
)
```

## 🧩 Architecture Deep Dive

The workflow is orchestrator-workers style, implemented as a [LangGraph](https://langchain-ai.github.io/langgraph/) StateGraph.

### Data State
The agent maintains a `State` dictionary containing:
- `topic`: Input topic.
- `plan`: The generate outline (Goals, Tasks, Word counts).
- `evidence`: Gathered research material.
- `sections`: List of written sections.
- `merged_md`: The final assembled markdown.

### Node Breakdown

1.  **Router (`router_node`)**:
    - **Input**: User topic.
    - **Logic**: Uses LLM structured output to categorize topic into `closed_book` (internal knowledge), `open_book` (needs research), or `hybrid`.
    - **Output**: Search queries if research is needed.

2.  **Research (`research_node`)**:
    - **Trigger**: Only if `needs_research` is True.
    - **Tools**: Queries **Tavily API** with the generated queries.
    - **Logic**: Deduplicates results and summarizes them into `EvidenceItem` objects.

3.  **Orchestrator (`orchestrator_node`)**:
    - **Role**: Editor-in-Chief.
    - **Logic**: Consumes topic + evidence. Generates a structured `Plan` containing multiple `Task` objects.
    - **Schema**: Each `Task` has a title, word count target, and specific sub-points (bullets) to cover.

4.  **Fanout (`fanout`)**:
    - **Logic**: Parallelizes execution. Converts the list of `Task`s into parallel `Worker` node calls.

5.  **Worker (`worker_node`)**:
    - **Role**: Section Writer.
    - **Logic**: Takes a single `Task` + `Plan` context + `Evidence`. Writes ~500 words of specific Markdown content.
    - **Constraint**: Strictly follows the assigned goal and bullets.

6.  **Reducer & Images (`reducer_subgraph`)**:
    - **Merge**: Concatenates all section markdown.
    - **Image Decision**: LLM scans the text to decide *where* diagrams/images would add value.
    - **Generation**: Calls local `x/z-image-turbo` to create images and inserts them into the final Markdown.

## 📄 License

This project is licensed under the MIT License.
