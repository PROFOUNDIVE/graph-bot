# graph-bot

This project implements a **Graph-augmented Buffer of Thoughts**. It helps Large Language Models (LLMs) think better by using a graph structure to store and reuse "thoughts."

Currently, many parts of this code are **placeholders (stubs)**. This means the structure is ready, but the real logic will be added later.


## 1. Repository Structure
The project is organized into several top-level folders:

* **`src/`**: The main source code.
* **`configs/`**: Files that set how the program runs (settings and templates).
* **`libs/`**: External libraries or submodules used by the project.
* **`tests/`**: Automated tests to check if the code works.
* **`pyproject.toml`**: Lists the project dependencies and tools.
* **`.env.example`**: A template for your private keys and API settings.


## 2. Technical Architecture (`src/graph_bot`)

The core logic lives inside `src/graph_bot`. Below are the most important files:

### **Data Models (`types.py`)**
This file defines how data looks using Pydantic models.
* **`SeedData`**: The starting information used to create thoughts.
* **`ReasoningNode` / `ReasoningEdge`**: The basic parts of a thought graph (nodes and connections).
* **`ReasoningTree`**: A collection of nodes and edges that represent a complete thought process.
* **`UserQuery`**: The question asked by the user.

### **Configuration (`settings.py`)**
This file manages settings using environment variables (starting with `GRAPH_BOT_`).
* **LLM Settings**: Choose the provider (like OpenAI) and the model name.
* **Graph Settings**: Set the maximum depth of the thought tree and how many paths to search.

### **Adapters (`adapters/`)**
Adapters connect the code to external tools.
* **`hiaricl_adapter.py`**: A placeholder that generates "thoughts" based on seed data.
* **`graphrag.py`**: A placeholder for storing and finding thoughts in a database (like Neo4j or SQLite).

### **Pipelines (`pipelines/`)**
Pipelines organize the steps of the process.
* **`build_trees.py`**: Turns seeds into reasoning trees.
* **`retrieve.py`**: Finds the best paths in the graph to answer a question.
* **`main_loop.py`**: Connects everything—building trees, storing them, and answering queries.

## 3. Command Line Interface (CLI)

You can run the project using the `cli.py` tool. It uses a library called **Typer**.

### **Server Management**
* **`graph-bot llm-server start`**: Starts a vLLM server to host the AI model.
* **`graph-bot llm-server stop`**: Stops the server.

### **Graph Operations**
* **`seeds-build`**: Reads a file and creates new thought trees.
* **`trees-insert`**: Saves trees into the graph storage.
* **`retrieve`**: Takes a user question and finds the answer using the graph.
* **`loop-once`**: Runs a full cycle (find answer -> create new thought -> save it).

## 4. How Data Flows

1.  **Ingestion**: `SeedData` is sent to the **Tree Builder**.
2.  **Generation**: The **HiAR-ICL Adapter** creates a `ReasoningTree` (placeholder thoughts).
3.  **Storage**: The trees are saved in the **GraphRAG Adapter**.
4.  **Query**: A user asks a question. The **Retrieval Pipeline** finds the best paths in the tree.
5.  **Answer**: The system combines the paths and the question to create an **LLM Answer**.


## 5. Suggested Reading Order for Developers

If you are new to this code, look at the files in this order:
1.  **`README.md`**: For the big picture and setup steps.
2.  **`pyproject.toml`**: To see what tools the project uses.
3.  **`src/graph_bot/types.py`**: To understand how thoughts are structured.
4.  **`src/graph_bot/cli.py`**: To see how to run the program.
5.  **`src/graph_bot/pipelines/`**: To see how thoughts move through the system.