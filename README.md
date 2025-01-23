# A bunch of experiments with LLMs

## Build RAG application with local LLMs

### Install and run Ollama:

```bash
# My GPU is "NVIDIA GeForce RTX 4070 Laptop GPU with 8G VRAM", so I downloaded 7B version models.

~$ mkdir ~/workspace/ai/

~/workspace/ai$ curl -fsSL https://ollama.com/install.sh | sh

~/workspace/ai$ ollama pull mistral
~/workspace/ai$ ollama pull qwen2.5
~/workspace/ai$ ollama pull nomic-embed-text

~/workspace/ai$ mkdir bin

# make sure you have all the scripts placed in ~/workspace/ai/bin

~/workspace/ai$ ./bin/run_ollama
```

### Run AgenticRAG to analyze and search for JinPingMei

```bash
~/workspace/ai$ python3 -m venv test_env2
~/workspace/ai$ source test_env2/bin/activate
~/workspace/ai$ pip install -r llm/rag/requirements.txt

~/workspace/ai$ python bin/agentic_rag.py

# NOTE: to be able to handle org file, you have to run "pip install pypandoc-binary"
```

### Run GraphRAG to analyze and search for JinPingMei

```bash
~/workspace/ai$ python3 -m venv test_env
~/workspace/ai$ source test_env/bin/activate
~/workspace/ai$ pip install -r llm/graphrag/requirements.txt

~/workspace/ai$ mkdir ~/workspace/ai/ragtest
~/workspace/ai$ cp -rf llm/graphrag/ragtest ~/workspace/ai/ragtest

# apply changes in modified_graphrag to installed graphrag for mistral/qwen2.5 accordingly
# NOTE: finetuned prompt has been provided, if you wanna do it yourself, run ./bin/prompt_tuning

~/workspace/ai$ ./bin/graphrag_index

~/workspace/ai$ ./bin/local_query "这个章节中，西门庆有几个老婆，他们的关系如何?"

# NOTE: global_query is not working due to graphrag code broken
```

### Results

```bash
# The model used for these images is qwen2.5, as it's good at Chinese.

# NOTE: The result shows AgenticRAG beats GraphRAG most of the time, strange...
```

![西门庆和潘金莲什么关系?](images/graphrag_sample1.png)
![这个章节中，西门庆有几个老婆，他们的关系如何?](images/graphrag_sample2.png)
![这本书主要讲的是什么故事?](images/graphrag_sample3.png)
![langchain_rag_questions](images/langchain_rag_sample1.png)
