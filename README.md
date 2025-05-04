# kaggle-genai-course
This repository is created for notebooks and material used in the the Kaggle 5-day intensive genai course

## Day-1

### Whitepaper Summary

*   **Large Language Models (LLMs) represent a significant advancement in AI**, capable of processing, understanding, and generating human-like text through deep neural networks trained on massive datasets. These models are transforming interactions with information and technology.
*   LLMs have demonstrated **impressive performance across various NLP tasks**, including language translation, code generation, text generation, classification, and question-answering, often surpassing previous state-of-the-art models. They can also exhibit emergent behaviors and be adapted for specific tasks through fine-tuning and guided by prompt engineering.
*   The **transformer architecture**, developed in 2017, is the core building block of most modern LLMs, utilizing self-attention mechanisms to process sequences in parallel and model long-term contexts more effectively than RNNs. The original transformer consists of an encoder and a decoder.
*   **Training transformers involves data preparation** (cleaning, tokenization, embedding, positional encoding) and using a **loss function** to update the model's parameters based on the difference between predicted and target sequences. The **context length**, or the number of previous tokens the model can remember, is an important factor during training.
*   The whitepaper outlines the **evolution of transformer-based LLMs**, including encoder-only (like BERT), decoder-only (like GPT family, LaMDA, Gopher, Gemma, LLaMA, Mixtral), and multimodal (like Gemini) architectures, highlighting their key innovations, datasets, and parameter sizes. The scaling of data and parameters has led to improved performance and emergent abilities.
*   **Fine-tuning** is crucial for specializing pre-trained LLMs for specific tasks or improving desired behaviors like instruction following, dialogue capabilities, and safety. Techniques include Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF). **Parameter Efficient Fine-Tuning (PEFT)** methods like adapters and LoRA significantly reduce the computational cost of fine-tuning.
*   **Prompt engineering**, the art of designing and refining text inputs, and **sampling techniques** (like greedy search, random sampling, temperature sampling, top-K, and top-P) significantly influence the output quality, creativity, and diversity of LLMs.
*   **Accelerating inference** is vital for reducing latency and cost when using LLMs in applications. Techniques are categorized as output-approximating (like quantization and distillation, which may have minor quality trade-offs) and output-preserving (like Flash Attention and prefix caching). **Speculative decoding** is another quality-neutral method to speed up the decoding process.
*   LLMs have a wide range of **applications** across various domains, including code and mathematics, machine translation, text summarization, question-answering, chatbots, content generation, natural language inference, text classification, text analysis, and multimodal applications.
*   Key takeaways emphasize the foundational importance of the transformer architecture, the significance of data and fine-tuning strategies, the ongoing research in efficient inference, and the diverse and expanding applications of LLMs, highlighting the need for effective prompt engineering and parameter tuning for specific tasks.

[NotebookLM Link](https://notebooklm.google.com/notebook/95087705-ecfd-4385-816f-1f23105c6201?_gl=1*fyxxyk*_ga*MjExMzc2Mjc0NS4xNzQzNTA0MDg5*_ga_W0LDH41ZCB*MTc0MzU2MDQ2Ni4yLjAuMTc0MzU2MDQ2Ny41OS4wLjA.) (please request access)

### Notebook Summary

#### Notebook 1 - Prompting
1. LLM output can be obtained using Google's new google-genai python SDK.
2. models.generate_content is an API which is used to obtain the LLM response. We can pass parameters like prompt, model name (e.g. - gemini-2.0-flash), configuration of response etc.).
3. We can start a chat using chats.create API, this will create a multi-turn structure of the interation with LLM. We can also input a chat history when we initiate the chat to provide a context to the chat.
4. LLM response can be configured multiple ways e.g. -
  * max_output_tokens: limit output length
  * temperature: control randomness in response
  * top-p: control diversity in response
5. Gemini API has an Enum mode feature that allows you to constrain the output to a fixed set of values
6. config parameter in the models.generate_content API can also be used to specify an output format (e.g.- json). This can be obtained by passing response_schema parameter in the config file
7. Chain of thought response can be obtained by adding prompts like - "think step by step"
8. We can run a ReAct prompt directly in the Gemini API and perform the searching steps yourself. As this prompt follows a well-defined structure, there are frameworks available that wrap the prompt into easier-to-use APIs that make tool calls automatically, such as the LangChain.
9. To capture a single step at a time, while ignoring any hallucinated Observation steps, you will use stop_sequences to end the generation process. The steps are Thought, Action, Observation, in that order.
10. models.generate_content_stream API can be used to initiate "thinking mode" in Gemini 2.0 flash model, "Thinking" model has been trained to generate the "thinking process" the model goes through as part of its response, using a "thinking mode" model can provide you with high-quality responses without needing specialised prompting like the previous approaches.
11. Gemini also has features to help with coding such as - generating, executing, and explaning code.



#### Notebook 2 - Evaluation and Structures Response
1. To evaluate an LLM response, we can evaluate a number of aspects, like how well the model followed the prompt ("instruction following"), whether it included relevant data in the prompt ("groundedness"), how easy the text is to read ("fluency"), or other factors like "verbosity" or "quality".
2. In this notebook, we learn different types of LLM evaluation techniques
3. In this notebook we evaluate the summarization task of an LLM, we can perform a verbose evaluation or a structure rating based evaluation.
4. We also look at performing pointwise evaluation for QA task
5. We generated answers for questions wrt the document (context) with different kinds of instructions - terse, moderated, and cited and evaluation. THe response can be evaluated using verbose evaluation or structured evaluation
6. We also look at technique to perform pairwise evaluation to perform comparison of two responses and evaluate which is a better response using an LLM (which uses the evaluation criteria). This technique has various uses in ranking and sorting algorithms.
7. We develop a logic to compare different prompts against each other and rank them
8. Challenges in using LLM as a prompt - LLM struggle in certain tasks such as counting numbers of characters in text so an LLM evaluator will also struggle in that kind of task
9. Improving confidence - in order to improve the evaluator we can pit the LLMs against each other and take the best one


## Day-2

### Whitepaper Summary

1.  The whitepaper explores the power of **embeddings**, which transform diverse data like images, text, and audio into a unified vector representation for use in various applications. Embeddings are low-dimensional numerical representations where the geometric distance between vectors in the vector space reflects the relationship and semantic similarity between the real-world objects they represent.
2.  Embeddings are essential for handling **multimodal data** and providing **compact representations** of data while allowing comparison of data objects based on numerical similarity. They also act as a form of lossy compression while retaining important semantic properties.
3.  Key applications for embeddings include **retrieval and recommendations**, especially from massive search spaces like the entire internet for Google Search. These systems rely on precomputing embeddings, mapping query embeddings into the same space, and efficiently retrieving nearest neighbors. Embeddings also shine in the world of multimodality by mapping multiple types of objects into a joint embedding space.
4.  **Evaluating embedding quality** depends on the specific task, but common metrics focus on the ability to retrieve similar items while excluding dissimilar ones. For search use cases, important metrics are **precision** (relevant documents retrieved out of total retrieved) and **recall** (relevant documents retrieved out of total relevant in corpus). When document relevancy ordering is known, metrics like **Normalized Discounted Cumulative Gain (nDCG)** are used to measure ranking quality. Public benchmarks like BEIR and MTEB are widely used for evaluation.
5.  A common search application described is **Retrieval Augmented Generation (RAG)**, where embeddings are used to identify relevant documents before inserting them into an LLM's prompt for summarization. This process involves an index creation phase where document chunks are embedded and stored in a vector database, and a query phase where the user's query is embedded to search the database for relevant documents. **Dual encoder architectures** are often used, with separate networks for queries and documents, particularly when questions and answers are phrased differently.
6.  The quality of embedding models has been improving rapidly, with current models showing significantly higher performance on benchmarks like BEIR compared to earlier models like BERT. **Model upgrades** are frequent, necessitating good evaluation suites and potentially using platforms with established upgrade paths.
7.  There are various **types of embeddings** depending on the data modality. **Text embeddings** for NLP can be categorized into word embeddings and document embeddings. The process often starts with **tokenization**, where text is split into meaningful pieces (tokens), assigned unique integer IDs, and potentially one-hot encoded.
8.  **Word embeddings** represent the meaning of individual words, often based on their neighbors in the text. Examples include **Word2Vec** (using CBOW or skip-gram methods) which accounts for local word statistics, and **GloVe** and **SWIVEL** which leverage both global and local statistics using co-occurrence matrices.
9.  **Document embeddings** represent the meaning of paragraphs and documents. Early methods include **shallow Bag-of-Words (BoW) models** like LSA, LDA, and TF-IDF based models (e.g., BM25) which treat documents as unordered word collections, ignoring word ordering and full semantic meaning. **Doc2Vec** improved upon Word2Vec by adding a 'paragraph' or document embedding to predict words.
10. More recently, **deeper pretrained large language models** like BERT, T5, PaLM, Gemini, GPT, and Llama have led to significant advancements in document embeddings. These models use complex neural networks (often transformers), massive pre-training on unlabeled text, subword tokenization, and fine-tuning for downstream tasks. They output contextualized embeddings, meaning the embedding for a word can differ based on its context. New models based on LLMs, such as those from Vertex AI, offer superior performance. **Multi-vector embeddings** (e.g., ColBERT, XTR, ColPali) can enhance representational power.
11. **Image and multimodal embeddings** are also crucial. Unimodal image embeddings can be derived from training CNNs or Vision Transformers. **Multimodal embeddings** represent the joint semantic relationships between different modalities like text and images in the same latent space. Approaches like ColPali enable retrieval from text queries on multimodal documents.
12. **Structured data embeddings** can be created for tabular data. This might involve dimensionality reduction techniques like PCA. Use cases include **anomaly detection** and providing features for downstream ML tasks. For user/item data (common in recommendations), embeddings can correlate to entities like users and products, often mapping them into the same space.
13. **Graph embeddings** represent not only object information but also relationships with neighbors in a graph structure, like social networks. This can enrich embeddings and be used for tasks like node classification, link prediction, clustering, and recommendation systems. Popular algorithms include DeepWalk, Node2vec, LINE, and GraphSAGE.
14. Current embedding models often use a **dual encoder (two tower) architecture**, with one tower for queries/inputs and another for documents/items. Training commonly employs a variation of **contrastive loss** to bring positive examples closer and push negative examples apart. Training typically involves initializing from foundation models and then **fine-tuning** on specific datasets, which can be human-labeled, synthetically generated, or created via other methods. Embeddings can also be used as input features for downstream models, either frozen or fine-tuned.
15. **Vector search** goes beyond traditional keyword matching by using the vector or embedded semantic representation of data. It allows searching for meaning across various data modalities like text, images, and videos, even when the wording differs. This involves embedding the query in the same vector space as the items and finding the most similar matches using distance metrics like **Euclidean distance, cosine similarity, or dot product**.
16. For large datasets, **Approximate Nearest Neighbor (ANN) search** techniques are more practical than linear search (O(N)). ANN finds closest points with a small error margin but significantly fewer computations (O(logN)). Techniques include **quantization, hashing, clustering, and trees**.
17. Important vector search algorithms include **Locality Sensitive Hashing (LSH)**, which maps similar items to the same hash bucket for faster lookups. **Tree-based algorithms** like KD-tree and Ball-tree partition the data space based on dimension-wise medians or radial distance, respectively.
18. **Hierarchical Navigable Small Worlds (HNSW)**, used in libraries like FAISS, builds a proximity graph with a hierarchical structure for efficient ANN search. It traverses layers greedily to find nearest neighbors in sub-linear time (O(Logn)).
19. **Scalable Approximate Nearest Neighbor (ScaNN)**, developed by Google, is used across Google products and available via Vertex AI Vector Search and Google Cloud Databases. ScaNN uses optional partitioning to group related vectors, selects top partitions at query time, and then performs a scoring step (exact or approximate distance). An optional final step can rescore top results for better accuracy.
20. **Vector databases** are specialized systems built to manage embeddings and support efficient vector search at scale. They combine the storage and management capabilities of traditional databases with vector search algorithms. The workflow typically involves embedding data, indexing vectors with metadata, embedding queries, and searching for similar items. Many traditional databases are incorporating vector search functionality for **hybrid search** (combining semantic and traditional keyword search).
21. Examples of vector databases include Google Cloud’s **Vertex AI Vector Search** (using ScaNN), Google Cloud’s **AlloyDB & Cloud SQL Postgres** (supporting pgvector and ScaNN extensions), **Pinecone**, and open source options like **Weaviate** and **ChromaDB**.
22. **Operational considerations** for vector databases include horizontal and vertical scalability, availability, data consistency, real-time updates, backups, access control, and compliance. Additionally, embeddings may need updating as models evolve or data drifts, which can be expensive. Combining vector search with **full-text search** can help address cases where embeddings struggle with specific domain-specific words or IDs. The choice of vector database also depends on workload type (e.g., OLTP vs. OLAP).
23. Embeddings and vector stores are powerful tools for various applications, including **Retrieval Augmented Generation (RAG)** for LLMs, **Search**, **Recommendation Systems**, **Anomaly detection**, and **few-shot classification**. For ranking problems like search and recommendations, embeddings are often used in an initial stage to retrieve a set of candidates that can then be refined by more sophisticated models.
24. **RAG for Q&A** retrieves relevant documents and uses prompt expansion to generate answers based on those documents. This helps mitigate LLM **hallucination** by grounding responses in data and keeps information current without costly retraining. Providing sources from the retrieval process further helps verify the LLM's output.

### Notebook Summary

#### Notebook 1 - Document RAG with Q&A

[Colab Notebook Link](https://colab.research.google.com/github/singhvis29/kaggle-genai-course/blob/main/Day_2_Document_Q%26A_with_RAG.ipynb)

1. RAG is used to mitigate the two drawbacks of LLM i.e.- 1) that they only "know" the information that they were trained on, and 2) that they have limited input context windows.
2. Indexing happens ahead of time, and allows you to quickly look up relevant information at query-time. When a query comes in, you retrieve relevant documents, combine them with your instructions and the user's query, and have the LLM generate a tailored answer in natural language using the supplied information.
3. In this notebook we use the Gemini API to create a vector database, retrieve answers to questions from the database and generate a final answer. We use Chroma, an open-source vector database.
4. We use Gemini client.models.embed_content API to embed the input documents. The embedding used are 'models/text-embedding-004'
5. We create a function - GeminiEmbeddingFunction to generate embeddings for documents. The function can be used to generate embeddings for documents or queries
6. We create a Chroma database client that uses the GeminiEmbeddingFunction and populate the database with the documents
7. To search the Chroma database, we call the query method. Note that we also switch to the retrieval_query mode of embedding generation. We retreive the relevant passage wrt our query from the db
8. We finally assemble a generation prompt to have the Gemini API generate a final answer.

#### Notebook 2 - Embeddings and similarity scores

[Colab Notebook Link](https://colab.research.google.com/github/singhvis29/kaggle-genai-course/blob/main/Day_2_Embeddings_and_similarity_scores.ipynb)

1. Gemini supports 4 models for embedding task -
 * models/embedding-001
 * models/text-embedding-004
 * models/gemini-embedding-exp-03-07
 * models/gemini-embedding-exp
2. We embed the list of texts using the models/text-embedding-004 model
3. A similarity score of two embedding vectors can be obtained by calculating their inner product. If $\mathbf{u}$ is the first embedding vector, and $\mathbf{v}$ the second, this is $\mathbf{u}^T \mathbf{v}$. As the API provides embedding vectors that are normalised to unit length, this is also the cosine similarity.

#### Notebook 3 - Classify embeddings with Keras

[Colab Notebook Link](https://colab.research.google.com/github/singhvis29/kaggle-genai-course/blob/main/Day_2_Classifying_embeddings_with_Keras.ipynb)


## Day-3

### Whitepaper Summary

#### Whitepaper 1 - Agents

1.  **Generative AI agents** are defined as applications that aim to achieve a goal by observing the world and acting upon it using available tools.
2.  Just like humans use tools (books, Google Search, calculators) to supplement prior knowledge, **Generative AI models can be trained to use tools** to access real-time information or suggest real-world actions.
3.  An agent invokes the concept of a program that **extends beyond the standalone capabilities of a Generative AI model** by combining reasoning, logic, and access to external information connected to the model.
4.  Agents are **autonomous and can act independently** of human intervention, especially when given proper goals or objectives. They can also be proactive, reasoning about the next steps even without explicit instructions.
5.  The inner workings of an agent are driven by foundational components that form a **cognitive architecture**, which can be configured in various ways.
6.  The three essential components in an agent’s cognitive architecture are **the model, the tools, and the orchestration layer**.
7.  **The model** is the language model (or multiple models) used as the **centralized decision maker** for agent processes. It should be capable of following instruction-based reasoning frameworks like ReAct, Chain-of-Thought, or Tree-of-Thoughts.
8.  **Tools bridge the gap between foundational models and the outside world**, empowering agents to interact with external data and services, thus unlocking a wider range of actions. Tools align with common web API methods and can fetch data or update information.
9.  **The orchestration layer** describes a **cyclical process** governing how an agent takes in information, performs internal reasoning, and uses that reasoning to inform its next action or decision. This loop continues until the agent reaches its goal.
10. **Models are limited to their training data**, while **agents extend knowledge through connection with external systems via tools**. Models typically perform single inferences, while agents manage session history for multi-turn interactions.
11. Models do not have native tool implementation or logic layers, while **agents have native cognitive architecture** that uses reasoning frameworks and natively implemented tools.
12. The orchestration layer is responsible for maintaining **memory, state, reasoning, and planning**. It uses prompt engineering frameworks to guide these processes.
13. Popular reasoning frameworks include **ReAct** (synergizes Reasoning and Acting), **Chain-of-Thought (CoT)** (enables reasoning through intermediate steps), and **Tree-of-Thoughts (ToT)** (generalizes CoT for exploration and strategic lookahead).
14. **Tools** are the link between foundational models and the outside world, allowing agents to perform tasks with greater accuracy and reliability, such as adjusting smart home settings or fetching user information.
15. As of the publication date, the three primary tool types Google models interact with are **Extensions, Functions, and Data Stores**.
16. **Extensions** act as a **bridge between an API and an agent**, standardizing API execution regardless of underlying implementation. They teach the agent how to use an API endpoint using examples and specifying required arguments. Extensions are executed on the **agent-side**.
17. **Functions** work similarly but the **model outputs a Function and its arguments without making a live API call**. Functions are executed on the **client-side**. This gives developers more granular control over data flow, especially for security, timing, or additional data transformation needs.
18. **Data Stores** provide agents with access to **dynamic and up-to-date information** beyond their static training data, ensuring responses are grounded in factuality.
19. Data Stores are typically implemented as a **vector database**. They allow developers to provide additional data in various formats (spreadsheets, PDFs, website content) without time-consuming transformations or retraining.
20. A prolific example of Data Store usage is **Retrieval Augmented Generation (RAG)**, where the agent accesses data from the vector database to supplement its knowledge.
21. **Model performance in tool selection** can be enhanced through targeted learning approaches like **in-context learning** (learning 'on the fly' with prompts and few-shot examples), **retrieval-based in-context learning** (dynamically populating prompts with relevant information from external memory/data stores), and **fine-tuning** (pre-training on larger datasets of specific examples).
22. Libraries like **LangChain and LangGraph** allow users to build custom agents by chaining together sequences of logic, reasoning, and tool calls.
23. Building production-grade agent applications often involves integrating core components with additional tools like UIs, evaluation frameworks, and continuous improvement mechanisms, simplified by platforms like **Google’s Vertex AI**.
24. The future of agents involves advancements in tools and reasoning, enabling the solution of increasingly complex problems, including the concept of **'agent chaining'** or 'mixture of agent experts'.

#### Whitepaper 2 - Agents Companion

Certainly, here is a summary of the provided paper excerpts in numbered points:

1.  **Generative AI agents** represent a significant advancement from traditional language models, designed to achieve specific objectives by perceiving their environment and acting strategically using tools. An agent's architecture comprises three essential elements: a **Model** (the language model acting as the central decision-making unit), **Tools** (facilitating interaction with the external world via extensions, functions, and data stores), and an **Orchestration layer** (a cyclical process governing information assimilation, reasoning, and action, responsible for memory, state, reasoning, and planning, often employing techniques like ReAct, Chain-of-Thought, and Tree-of-Thoughts).
2.  Operationalizing Generative AI agents requires **AgentOps**, a subcategory of GenAIOps that builds upon DevOps and MLOps best practices. Key additional components of AgentOps include internal and external tool management, agent brain prompt and orchestration, memory, and task decomposition. Metrics are critical for building, monitoring, and comparing agent revisions, including business metrics, goal completion rate, task/interaction success metrics, and application telemetry like latency and errors.
3.  **Agent evaluation** is crucial for transitioning agents from proof-of-concept to production. It goes beyond just evaluating the final output and requires understanding the agent's decision-making process. Evaluation is broken down into three components: **Assessing Agent Capabilities** (core abilities like understanding instructions and reasoning), **Evaluating Trajectory and Tool Use** (analyzing the steps and tools used), and **Evaluating the Final Response** (assessing quality, relevance, and correctness).
4.  **Evaluating Trajectory and Tool Use** involves comparing the actual sequence of agent actions against an expected path. Ground-truth-based automated evaluations include metrics like Exact match, In-order match, Any-order match, Precision (relevant tool calls), Recall (essential tool calls captured), and Single-tool use (checking for specific actions). This is particularly useful for debugging but requires a reference trajectory.
5.  **Evaluating the Final Response** centers on whether the agent achieves its goals. This can be automated using an **autorater**, which is an LLM that acts as a judge, assessing the response against user-provided criteria. Precision in defining evaluation criteria is crucial for this method.
6.  **Human-in-the-Loop Evaluation** is essential alongside automated methods, especially for tasks requiring subjective judgment, creativity, or common sense. Human feedback helps calibrate automated evaluations (like autoraters) and provides insights for iterative improvement. Methods include direct assessment by experts, comparative evaluation, and user studies.
7.  The field is moving towards **multi-agent architectures**, where multiple specialized agents collaborate to achieve complex objectives. Advantages over single-agent systems include Enhanced Accuracy (agents cross-check), Improved Efficiency (parallel work), Better Handling of Complex Tasks (decomposition), Increased Scalability, Improved Fault Tolerance, and Reduced Hallucinations and Bias. Agents can be categorized by function, such as Planner, Retriever, Execution, and Evaluator agents.
8.  **Multi-agent design patterns** define how agents interact and delegate tasks. Common patterns include Sequential (pass output in order), Hierarchical (manager agent coordinates workers), Collaborative (agents share info/resources), and Competitive (agents compete). The choice depends on the application and desired interaction level.
9.  **Evaluating Multi-Agent Systems** is a progression from single-agent evaluation, using the same tools like Agent Success Metrics (business KPIs, goals, telemetry), traces for debugging, and evaluating Trajectories and Final Responses. Unique questions arise, such as assessing Cooperation/Coordination, Planning/Task Assignment, Agent Utilization, and Scalability.
10. **Agentic Retrieval-Augmented Generation (RAG)** is an evolution that uses autonomous retrieval agents to actively refine searches based on iterative reasoning. This enhances retrieval by employing context-aware query expansion, multi-step reasoning, adaptive source selection, and validation/correction. It improves accuracy, explainability, and adaptability, especially valuable in complex domains. **Optimizing the underlying search engine** (parsing, chunking, metadata, embeddings, rankers) is foundational for improving RAG performance before adding agents.
11. **Agents in the enterprise** are expected to assist employees ("Assistants") or run autonomously in the background ("Automation agents"). Knowledge workers will increasingly manage fleets of agents using novel user interfaces. **Google Agentspace** is presented as a platform designed for enterprises to create, configure, manage, invoke, monitor, and orchestrate agents, providing secure, company-branded, multimodal search and AI assistance over enterprise data. **NotebookLM Enterprise** is a related tool for synthesizing complex information from sources. **Agentspace Enterprise Plus** allows for the creation of custom specialized agents.
12. The concept of evolving agents into **"contractors"** is proposed to enhance reliability and utility, particularly for complex, high-stakes tasks. Contracts standardize the definition of outcomes, making them precise and verifiable (Deliverables & Specifications). They enable negotiation, clarification of ambiguities, and the generation of subcontracts for task decomposition. The Contract Lifecycle involves definition, negotiation, and execution.
13. **Automotive AI** serves as a real-world case study for multi-agent architecture, illustrating how specialized agents (e.g., Navigation, Media Search, Car Manual) collaborate to provide a seamless in-car experience. Common patterns like Hierarchical, Diamond, Peer-to-Peer, Collaborative, and Adaptive Loop are employed. Advantages in this domain include specialization, efficiency, speed (balancing on-device and cloud processing), and resilience (offline capability).
14. **Google Vertex AI Agent Builder** is presented as a comprehensive platform for developers to build and connect agents, incorporating engineering excellence, AI research, and AgentOps best practices. It includes the Vertex AI Agent Engine (managed runtime), Vertex AI Eval Service (evaluation tools), and a portfolio of agent tools (Search, RAG Engine, database tools, application integrations).
15. Key takeaways emphasize the need for AgentOps, metrics-driven improvement, automated and human evaluation, leveraging multi-agent systems and Agentic RAG, optimizing search, using agent/tool registries, prioritizing security, considering build vs. buy decisions, and recognizing the transformative impact of agents in the enterprise. Future directions include advanced evaluation, multi-agent coordination, real-world adaptation, explainability, long-term memory, communication protocols, and the "agents to contractors" evolution. Developers are encouraged to explore and build using available tools and resources.

### Notebook Summary

#### Notebook 1 - Function Calling with Gemini API

[Colab Notebook Link](https://colab.research.google.com/github/singhvis29/kaggle-genai-course/blob/main/Day_3_Function_calling_with_the_Gemini_API.ipynb)

1. In this notebook, we start the example by creating a SQLite database and adding some synthetic data which we can query.
2. Gemini API can interact with external functions by two approaches, which in this case are used to access and query a database -
 * Defining an OpenAPI Schema: This involves creating a formal schema that describes the available functions and their parameters using the OpenAPI specification. This schema is then passed directly to the Gemini model.
 * Defining Python Functions: This approach involves writing regular Python functions and allowing the SDK to automatically generate the schema from them. This is the approach taken in this notebook.
3. When using Python functions for function calling, type annotations and accurate docstrings are crucial. This is because:
 * Type Annotations: Provide information about the expected data types of function parameters and return values.
 * Docstrings: Serve as detailed descriptions of what the function does, its parameters, and what it returns.
The Gemini model relies on these annotations and docstrings as its sole source of information about the functions. It cannot directly inspect the function's code to understand its behavior. Therefore, the docstrings effectively act as the interface between the model and your functions.
4. The code in the notebook defines three essential functions to interact with the database:
 * list_tables(): Retrieves the names of all tables in the database.
 * describe_table(table_name): Provides the schema (column names and types) of a given table.
 * execute_query(sql): Executes an SQL query and returns the results.
By providing these functions to the Gemini model (LLM), it gains the ability to:
 * Understand the Database Structure: Using list_tables and describe_table, the model can learn what tables and columns are available.
 * Query the Database: With execute_query, the model can run SQL queries to retrieve specific data.
5. The goal of this setup is to empower the Gemini model to act as a database interface for the user. Just like a human user would interact with a database (e.g., using SQL commands), the model is equipped with the necessary tools to understand and retrieve information from the database. This enables a more interactive and dynamic conversation where the model can:
 * Answer user questions that require database lookups.
 * Perform actions on the database based on user requests.
6. Function calls are implemented to obtain LLM response for a user query from the database. Function calling works by adding specific messages to a chat session. When function schemas are defined and made available to the model and a conversation is started, instead of returning a text response, the model may return a function_call instead. When this happens, the client must respond with a function_response, indicating the result of the call.
7. The function calling interaction normally happens manually, allowing you, the client, to validate and initiate the call. However the Python SDK also supports automatic function calling, where the supplied functions will be automatically invoked.
8. In this jupyter NB, we implement automatic function calling where we attach all the user defined function to LLM. This enables the client to automatically invoke the function and respond to user queries
9. Gemini 2.0 is quite powerful and can answer question that involce multipe steps - For e.g. - "What products should salesperson Alice focus on to round out her portfolio? Explain why."
10. Gemini 2.0 has the ability to compose user-provided function calls together while generating code. This means that the model is able to take the available tools, generate code that uses it, and execute it all.
11. We define an asynchronous function called `handle_response`. Its primary role is to manage the stream of responses coming from a large language model (LLM) like Gemini, especially when the model uses external tools or functions. It handles displaying the model's output and managing any function calls that the model might make.
12. We then set up a connection to the Gemini Live API, provide instructions and a tool (execute_query) to the model, start a chat session, and send a message requesting the model to generate and insert data into the database. The `handle_response` function is then used to process the model's response and handle any interactions with the provided tool.
13. Finally we are able to use the live connection to generate a code from data in sql database and plot and seaborn plot.


#### Notebook 2 - Building an Agent with LangGraph

[Colab Notebook Link](https://colab.research.google.com/github/singhvis29/kaggle-genai-course/blob/main/Day_3_Building_an_agent_with_LangGraph.ipynb)

1. This notebook teaches us to build an agent using LangGraph. LangGraph applications are built around a **graph** structure. As the developer, you define an application graph that models the state transitions for your application. Your app will define a **state** schema, and an instance of that schema is propagated through the graph.
2. 2.Each **node** in the graph represents an action or step that can be taken. Nodes will make changes to the state in some way through code that you define. These changes can be the result of invoking an LLM, by calling an API, or executing any logic that the node defines.
3. Each **edge** in the graph represents a transition between states, defining the flow of the program. Edge transitions can be fixed, for example if you define a text-only chatbot where output is always displayed to a user, you may always transition from `chatbot -> user`. The transitions can also be conditional, allowing you to add branching (like an `if-else` statement) or looping (like `for` or `while` loops).
4. State is a fundamental concept for a LangGraph app. A state object is passed between every node and transition in the app. Here you define a state object, OrderState, that holds the conversation history, a structured order, and a flag indicating if the customer has finished placing their order.
5. Each node in the graph operates on the state object. The state (a Python dictionary) is passed as a parameter into the node (a function) and the new state is returned. For the chatbot node, the state is updated by adding the new conversation message.
6. We introduce a human node in the graph. The human node is used to have back and forth conversation with the chatbot. This is done by adding a **conditional node** to the graph which checks the logic to end a conversation or continue back to chatbot.
7. We can add tools to the LangGraph, this is done by adding conditional nodes which will use tools to transition from one state to another. In a LangGraph app, you can annotate Python functions as tools by applying the `@tools` annotation. The tools are also bound to the llm object so that the underlying model knows they exist.
8. Instead of hard-coding a menu, we can create a more dynamic and could respond to fluctuating stock levels, we will put the menu into a custom tool. The tools are wrapped in `ToolNode`. that handles calling the tool and passing the response as a message through the graph. The tools are also bound to the llm object so that the underlying model knows they exist. As you now have a different llm object to invoke, you need to update the chatbot node so that it is aware of the tools.
9. Next we create an order processing node and add tools to handle the order. The final graph looks as follows - 

![image](https://github.com/user-attachments/assets/1796a855-0fda-480e-99df-860e241d044d)

10. This way we can define nodes and edges of the graph to build our app. The LLM responses are used to navigate throughout the nodes to perform functions of the app and terminate based on user defined conditions.


## Day-4: Domain-Specific LLMs

### Whitepaper Summary

1.  This whitepaper, explores the potential of LLMs for tackling complex challenges in specific domains, focusing on **cybersecurity and medicine**.
2.  While early LLMs focused on general-purpose tasks, recent developments highlight the potential of **fine-tuning LLMs to address specific problems within specialized fields**.
3.  In cybersecurity, LLMs face challenges such as a scarcity of publicly available data, a wide diversity of highly technical concepts, rapidly changing threat information, and sensitive use cases like malware analysis.
4.  Security practitioners face major challenges today: **new and evolving threats**, **operational toil** (repetitive tasks), and a **talent shortage**. Attackers are adopting advanced technologies, including AI, to extend their reach and quicken exploitation.
5.  Generative AI (GenAI) and LLMs can help address these challenges by automating repetitive tasks, freeing up time for strategic activities, providing new opportunities to access knowledge, and improving the working lives of security novices and experienced practitioners.
6.  GenAI can assist security analysts by **translating natural language queries** into domain-specific security event query languages, and provide autonomous capabilities for investigation, grouping, and classification of alerts. It can also help provide personalized remediation planning.
7.  For threat researchers and system administrators, GenAI can perform automated **reverse engineering** and code analysis to explain, analyze, and classify potentially malicious artifacts. For developers and IT administrators, it can help identify attack paths and align access policies.
8.  To tackle cybersecurity problems holistically, a multi-layered approach is needed, including existing security tools, a **security-specialized model API** with reasoning and planning capabilities, and datastores of security intelligence.
9.  **SecLM** is envisioned as a ‘one-stop shop’ API for security questions, regardless of complexity, allowing users to pose questions in natural language and get answers that automatically incorporate necessary information.
10. Key requirements for the SecLM API include **freshness** (accessing the latest threat data daily, which is infeasible with retraining), the ability to operate on **user-specific data** without exposing it, understanding high-level **security expertise** and breaking down concepts, and **reasoning** about data in a multi-step fashion.
11. General-purpose models do not perform as well as needed on some security tasks due to the **lack of publicly available security data**, limited depth of security content, and their design to avoid sensitive use cases like malware analysis, which are crucial for practitioners.
12. This necessitates the development of **security-focused LLMs** trained on a variety of cybersecurity-specific content and tasks, designed to operate across multiple security platforms and environments.
13. The training process for SecLM involves starting from a **robust foundational model**, followed by continued pre-training on security-specific content (blogs, threat intelligence, IT books), and supervised fine-tuning on tasks like script analysis and query generation. Lightweight, parameter-efficient tuning (PET) can be used for sensitive user data.
14. SecLM employs a **flexible planning framework** that enables the dynamic use of tools and interaction among multiple domain-specialized agents to reason over data. This multi-step reasoning is crucial for complex tasks like analyzing Advanced Persistent Threat (APT) group tactics.
15. **MedLM** is a family of foundation models fine-tuned for the healthcare industry, built on Med-PaLM, an LLM designed for medical applications. The goal is to improve health outcomes by making technology available to researchers, clinicians, and patients.
16. Medical question-answering (QA) is a grand challenge due to the vast and evolving nature of medical knowledge. LLMs have shown promising results on medical QA benchmarks, demonstrating the ability to understand and apply complex medical concepts.
17. Gen AI has the potential to transform medicine in numerous ways, such as empowering users to ask context-aware questions based on their health records, **triaging patient messages** for clinicians, enhancing patient intake, providing feedback on clinician-patient conversations, and helping clinicians tackle unfamiliar scenarios.
18. Evaluating medical AI systems like Med-PaLM goes beyond multiple-choice accuracy, including a **qualitative assessment rubric** by expert clinicians covering factuality, reasoning, helpfulness, health equity, and potential harm in long-form answers.
19. **Med-PaLM was the first AI system to exceed the passing mark on US Medical License Exam (USMLE)-style questions**, reaching 67%, and Med-PaLM 2 was the first to reach **expert-level performance** at 86.5% accuracy on this benchmark.
20. The integration of medical technology requires a thoughtful, phased approach starting with **retrospective evaluations**, moving to prospective observational studies, and finally prospective interventional studies in real clinical environments to ensure **robustness and reliability**.
21. While medically specialized models like Med-PaLM show value, excelling in one medical domain task doesn't guarantee success in another, and each specific task requires validation and adaptation.

These examples in cybersecurity and healthcare showcase the possibilities of LLMs in solving domain-specific problems, emphasizing the importance of combining these models with human expertise and careful implementation.

### Notebook Summary

#### Notebook 1 - Fine tuning a custom model

[Colab Notebook Link](https://colab.research.google.com/github/singhvis29/kaggle-genai-course/blob/main/Day_4_Fine_tuning_a_custom_model.ipynb)

1. In this notebook, we fine-tune an LLM to perform a classification task.
 * Data: 20 Newsgroups Text Data
 * Model: `gemini-1.5-flash-001`
2. We only use 50 samples per class for fine-tuning. this Technique (parameter-efficient fine-tuning, or PEFT) updates a relatively small number of parameters and does not require training a new model or updating the large model.
3. Before fine-tuning, we perform evaluation on existing models -
 * Passing the text directly
 * Different types of prompt engineering
   We do not obtain desired results with this approach
4. Tuning the model enables us to use the model without any prompts or system instructions and outputs succinct text from the classes you provide in the training data
5. The data contains both input text (the processed posts) and output text (the category, or newsgroup), and we can configure the training using hyperparameters - epoch_count, batch_size, learning_rate
6. We use `client.tunings.tune` API to create and run a tuning job and pass the configuration using `types.CreateTuningJobConfig` . client.tunings.tune uses input data in specific format - records as dictionaries with fields textInput and output
7. Tuning jobs are queued and run in the backgroud. We set up a method to check if a fine-tuned models exists, if not then we use an existing model
8. We calculate the model accuracy of the fine-tuned model. We see that it is more accurate than stock model. We also observe that token usage of the fine-tuned model is significantly less due to model given pre-defined classes as response instead of verbose response.

#### Notebook 2 - Google Search Grounding

[Colab Notebook Link](https://colab.research.google.com/github/singhvis29/kaggle-genai-course/blob/main/Day_4_Google_Search_grounding.ipynb)

1. In this notebook, we implement grounding with Google Search results in the Gemini API. Search grounding is similar to using the RAG system. The model generates Google Search queries and invokes the searches automatically, retrieving relevant data from Google's index of the web and providing links to search suggestions that support the query, so users can verify the sources.
2. We can ask Gemini and ground it's answers using Google Search in the Google AI studio by enabling grounding under 'tools'
3. To enable search grounding, we can specify it as a tool: `google_search`. Like other tools, this is supplied as a parameter in `GenerateContentConfig`, and can be passed to `generate_content` calls as well as `chats.create` (for all chat turns) or `chat.send_message` (for specific turns).
4. When search grounding is used, the model returns extra metadata that includes links to search suggestions, supporting documents and information on how the supporting documents were used.
5. The `grounding_supports` in the metadata provide a way for you to correlate the grounding chunks used to the generated output text. The API provides a text reference and confidence score to validate the grounding


## Day-5: MLOps for Generative AI

### Whitepaper Summary

1. **The core theme is adapting MLOps principles for Generative AI (Gen AI) applications built on Foundation Models (FMs)**, recognizing that while the fundamental tenets remain, novel challenges and practices emerge.
2. Gen AI systems have a distinct lifecycle, starting with **Discovering** suitable FMs, moving through **Development and Experimentation** (especially prompt engineering and chaining), **Data Practices**, **Evaluation**, and finally **Deployment**, **Monitoring**, and **Governance**. The operationalization of the FMs themselves is a specialized task, often handled by a few organizations, allowing most to focus on adapting existing FMs.
3. **The Foundational Model Paradigm introduces key shifts**: FMs are multi-purpose (not task-specific like traditional predictive models), exhibit 'emergent properties', and are highly sensitive to input. This makes the **"prompted model component"** (a combination of a model and a prompt) the core unit of a Gen AI application, shifting focus from just the model.
4. **Development and Experimentation are iterative**, driven by evaluation feedback. **Prompt Engineering** (crafting and refining prompts) is a crucial activity. Prompt artifacts themselves have a **hybrid nature**, parts acting like data (few-shot examples, knowledge bases) and parts like code (templates, guardrails), requiring different management strategies (data-centric and code-centric MLOps practices).
5. **Chaining and Augmentation** are essential techniques to enhance FMs, addressing limitations like recency and hallucinations. This involves orchestrating multiple prompted models, external APIs, and logic. Key patterns include **Retrieval Augmented Generation (RAG)**, which grounds outputs with external data, and **Agents**, which use LLMs as mediators to interact with tools. Unlike traditional ML, developing Gen AI chains often requires **end-to-end experimentation and evaluation** rather than isolated component testing.
6. **Tuning and Training** (like supervised fine-tuning or Reinforcement Learning from Human Feedback - RLHF) remain methods to adapt FMs for specific tasks or to align with human preferences, sharing similar MLOps requirements with traditional model training, such as artifact tracking and evaluation.
7. **Data Practices shift focus from training data to diverse adaptation data**. A single Gen AI application can use multiple data types from various sources, including conditioning prompts, few-shot examples, grounding/augmentation data, task-specific datasets for fine-tuning, human preference datasets, and the base FM's pre-training corpora. This complexity requires new pipelines for managing, evolving, adapting, and integrating these diverse data types in a reproducible and versionable way, moving beyond traditional ETL. Large models are increasingly used for **synthetic data generation, correction, and augmentation**.
8. **Evaluation is a core activity** for any Gen AI system. While initial evaluation can be manual, **automation is necessary** for speed and reliability. However, automated Gen AI evaluation is challenging due to the complexity and high-dimensionality of outputs, the lack of ground truth, and subjectivity. Custom methods, including using another LLM as a judge ("LLM as a Judge" or "Autorater"), are often needed and must align with human judgment. Evaluation must also include tests for **adversarial prompting**.
9. **Deployment involves operationalizing complex Gen AI systems**, composed of many components (prompts, models, data stores), distinct from deploying the base FMs. **Standard software engineering practices (version control, CI/CD) are applied** to all components, including new artifacts like prompt templates and chain definitions. CI faces challenges with generating comprehensive test cases for open-ended outputs and reproducibility issues. Deploying the large **Foundation Models** requires significant compute resources (GPUs, TPUs), necessitating **infrastructure validation** and **model optimization techniques** like quantization, distillation, and pruning to reduce resource needs.
10. **Governance extends beyond traditional model lineage** to include tracking all components in a chain (data, models, code, evaluation) for auditability, debugging, and transparency. Traditional MLOps/DevOps governance practices for data, tuned models, and code still apply.
11. **Agents represent a significant evolution** and have unique operational demands. Their core components are a Foundation Model, Instructions, and Tool Descriptions. Managing diverse tools for agents requires a centralized **Tool Registry**. Agent evaluation has specific stages like tool unit testing, tool selection evaluation, and reasoning evaluation, in addition to end-to-end metrics.
12. **Observability and Memory** are crucial for understanding agent behavior, utilizing short-term (conversation history) and long-term memory to provide context and enable traceability.
13. **AI platforms like Vertex AI** play a critical role by providing a unified environment and tools specifically tailored for the Gen AI lifecycle. Vertex AI offers features for discovering models (Model Garden), prototyping (AI Studio, Notebooks), customizing models (Training & Tuning, including SFT, RLHF, Distillation), orchestrating workflows (Pipelines), chaining and augmenting (Grounding, Extensions, RAG building blocks like Vector Search, Feature Store, Search), evaluating (Experiments, TensorBoard, evaluation pipelines, AutoSxS, Autoraters), predicting (Endpoints, with built-in responsible AI features like citation checks, safety scores, watermarking), and governing (Feature Store, Model Registry, Dataplex).
14. The paper concludes by reinforcing that **MLOps principles are adaptable and essential for building scalable, robust production Gen AI applications** now and in the future.

[NotebookLM Link](https://notebooklm.google.com/notebook/ded5b819-37ec-494f-b652-0cb10e3a8207) (please request access) 
