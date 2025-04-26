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

### Notebook Summary

#### Notebook 1 - Document RAG with Q&A
1. RAG is used to mitigate the two drawbacks of LLM i.e.- 1) that they only "know" the information that they were trained on, and 2) that they have limited input context windows.
2. Indexing happens ahead of time, and allows you to quickly look up relevant information at query-time. When a query comes in, you retrieve relevant documents, combine them with your instructions and the user's query, and have the LLM generate a tailored answer in natural language using the supplied information.
3. In this notebook we use the Gemini API to create a vector database, retrieve answers to questions from the database and generate a final answer. We use Chroma, an open-source vector database.
4. We use Gemini client.models.embed_content API to embed the input documents. The embedding used are 'models/text-embedding-004'
5. We create a function - GeminiEmbeddingFunction to generate embeddings for documents. The function can be used to generate embeddings for documents or queries
6. We create a Chroma database client that uses the GeminiEmbeddingFunction and populate the database with the documents
7. To search the Chroma database, we call the query method. Note that we also switch to the retrieval_query mode of embedding generation. We retreive the relevant passage wrt our query from the db
8. We finally assemble a generation prompt to have the Gemini API generate a final answer.

#### Notebook 2 - Embeddings and similarity scores
1. Gemini supports 4 models for embedding task -
 * models/embedding-001
 * models/text-embedding-004
 * models/gemini-embedding-exp-03-07
 * models/gemini-embedding-exp
2. We embed the list of texts using the models/text-embedding-004 model
3. A similarity score of two embedding vectors can be obtained by calculating their inner product. If $\mathbf{u}$ is the first embedding vector, and $\mathbf{v}$ the second, this is $\mathbf{u}^T \mathbf{v}$. As the API provides embedding vectors that are normalised to unit length, this is also the cosine similarity.


## Day-3


### Whitepaper Summary

### Notebook Summary

#### Notebook 1 - Function Calling with Gemini API

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


## Day-4

### Whitepaper Summary

### Notebook Summary

#### Notebook 1 - Fine tuning a custom model

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
