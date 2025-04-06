# kaggle-genai-course
This repository is created for notebooks and material used in the the Kaggle 5-day intensive genai course

## Day-1

### Whitepaper Summary

[NotebookLM Link](https://notebooklm.google.com/notebook/95087705-ecfd-4385-816f-1f23105c6201?_gl=1*fyxxyk*_ga*MjExMzc2Mjc0NS4xNzQzNTA0MDg5*_ga_W0LDH41ZCB*MTc0MzU2MDQ2Ni4yLjAuMTc0MzU2MDQ2Ny41OS4wLjA.) (please request access)

### Notebook Summary

Here is a 10-point summary of the whitepaper "Foundational Large Language models & text generation\_v2.pdf":

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


