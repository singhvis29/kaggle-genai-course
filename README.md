# kaggle-genai-course
This repository is created for notebooks and material used in the the Kaggle 5-day intensive genai course

## Day-1

### Whitepaper Summary

*NotebookLM Link* - https://notebooklm.google.com/notebook/95087705-ecfd-4385-816f-1f23105c6201?_gl=1*fyxxyk*_ga*MjExMzc2Mjc0NS4xNzQzNTA0MDg5*_ga_W0LDH41ZCB*MTc0MzU2MDQ2Ni4yLjAuMTc0MzU2MDQ2Ny41OS4wLjA. (please request access)

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
