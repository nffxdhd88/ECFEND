
---

# News Authenticity Analysis Using LLMs

## Model Versions
The models used are:
- **Llama2-7b**
- **Llama3-13b-instruct**
- **GLM4-9b-chat**

The model parameters are sourced from Huggingface.

## Prompt Used
The prompt used for all models is:
```
Analyze the given news item to determine its authenticity and give your choice:

News Content: 
{news_content}

Evidences: 
{evidences}

Verdict: Is the news item true or false?

[Options]:

True
False

Answer:
```

- `{news_content}` is filled with the claim from the news.
- `{evidences}` is filled with multiple pieces of evidence from the news, with a maximum token limit of 2048.

## Determining News Authenticity
The news authenticity is determined by inputting the prompt into the LLM. If the output contains "True" or "False," it is classified accordingly. In the SnopesCG dataset, categories like "most true" are classified as "true" in the final evaluation.

### Example:
```
***LLM OUTPUT***

"answer:

false

reasoning:

the news item claims that cabbage patch dolls were designed to get people accustomed to the appearance of mutants following a thermonuclear war or were modeled upon mentally defective children. however, upon analyzing the content, it becomes clear that the information provided is a collection of unrelated statements, claims, and references that do not support the main claim. the article jumps between different topics, including:

* a reference to a snopes article that debunks the claim
* a mention of the"

*******************
```
This example is classified as "false."

## Data Storage
In the `PolitFact` and other folders within this file, the outputs of different models for each dataset are stored. These include:
- Sample ID
- Claim text from the news
- Ground truth
- LLM's predicted label
- LLM's response content

---