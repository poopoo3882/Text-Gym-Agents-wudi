from langchain_openai import AzureChatOpenAI, ChatOpenAI


llm = ChatOpenAI(base_url='http://localhost:11434/v1',api_key='ollama',model='llama3.1')
usage={'token': 0, 'cost': 0}
messages = []
messages.append({"role": "user",  "content":"hello!"})
response = llm.invoke(messages)
print(response)
token_used = response.response_metadata['token_usage']['total_tokens']
response=response.content
print(token_used)