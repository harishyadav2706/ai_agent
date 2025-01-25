from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import gradio as gr

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="gemma")

chain = prompt | model

def generate(question, history):
    return chain.invoke({"question": {question}})

gr.ChatInterface(
    generate,
    type="messages",
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me any question", container=False, scale=7),
    title="AI Companion",
    description="Ask AI Companion any question",
    theme="ocean",
    cache_examples=True,
).launch()