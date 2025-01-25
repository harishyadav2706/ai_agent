from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import gradio as gr

template = """You act as interactive AI Chatbot, Who help user to provide answer to their question. 
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="gemma")

chain = prompt | model

def generate(question, history):
    return chain.invoke({"question": question, "history": history})

with gr.Blocks(theme="soft", fill_height=True, fill_width=True) as demo:
    with gr.Row(equal_height=True, variant="panel"):
        gr.Markdown(
            "# AI Companion",
            container=True
        )
    with gr.Row():
        with gr.Column():
            gr.Image(
                "https://th.bing.com/th/id/OIP.6WOkEVbGywLjo5gTZXJmgwHaEW?rs=1&pid=ImgDetMain",
                elem_id="ollama-logo",
                show_download_button=False,
                show_label=False,
                height=180,
                width=300,
                show_fullscreen_button=False
            )
            gr.Markdown(
            "### An AI companion chatbot is a virtual assistant designed to provide personalized, human-like interactions. Available 24/7, it adapts to user needs, offering support for tasks, casual conversations, and emotional well-being. From productivity to mental health and learning, AI chatbots enhance daily life with intelligent, empathetic engagement.",
            container=True
        )

        with gr.Column(scale=3):
            gr.ChatInterface(
                generate,
                multimodal=True,
                type="messages",
            )
        
demo.launch()