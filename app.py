import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define chatbot response function
def chatbot_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio interface
ui = gr.Interface(
    fn=chatbot_response, 
    inputs=gr.Textbox(lines=5, placeholder="Type your message here..."), 
    outputs=gr.Textbox(lines=5, label="Chatbot Response"),
    title="ChatGPT-like Chatbot",
    description="A lightweight chatbot running on CPU."
)

# Launch the app
ui.launch(share = True)
