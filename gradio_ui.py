import os
import gradio as gr
import requests
import time

# --- Configuration ---
# This assumes your FastAPI server is running on the default port 8000
API_BASE_URL = "http://127.0.0.1:8000"
UPLOAD_URL = f"{API_BASE_URL}/upload"
ASK_URL = f"{API_BASE_URL}/ask"


# --- API Communication Functions ---

def upload_pdf(file_obj):
    """
    Called by the 'Process Document' button.
    Uploads the PDF to the /upload endpoint.
    """
    if file_obj is None:
        return (
            [],
            "No file selected. Please upload a PDF.",
            gr.update(interactive=False),
            gr.update(interactive=False)
        )
    
    print(f"Sending file to {UPLOAD_URL}...")
    try:
        # Get the filename for the 'multipart/form-data' header
        filename = os.path.basename(file_obj.name)
        
        # --- THIS IS THE FIX ---
        # We must explicitly open the file by its path in binary-read mode ('rb')
        # and pass that file handle to requests.
        with open(file_obj.name, 'rb') as f_in:
            files = {'file': (filename, f_in, 'application/pdf')}
            
            # Make the request *inside* the 'with' block
            response = requests.post(UPLOAD_URL, files=files, timeout=60)
        
        response.raise_for_status()  # Raise an exception for 4xx or 5xx errors
        
        # Success
        print("File uploaded successfully.")
        return (
            [],
            response.json().get("message", "Processing successful!"),
            gr.update(interactive=True),  # Enable question box
            gr.update(interactive=True)   # Enable ask button
        )

    except requests.exceptions.HTTPError as e:
        # Handle server-side errors (like 400, 500)
        detail = e.response.json().get('detail', 'Unknown server error')
        print(f"Server error uploading file: {detail}")
        return (
            [],
            f"Server Error: {detail}",
            gr.update(interactive=False),
            gr.update(interactive=False)
        )
    except Exception as e:
        # Handle other errors (like connection errors)
        print(f"Error uploading file: {e}")
        return (
            [],
            f"Error: {e}",
            gr.update(interactive=False),
            gr.update(interactive=False)
        )

def ask_question(question, history):
    """
    Called by the 'Ask' button.
    Sends the question to the /ask endpoint and yields the streaming answer.
    """
    if not question:
        yield "Please ask a question."
        return

    print(f"Asking question: {question}")
    try:
        response = requests.post(
            ASK_URL,
            json={"question": question},
            timeout=60
        )
        response.raise_for_status()
        
        answer = response.json().get("answer", "No answer received.")
        
        # Simple "typing" effect
        for i in range(0, len(answer), 3):
            time.sleep(0.01)
            yield answer[:i+3]

    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get('detail', 'Unknown server error')
        print(f"Server error asking question: {detail}")
        yield f"Server Error: {detail}"
    except Exception as e:
        print(f"Error asking question: {e}")
        yield f"Error: {e}"

def handle_user_question(question, history):
    """
    Manages the chat history and calls the API function.
    """
    # 1. Add user's question to the history
    history.append([question, None])
    yield history, ""  # Update chatbot, clear textbox

    # 2. Get the bot's response
    # We yield from the generator to get the streaming effect
    response_generator = ask_question(question, history)
    
    # 3. Stream the response into the history
    bot_response = ""
    for chunk in response_generator:
        bot_response = chunk
        history[-1][1] = bot_response
        yield history, "" # Update chatbot, keep textbox clear

# --- Gradio UI Layout ---

print("Starting Gradio UI...")

with gr.Blocks(theme=gr.themes.Soft(), title="Chat with your PDF") as demo:
    gr.Markdown("# Chat with your PDF (Gemini Edition)")
    gr.Markdown("Powered by FastAPI, LangChain, and Google Gemini. Upload a PDF and ask questions about it.")
    
    with gr.Row():
        # --- Left Column: PDF Upload ---
        with gr.Column(scale=1):
            gr.Markdown("## 1. Upload Document")
            pdf_upload = gr.File(
                label="Upload your PDF",
                file_types=[".pdf"]
            )
            process_button = gr.Button("Process Document", variant="primary")
            status_box = gr.Textbox(
                label="Status",
                placeholder="Upload a PDF and click 'Process'",
                interactive=False
            )
            
        # --- Right Column: Chat Interface ---
        with gr.Column(scale=2) as chat_column:
            gr.Markdown("## 2. Ask Questions")
            
            # We are building the chat UI manually for more control
            chatbot = gr.Chatbot(
                label="Chat History",
                bubble_full_width=False,
                height=400
            )
            
            question_box = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here...",
                interactive=False  # Disabled by default
            )
            
            ask_button = gr.Button(
                "Ask",
                variant="primary",
                interactive=False  # Disabled by default
            )

    # --- UI Event Wiring ---
    
    # When the "Process" button is clicked...
    process_button.click(
        fn=upload_pdf,
        inputs=[pdf_upload],
        outputs=[
            chatbot,        # Clear the chatbot history
            status_box,     # Update the status message
            question_box,   # Enable the question textbox
            ask_button      # Enable the ask button
        ]
    )
    
    # When the "Ask" button is clicked...
    ask_button.click(
        fn=handle_user_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box] # Update chatbot, clear textbox
    )
    
    # When the user presses Enter in the textbox...
    question_box.submit(
        fn=handle_user_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box] # Update chatbot, clear textbox
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)
    print(f"connect to http://127.0.0.1:{8000}")


