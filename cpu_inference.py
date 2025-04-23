from llama_cpp import Llama
import socket

def start_client(message, host='0.0.0.0', port=3001):
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect the socket to the server's address and port
        client_socket.connect((host, port))
        print(f"Connected to {host}:{port}")
        client_socket.sendall(message.encode('utf-8'))

    finally:
        # Clean up the connection
        client_socket.close()

def stream_output(llm_instance):
    response = llm_instance.create_completion(
        prompt=prompt,
        max_tokens=MAX_TOKEN, 
        echo=False,
        stop=["User:", "Assistant:", "<|endoftext|>", "</s>", "<|end_of_text|>", "Q:"],
        stream=True  # Enable streaming mode
    )
    
    # print(prompt, end='', flush=True)  
    text = ''
    for token in response:
        if 'choices' not in token or len(token['choices']) == 0: continue
        choice = token['choices'][0]
        
        if 'text' in choice:
            new_text = choice['text']
            
            # Print the new text as it comes in (you can also process each token here)
            # print(new_text, end='', flush=True)  
            
            text += new_text
        
    return text


if __name__ == "__main__": 
    MAX_TOKEN = 256

    # Load the Model and Tokenizer
    llm = Llama(
        model_path="quantized_model/unsloth.Q4_K_M.gguf", 
        temperature=0.7,  # Adjust as needed
        max_tokens=MAX_TOKEN,   # Adjust as needed
        n_ctx=MAX_TOKEN,       # Context window size
        verbose=False
    )

    prompt = "I like Chicken Nuggets"

    # Use the function to stream output
    full_response = prompt + " " + stream_output(llm)
    start_client(full_response, host='0.0.0.0', port=3001)


