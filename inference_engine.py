import socket
import queue
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import random
import os
import time
import csv

prompt_queue = queue.Queue()

def start_client(message, host='0.0.0.0', port=3000):
    """CALL THIS INSTANCE 4 TIMES OR MODIFY TO ACCOMODATE"""
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

def load_tokenizer(model_name): 
    return AutoTokenizer.from_pretrained(model_name)

def load_all_models(model_names, serverPorts): 
    """
    Receives: a list of str model names
    Return: a list of model objects 
    """
    models = []
    for i, model_name in enumerate(model_names):
        models.append((AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            load_in_8bit=True), serverPorts[i]))
    return models

def from_model_generate(model, tokenizer, prompt): 
    """
    Receives: model, tokenizer and str prompt
    Returns: the str output
    """
    text_streamer = TextStreamer(tokenizer)

    # Tokenize the prompt and generate inputs
    inputs = tokenizer(prompt, truncation=True, max_length=MAX_TOKEN//2, return_tensors="pt").to("cuda")  # Move inputs to GPU as well
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        streamer=text_streamer,  # Stream the output tokens
        max_new_tokens=MAX_TOKEN,  # You can adjust the number of tokens generated
        do_sample=True,  # Use sampling (set to False for greedy generation)
        top_k=50,  # Limit the sampling pool to top-k tokens
        top_p=0.95,  # Use nucleus sampling (top-p)
        temperature=0.7  # Control the randomness of predictions
    )
    # Count the number of tokens in the output
    num_output_tokens = outputs.shape[1] - inputs.input_ids.shape[1]

    print(f"Number of tokens generated: {num_output_tokens}")

    decoded_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print()
    return decoded_text

def send_message_to_client(host, port, message):
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

# TCP Server Thread
def run_tcp_server(port=4000):
    host = '0.0.0.0'

    print(f'\nThis is port {port}')
    
    # Start the TCP/IP Socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"TCP Server listening on {host}:{port}")
    
    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        client_id = addr[0] + ":" + str(addr[1])
        
        with client_socket:
            while True:
                # Receive connection from client
                data = client_socket.recv(1024)
                if not data:
                    break
                    
                # Decode the message
                msg = data.decode('utf-8')
                print("Received:", msg)
                
                # Simply pass the entire message to the TTS queue
                prompt_queue.put((msg, client_id))    

def shuffle_without_reassignment(models, texts):
    if len(models) != 4 and len(texts) != 4:
        raise ValueError("The input list must contain exactly 4 items.")
    
    # Create a list of indices
    indices = list(range(4))
    
    # Shuffle the indices until no index remains in its original position
    while True:
        random.shuffle(indices)
        if all(indices[i] != i for i in range(4)):
            break
    
    # Reassign the items based on the shuffled indices
    shuffled_models = [models[indices[i]] for i in range(4)]
    shuffled_texts = [texts[indices[i]] for i in range(4)]
    
    return shuffled_models, shuffled_texts

def get_random_row_from_csv(csv_files):
    # Randomly select a CSV file from the list
    selected_file = random.choice(csv_files)
    
    try:
        # Increase the field size limit
        max_int = 2**31 - 1
        try:
            csv.field_size_limit(max_int)
        except OverflowError:
            csv.field_size_limit(int(max_int / 10))
        
        # Open the CSV file
        with open(selected_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)  # Convert to a list of rows
            
            # Check if the file is empty
            if not rows:
                raise ValueError(f"The file {selected_file} is empty.")
            
            # Randomly select a row
            random_row_index = random.randint(0, len(rows) - 1)
            selected_row = rows[random_row_index]
            
            return selected_row  # Return the selected row
        
    except Exception as e:
        print(f"Error reading file {selected_file}: {e}")
        return None

def map_files_to_paths(dataset_dir, csv_list):
    return [os.path.join(dataset_dir, file) for file in csv_list]

if __name__ == "__main__": 
    # Start the TCP server thread
    """MIGHT WANT TO IMPLEMENT THIS FOUR TIMES IF I WANT REPLY FROM THE TTSX3"""
    tcp_thread = threading.Thread(target=run_tcp_server, daemon=True)
    tcp_thread.start()

    # Declare variables for CSV Dataset fetching
    DATASET_DIR = "datasets"
    csv_list = map_files_to_paths(DATASET_DIR, os.listdir(DATASET_DIR))

    # Declare variables for inference engine
    MAX_TOKEN = 128
    serverIP = "136.244.192.76"
    serverPorts = [3001, 3002, 3003, 3004]
    modelNames = ["models/1", "models/2", "models/3", "models/4"]

    # Load the tokenizer and all models into VRAM
    tokenizer = load_tokenizer(modelNames[0])
    models = load_all_models(modelNames, serverPorts)

    # Start the time to check
start_time = time.time()
while True: 
    try:
        current_client_id = None
        should_interrupt = False
        
        # Check if there's a prompt in the queue
        try: 
            prompt, current_client_id = prompt_queue.get(timeout=1)
        except queue.Empty: 
            # If queue is empty, check elapsed time for auto generation
            elapsed_time = time.time() - start_time
            print(f"\nElapsed time: {elapsed_time}")
            
            if elapsed_time >= 8:  # Time threshold for auto generation
                prompt = get_random_row_from_csv(csv_list)
                print(f"\nAuto-generating with prompt: {prompt}")
            else:
                # No prompt yet and not enough time elapsed, continue waiting
                continue
        
        # Function to check for interruption and modify the outer variable
        def check_for_new_prompt():
            try:
                # Check if there's a new prompt without removing it
                if not prompt_queue.empty():
                    return True
                return False
            except Exception as e:
                print(f"Error checking queue: {e}")
                return False
        
        # Set up for generation
        iterations = random.randint(1, 4)
        print(f"\nStarting generation with {iterations} iterations\n")
        
        prev_models = models
        prev_text = [prompt] * len(models)
        current_text = []
        
        # Main generation loop with interruption checks
        for iteration in range(iterations):
            if should_interrupt:
                print("Interrupting generation due to new prompt")
                break
                
            for i, model in enumerate(prev_models):
                # Check for interruption before each model generation
                if check_for_new_prompt():
                    should_interrupt = True
                    break
                    
                # Generate response
                text = from_model_generate(model[0], tokenizer, prev_text[i])
                send_message_to_client(serverIP, serverPorts[0], text[0])
                current_text.append(text)
                
                # Check for interruption after each model generation
                if check_for_new_prompt():
                    should_interrupt = True
                    break
            
            # If interrupted during model loop, break out of iterations loop
            if should_interrupt:
                break
                
            # Prepare for next iteration if not interrupted
            prev_models, prev_text = shuffle_without_reassignment(prev_models, current_text)
            prev_text = current_text
            current_text = []
        
        # Reset timer regardless of completion or interruption
        start_time = time.time()
        
    except Exception as e: 
        print(f"Error in inference engine: {e}")

    # start_time = time.time()
    # while True: 
    #     try:   
    #         # Attempt to get a prompt with a timeout of 1 second
    #         try: 
    #             prompt, client_id = prompt_queue.get(timeout=1)
    #         except queue.Empty: 
    #             # If the queue is empty, proceed with the loop
    #             prompt = None
    #             client_id = None

    #         # Measured elapsed time since the loop started
    #         elapsed_time = time.time() - start_time

    #         print(f"\nThis is elapsed_time {elapsed_time}")
            
    #         if prompt is None and elapsed_time >=10:  # shutdown signal
    #             prompt = get_random_row_from_csv(csv_list)
    #             iterations = random.randint(1, 4)

    #             print(f"\nThis is iteration {iterations}")
                
    #             # Previous model will talk to the current models and so forth
    #             prev_models = models
    #             prev_text = [prompt, prompt, prompt, prompt]
    #             current_text = []
    #             for _ in range(iterations):
    #                 if 
    #                 for i, model in enumerate(prev_models):
    #                     # Generate a response from each model
    #                     text = from_model_generate(model[0], tokenizer, prev_text[i])
    #                     send_message_to_client(serverIP, serverPorts[0], text[0])
    #                     current_text.append(text)

    #                 # Shuffle the model to ensure they are not talking among itself
    #                 prev_models, prev_text = shuffle_without_reassignment(prev_models, current_text)
    #                 prev_text = current_text
    #                 current_text = []
    #             # Reset start timer for the next condition check
    #             start_time = time.time()
    #         else: 
    #             print(f"\nThis is full message: {prompt}\n")
    #             iterations = random.randint(1, 4)
                
    #             print(f"\nThis is iteration {iterations}")

    #             # Previous model will talk to the current models and so forth
    #             prev_models = models
    #             prev_text = [prompt, prompt, prompt, prompt]
    #             current_text = []
    #             for _ in range(iterations):
    #                 for i, model in enumerate(prev_models):
    #                     # Generate a response from each model
    #                     text = from_model_generate(model[0], tokenizer, prev_text[i])
    #                     send_message_to_client(serverIP, serverPorts[0], text[0])
    #                     current_text.append(text)

    #                 # Shuffle the model to ensure they are not talking among itself
    #                 prev_models, prev_text = shuffle_without_reassignment(prev_models, current_text)
    #                 prev_text = current_text
    #                 current_text = []
    #             # Reset start timer for the next condition check
    #             start_time = time.time()
    #     except Exception as e: 
    #         print(f"Error in inference engine: {e}")
