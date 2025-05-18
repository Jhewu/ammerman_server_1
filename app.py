from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import os
import socket

# Initialize Flask app
app = Flask(__name__)

# Configure secret key for session management (required by Flask-SocketIO)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key')

# Initialize SocketIO with the Flask app
socketio = SocketIO(app)

# Mock data storage
messages = []

@app.route('/')
def index():
    return render_template('index.html')

def send_message_to_client(host, port, message):
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect the socket to the server's address and port
        client_socket.connect((host, port))
        print(f"Connected to {host}:{port}")
        if isinstance(message, list):
            message = message[0]
        else:
            client_socket.sendall(message.encode('utf-8'))

    finally:
        # Clean up the connection
        client_socket.close()

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send existing messages to the newly connected client
    # emit('update_messages', {'messages': messages})

@socketio.on('message')
def handle_message(data):
    print(f"Received message: {data['message']}")
    
    # Add the message to the list
    messages.append(data['message'])
    
    # Emit the updated messages to all connected clients
    emit('update_messages', {'messages': messages}, broadcast=True)

    # Send message to inference engine
    send_message_to_client("0.0.0.0", 4000, data["message"])

@socketio.on('backend_message')
def handle_backend_message(data):
    print(f"Received message from backend: {data['message']}")
    
    # Process the message exactly like a user message
    message_data = {'message': data['message']}
    handle_message(message_data)

if __name__ == '__main__':
    # Run the Flask app with SocketIO support
    socketio.run(app, debug=True)