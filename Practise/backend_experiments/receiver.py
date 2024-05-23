import socket
# Define the IP address and port to listen on
HOST = '127.0.0.1'  # Localhost
PORT = 8080

# Create a UDP socket
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_socket:
    # Bind the socket to the address and port
    server_socket.bind((HOST, PORT))

    print(f"UDP server is listening on {HOST}:{PORT}")

    # Listen for incoming UDP packets
    while True:
        # Receive data from the client
        data, address = server_socket.recvfrom(
            1024)  # Buffer size is 1024 bytes

        # Print the received data and client address
        print(f"Received data from {address}: {data.decode()}")
