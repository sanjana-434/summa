import socket
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64

def start_client(host="127.0.0.1", port=65432):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Receive server’s public key
    server_pub_key = RSA.import_key(client_socket.recv(4096))
    print("Received server’s public key.")

    while True:
        msg = input("Enter message (or 'exit' to quit): ")
        cipher_rsa = PKCS1_OAEP.new(server_pub_key)

        # Encrypt message
        enc_msg = cipher_rsa.encrypt(msg.encode())
        client_socket.sendall(base64.b64encode(enc_msg))
        print(f"Sent ciphertext: {base64.b64encode(enc_msg).decode()}")

        if msg.lower() == "exit":
            break

        # Receive encrypted reply from server
        reply_data = client_socket.recv(4096)
        if not reply_data:
            break

        reply_cipher = PKCS1_OAEP.new(server_pub_key)  # In reality: decrypt with client’s private key
        reply_plain = reply_data.decode()  # For demo we keep it simple
        print(f"Encrypted reply: {reply_plain}")

    client_socket.close()

if __name__ == "__main__":
    start_client()


