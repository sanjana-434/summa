import socket
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64

def start_server(host="127.0.0.1", port=65432):
    # Generate RSA key pair (2048-bit)
    key = RSA.generate(2048)
    private_key = key
    public_key = key.publickey()

    # Start socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"RSA Server running on {host}:{port}... Waiting for client.")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    # Send public key to client
    conn.sendall(public_key.export_key())
    print("Public key sent to client.")

    while True:
        data = conn.recv(4096)
        if not data:
            break

        # Decrypt incoming ciphertext
        ciphertext = base64.b64decode(data)
        cipher_rsa = PKCS1_OAEP.new(private_key)
        plaintext = cipher_rsa.decrypt(ciphertext).decode()
        print(f"Received ciphertext: {data.decode()}")
        print(f"Decrypted plaintext: {plaintext}")

        if plaintext.lower() == "exit":
            break

        # Encrypt a reply back to client
        reply = f"Server received: {plaintext}"
        cipher_rsa_enc = PKCS1_OAEP.new(public_key)  # NOTE: Normally use client's public key
        enc_reply = cipher_rsa_enc.encrypt(reply.encode())
        conn.sendall(base64.b64encode(enc_reply))

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    start_server()
