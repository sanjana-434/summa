import socket
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from Crypto.Util import Counter
import base64

def start_server(host="127.0.0.1", port=65432):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"AES Server running on {host}:{port}... Waiting for client.")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    # Receive mode
    mode = conn.recv(1024).decode()
    print(f"AES Mode selected by client: {mode}")

    # Receive key
    key = conn.recv(1024)
    print(f"AES Key received: {base64.b64encode(key).decode()}")

    while True:
        data = conn.recv(4096)
        if not data:
            break

        raw = base64.b64decode(data)

        if mode == "ECB":
            cipher = AES.new(key, AES.MODE_ECB)
            plaintext = unpad(cipher.decrypt(raw), AES.block_size)

        elif mode == "CBC":
            iv, ciphertext = raw[:16], raw[16:]
            cipher = AES.new(key, AES.MODE_CBC, iv)
            plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

        elif mode == "CFB":
            iv, ciphertext = raw[:16], raw[16:]
            cipher = AES.new(key, AES.MODE_CFB, iv=iv)
            plaintext = cipher.decrypt(ciphertext)

        elif mode == "OFB":
            iv, ciphertext = raw[:16], raw[16:]
            cipher = AES.new(key, AES.MODE_OFB, iv=iv)
            plaintext = cipher.decrypt(ciphertext)

        elif mode == "CTR":
            nonce, ciphertext = raw[:8], raw[8:]
            ctr = Counter.new(64, prefix=nonce)
            cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
            plaintext = cipher.decrypt(ciphertext)

        else:
            plaintext = b"Unsupported Mode"

        print(f"Ciphertext received (base64): {data.decode()}")
        print(f"Decrypted plaintext: {plaintext.decode(errors='ignore')}")

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    start_server()
