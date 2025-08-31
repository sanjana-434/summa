import socket
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Util import Counter
from Crypto.Random import get_random_bytes
import base64

def start_client(host="127.0.0.1", port=65432):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Step 1: Choose AES mode
    print("Available AES modes: ECB, CBC, CFB, OFB, CTR")
    mode = input("Choose AES mode: ").upper()
    client_socket.sendall(mode.encode())

    key = get_random_bytes(16)
    client_socket.sendall(key)
    print(f"AES Key sent: {base64.b64encode(key).decode()}")

    while True:
        msg = input("Enter message (or 'exit' to quit): ")
        if msg.lower() == "exit":
            break

        if mode == "ECB":
            cipher = AES.new(key, AES.MODE_ECB)
            ciphertext = cipher.encrypt(pad(msg.encode(), AES.block_size))
            payload = base64.b64encode(ciphertext)

        elif mode == "CBC":
            cipher = AES.new(key, AES.MODE_CBC)
            ciphertext = cipher.encrypt(pad(msg.encode(), AES.block_size))
            payload = base64.b64encode(cipher.iv + ciphertext)

        elif mode == "CFB":
            cipher = AES.new(key, AES.MODE_CFB)
            ciphertext = cipher.encrypt(msg.encode())
            payload = base64.b64encode(cipher.iv + ciphertext)

        elif mode == "OFB":
            cipher = AES.new(key, AES.MODE_OFB)
            ciphertext = cipher.encrypt(msg.encode())
            payload = base64.b64encode(cipher.iv + ciphertext)

        elif mode == "CTR":
            nonce = get_random_bytes(8)
            ctr = Counter.new(64, prefix=nonce)
            cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
            ciphertext = cipher.encrypt(msg.encode())
            payload = base64.b64encode(nonce + ciphertext)

        else:
            print("Unsupported mode.")
            continue

        print(f"Sending ciphertext (base64): {payload.decode()}")
        client_socket.sendall(payload)

    client_socket.close()

if __name__ == "__main__":
    start_client()
