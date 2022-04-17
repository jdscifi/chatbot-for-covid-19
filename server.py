from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread
import pickle
from sklearn import *
import json
import numpy as np
import en_core_web_lg
import spacy

MODEL = None
# save
with open('model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

RESPONSES = json.load(open("intent_response.json", "r"))
RESPONSES['default'] = "I am sorry I didn't get you. Can you reframe your question?"

LangProcessor = en_core_web_lg.load()


def encode_message(text):
    X = np.zeros((1, 300))
    doc = LangProcessor(text.decode("utf-8")).vector
    return doc


def accept_incoming_connections():
    """Sets up handling for incoming clients."""
    while True:
        client, client_address = SERVER.accept()
        client.send(bytes("Hola! I am the Covid Counsellor, but you can call me Coco!\nWhat's your name?", "utf8"))
        addresses[client] = client_address
        Thread(target=handle_client, args=(client,)).start()


def handle_client(client):
    global RESPONSES
    name = client.recv(BUFSIZ)
    welcome = 'Welcome %s! What are your Covid Concerns?' % name.decode("utf8")
    client.send(bytes(welcome, "utf8"))
    clients[client] = name.decode("utf8")

    while True:
        msg = client.recv(BUFSIZ)
        if "bye" in msg.decode("utf-8").lower():
            client.send(b"Coco: Goodbye & Stay Safe!")
            break
        y_pred = MODEL.predict([encode_message(msg)])
        client.send(name + b": " + msg)
        client.send(b"Coco: " + bytes(RESPONSES[y_pred[0]], "utf8"))


clients = {}
addresses = {}

HOST = '127.0.0.1'
PORT = 33000
BUFSIZ = 1024
ADDR = (HOST, PORT)

SERVER = socket(AF_INET, SOCK_STREAM)
SERVER.bind(ADDR)


if __name__ == "__main__":
    SERVER.listen(5)
    print("Waiting for connection...")
    ACCEPT_THREAD = Thread(target=accept_incoming_connections)
    ACCEPT_THREAD.start()
    ACCEPT_THREAD.join()
    SERVER.close()
