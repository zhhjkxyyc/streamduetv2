import os
import logging
from flask import Flask, request, jsonify
from dds_utils import ServerConfig
import json
import yaml
from .server import Server

app = Flask(__name__)
servers = {}

@app.route("/")
@app.route("/index")
def index():
    return "Much to do!"

@app.route("/init", methods=["POST"])
def initialize_server():
    args = yaml.safe_load(request.data)
    client_id = args["client_id"]
    if client_id not in servers:
        logging.basicConfig(
            format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
            level="INFO")
        servers[client_id] = Server(args, args["nframes"])
        os.makedirs(f"server_temp_{client_id}", exist_ok=True)
        os.makedirs(f"server_temp_{client_id}-cropped", exist_ok=True)
        return jsonify({"status": "New Init", "client_id": client_id})
    else:
        servers[client_id].reset_state(int(args["nframes"]))
        return jsonify({"status": "Reset", "client_id": client_id})

@app.route("/low/<client_id>", methods=["POST"])
def low_query(client_id):
    file_data = request.files["media"]
    results = servers[client_id].perform_low_query(file_data)
    return jsonify(results)

@app.route("/high/<client_id>", methods=["POST"])
def high_query(client_id):
    file_data = request.files["media"]
    results = servers[client_id].perform_high_query(file_data)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
