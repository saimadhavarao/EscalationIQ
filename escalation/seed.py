"""Submit generated demo conversations through the public API."""
import argparse
import json
from urllib.request import Request, urlopen
from uuid import uuid4


def seed(base: str):
    def request(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = Request(base.rstrip("/") + "/api" + path, data=data,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=60) as response:
            return json.load(response)

    for sample in request("/demo"):
        ticket = request("/tickets", sample["context"])
        for message in sample["messages"]:
            request(f'/tickets/{ticket["id"]}/messages', {"message_id": str(uuid4()), "message": message})
        print(ticket["id"], sample["name"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    seed(parser.parse_args().base)
