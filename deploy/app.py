import os
from snake_core import SnakeAgent
from snake_web import create_app

agent = SnakeAgent.load(os.environ.get("SNAKE_MODEL", "snake_model.pt"), device="cpu")
app = create_app(agent)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
