from middleware.operation_log import OperationLogMiddleware
from werkzeug.utils import redirect

from flask import Flask, url_for

app = Flask(__name__)
# add the custom middleware
app.wsgi_app = OperationLogMiddleware(app.wsgi_app)


@app.route("/")
def hello_world():
    return "Hello World"


@app.route("/hello/<name>")
def hello(name: str):
    return f"Hello {name}!"


@app.route("/apply/<name>")
def hello_redirect(name: str):
    if name == "admin":
        return redirect(url_for("hello_admin"))
    else:
        return redirect(url_for("hello_guest", guest=name))


@app.route("/admin")
def hello_admin():
    return "Hello Admin"


@app.route("/guest/<guest>")
def hello_guest(guest):
    return f"Hello {guest} as Guest"


if __name__ == "__main__":
    app.run(debug=True)
