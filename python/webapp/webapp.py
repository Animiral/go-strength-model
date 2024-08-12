from flask import Flask, request, render_template, flash
import model.run
import os
import tempfile

app = Flask(__name__)
# app.config["UPLOAD_DIR"] = "uploads/" # replace with temp dir
app.config["KATAGO"] = os.environ.get("KATAGO", "katago")
app.config["KATAMODEL"] = os.environ.get("KATAMODEL")
app.config["KATACONFIG"] = os.environ.get("KATACONFIG")
app.config["STRMODEL"] = os.environ.get("STRMODEL")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 # 100 kB limit, one game should be ~2kB

def upload(sgfs):
  # sgfdir = app.config["UPLOAD_DIR"]
  sgfdir = tempfile.TemporaryDirectory()
  for sgf in sgfs:
    if sgf:
      sgfpath = os.path.join(sgfdir.name, sgf.filename)
      sgf.save(sgfpath)
  return sgfdir

def process(sgfdir, player):
  args = {
    "inputs": [f.path for f in os.scandir(sgfdir)],
    "katago": app.config["KATAGO"],
    "katamodel": app.config["KATAMODEL"],
    "kataconfig": app.config["KATACONFIG"],
    "model": app.config["STRMODEL"],
    "featurename": "pick",
    "playername": player
  }
  _, rating, rank = model.run.main(args)[0]
  return rating, rank

@app.route("/", methods=["GET", "POST"])
def main():
  rating = None
  rank = None
  error = None
  player = None

  if request.method == "POST":
    player = request.form.get("player", None)

    try:
      with upload(request.files.getlist("sgfs")) as sgfdir:
        rating, rank = process(sgfdir, player)

    except model.run.KatagoException as e:
      error = str(e)

  return render_template("index.html", rating=rating, rank=rank, player=player, error=error)

if __name__ == "__main__":
  # rudimentary env check
  assert os.path.isfile(app.config["KATAGO"])
  assert os.path.isfile(app.config["KATAMODEL"])
  assert os.path.isfile(app.config["KATACONFIG"])
  assert os.path.isfile(app.config["STRMODEL"])
  app.run(debug=True)

