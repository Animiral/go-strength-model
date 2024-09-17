#!/usr/bin/env python3
# Contains the flask app.
# !! Importing this file immediately sets up app.config, which may fail and call sys.exit(1) !!

from flask import Flask, request, render_template, flash
import model.run
import os
import tempfile

app = Flask(__name__)

def configure(app):
  """Set up app.config from os.environ, with sanity checks"""
  # KATAGO must be an executable
  katago_path = os.environ.get("KATAGO", "katago")
  if not os.path.isfile(katago_path) or not os.access(katago_path, os.X_OK):
    print(f"Error: KATAGO path '{katago_path}' is not a valid executable.", file=sys.stderr)
    sys.exit(1)

  # KATAMODEL must be a model file
  katamodel_path = os.environ.get("KATAMODEL")
  if not os.path.isfile(katamodel_path):
      print("Error: KATAMODEL path '{katamodel_path}' is not a model file.", file=sys.stderr)
      sys.exit(1)

  # KATACONFIG must be a config file
  kataconfig_path = os.environ.get("KATACONFIG")
  if not os.path.isfile(kataconfig_path):
      print("Error: KATACONFIG path '{kataconfig_path}' is not a config file.", file=sys.stderr)
      sys.exit(1)

  # STRMODEL must be a model file
  strmodel_path = os.environ.get("STRMODEL")
  if not os.path.isfile(strmodel_path):
      print("Error: STRMODEL path '{strmodel_path}' is not a model file.", file=sys.stderr)
      sys.exit(1)

  # app.config["UPLOAD_DIR"] = "uploads/" # replace with temp dir
  app.config["KATAGO"] = katago_path
  app.config["KATAMODEL"] = katamodel_path
  app.config["KATACONFIG"] = kataconfig_path
  app.config["STRMODEL"] = strmodel_path
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
    "playername": player,
    "scale": [334.0281191511932, 1595.094753057906]
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
    sgfs = request.files.getlist("sgfs")

    if not all(sgf.filename.endswith(".sgf") for sgf in sgfs):
      error = "Only SGF records are supported."
      return render_template("index.html", rating=None, rank=None, player=None, error=error)

    try:
      with upload(sgfs) as sgfdir:
        if not player:
          player = model.run.findOmnipresentPlayerInFiles([f.path for f in os.scandir(sgfdir)])
        rating, rank = process(sgfdir, player)

    except model.run.KatagoException as e:
      error = str(e)

    except model.run.PlayerDetectException as e:
      error = str(e)

  return render_template("index.html", rating=rating, rank=rank, player=player, error=error)

# This runs on import, for smooth usage with gunicorn server
configure(app)

if __name__ == "__main__":
  app.run(debug=True)

