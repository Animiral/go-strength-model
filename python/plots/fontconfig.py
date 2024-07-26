from matplotlib import rcParams

# make font compatible with thesis, allow formatting
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Computer Modern Roman"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
