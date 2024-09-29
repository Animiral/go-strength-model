from matplotlib import rcParams

# make font compatible with thesis, allow formatting
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Computer Modern Roman"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# this will fit in thesis without latex warnings
ideal_figsize = (5.75, 4.2)
big_figsize = (8, 11)  # whole page figure, e.g. hp search
presentation_figsize = (4.2, 2.8)  # fits latex beamer 4:3 with title
presentation_pngsize = (12.6, 6)  # larger size for PNG figure in presentation
