#!/usr/bin/env python
import pyyrtpet as yrt

# Note: This file is not to be executed, but simply to be used as a documentation

scanner = yrt.Scanner("<path to the scanner's json file>")

# Prepare an empty histogram
outHis = yrt.Histogram3DOwned(scanner)
outHis.allocate()

# Read an image to Forward-project
imgParams = yrt.ImageParams("<path to the image parameters file>")
inputImage = yrt.ImageOwned(imgParams, "<path to input image>")

projectorType = yrt.OperatorProjector.ProjectorType.DD_GPU
# Available projectors: SIDDON, DD, DD_GPU
yrt.forwProject(scanner, inputImage, outHis, projectorType)

outHis.writeToFile("<path to save output histogram>")
