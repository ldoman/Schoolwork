'''
Usage: convertVideo.py [path to top level dir containing videos you want to convert]
'''

import os
import sys
import subprocess

def main(basePath, inFormat = 'mkv', outFormat = 'mp4'):
	# Debug
	print "Path: " + basePath
	print "In: " + inFormat
	print "Out: " + outFormat

	for root, dirs, files in os.walk(basePath):
		for file in files:
			if file.split(".")[-1] == inFormat:
				fileIn = os.path.join(root,file)
				fileOut = os.path.join(root, file.split(".")[0] + '.' + outFormat)
				convert(fileIn, fileOut) 

	print "Done."

def convert(fileIn, fileOut):
	print "In: " + fileIn
	print "Out: " + fileOut
	subprocess.call("vlc -I dummy -vvv %s --sout=#transcode{vcodec=h264,vb=2048,acodec=mp4a,ab=320,channels=2,deinterlace}:standard{access=file,mux=ts,dst=%s}" % (fileIn, fileOut), shell=True)

if __name__ == "__main__":
	main(*sys.argv[1:])
