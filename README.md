# Glasses-detection
Train the haar cascde to detect the glasses on the face.

steps:
1. opencv_createsamples -img glass.jpg -bg bg.txt -info info/info.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 800
2. opencv_createsamples -info info/info.lst -num 800 -w 20 -h 20 -vec positives.vec
3. opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 640 -numNeg 400 -numStages 20 -w 20 -h 20
