Application to use keras model which was trained with https://github.com/fizyr/keras-retinanet

Usage:

	1. convert keras model to tensorflow saved model, 
		to achieve this you can use python script keras-model-to-tensorflow-model.py, 
		but first copy script into keras-retinanet root folder
		script argumets: <keras model file path> <tensorflow model output folder path>
	2. Compile and run application
		2.1 compile java sources: mvn package
		2.2 run application: java -jar objectdetector-1.0-SNAPSHOT.jar <arguments>
		  arguments: 
			--image <arg>     image file path
			--model <arg>     tensorflow model folder path
			--labels <arg>    labels file path (optional)
			--gui             run gui (optional)
		
