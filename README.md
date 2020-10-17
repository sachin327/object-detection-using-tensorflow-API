# object-detection-using-tensorflow-API(don't forgot to change runtime to gpu)
  

# At first i have installed some libraries that are not present on google colab
  
!pip install tensorflow-gpu
  
!pip install trains
  
# then i have cloneed my repositories to colab
  
!git clone --depth 1 https://github.com/sachin327/models
  

!git clone --depth 1 https://github.com/sachin327/scripts
  
!git clone --depth 1 https://github.com/sachin327/workspace
  
# By default, pycocotools do not support per object statistic in eval, so we replace cocoeval.py script with a custom one that does support it.
  
!cp ./object_detection/metrics/cocoeval.py /usr/local/lib/python3.6/dist-packages/pycocotools/
  
%%bash
  
#Compile protos.
  
protoc object_detection/protos/*.proto --python_out=.
  
#Install TensorFlow 2 Object Detection API.
  
cp object_detection/packages/tf2/setup.py .
  
python -m pip install .
  
# save your training image to '//content/workspace/training_demo/images/train' directory and test images to '//content/workspace/training_demo/images/test' with annotations file with them .xml format
  
#Go to scripts directory
  
%cd '//content/scripts/preprocessing'
  
calling scripts(to save .record file to annotations folder)
  
!python generate_tfrecord.py -x '//content/workspace/training_demo/images/train' -l '//content/workspace/training_demo/annotations/label_map.pbtxt' -o 
  
!python generate_tfrecord.py -x '//content/workspace/training_demo/images/test' -l '//content/workspace/training_demo/annotations/label_map.pbtxt' -o 
  
# Then i have downloaded a model from(tensorflow zoo)
  
!wget -nc 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz' -O ./models/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
  
# save our maodel to training_demo/models folder
  
%cd '//content/workspace/training_demo'
  
!tar -xvf  ./models/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz -C ./pre-trained-model
  
# copy pipline.conif file
  
cp ./pre-trained-model/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config ./models/my_ssd_mobilenet_v2
  
# Now its time to make some changes to this file
  
Now its time to update your '//content/workspace/training_demo/models/my_ssd_mobilenet_v2/pipline.config' file
  

first open your file 
  
1. change num_classes acccording to your model
  
2. change batch_size to 2-8
  
3. fine_tune_checkpoint: "pre-trained-model/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"
  
4. fine_tune_checkpoint_type: "detection"
  
5. in train_input_reader
    1. label_map_path: "annotations/label_map.pbtxt"
    2. input_path: "annotations/train.record"
  
6. in eval_input_reader
    1. label_map_path: "annotations/label_map.pbtxt"
    2. input_path: "annotations/test.record
  
# Training model
  
!python model_main_tf2.py --model_dir=models/my_ssd_mobilenet_v2 --pipeline_config_path=models/my_ssd_mobilenet_v2/pipeline.config
  
# Save our model
  
!python ./exporter_main_v2.py --input_type=image_tensor --pipeline_config_path=./models/my_ssd_mobilenet_v2/pipeline.config --trained_checkpoint_dir=./models/my_ssd_mobilenet_v2/ --output_directory=./exported-model/my_model
  
# This is our test result
  
![alt text](https://github.com/sachin327/object-detection-using-tensorflow-API/blob/main/images/test_1.png)
  
![alt text](https://github.com/sachin327/object-detection-using-tensorflow-API/blob/main/images/test_2.png)
  
![alt text](https://github.com/sachin327/object-detection-using-tensorflow-API/blob/main/images/test_3.png)
  
![alt text](https://github.com/sachin327/object-detection-using-tensorflow-API/blob/main/images/test_4.png)
  
![alt text](https://github.com/sachin327/object-detection-using-tensorflow-API/blob/main/images/test_5.png)
  

# Now you can download your model to run it on your desktop
  
![alt text](https://github.com/sachin327/object-detection-using-tensorflow-API/blob/main/images/download.png)
