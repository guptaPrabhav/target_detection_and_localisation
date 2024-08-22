# Object detection code for the RB5 drone.

This repository is essentially a ROS package.

### To test out the ROS node-

1. Clone the repository
2. Build the package with `catkin_make`
3. Run the ROS Node `rosrun image_subscriber image_subscriber.py` while the expected topics are published.

This will most probably not work due to issues with the video feed format published by the voxl_mpa_to_ros node of the RB5 drone. But the script below should work normally.

### To test out RTSP node-

1. Make sure the drone and the laptop/computer are on the same network
2. Make sure the RTSP server is running on the drone - `voxl-streamer -s -p 8901 -d 1 -i hires_small_color`
3. Execute `test_script.py` as you would a normal python script within the docker container as given [here](https://github.com/guptaPrabhav/rb5_noetic)

