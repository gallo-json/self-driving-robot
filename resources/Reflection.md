# Reflection

This reflection is just about the time I took just to detect stop signs. The reflection for the project as a whole is [here](https://github.com/gallo-json/self-driving-robot/blob/main/resources/Reflection.md).

## Problems I faced

### How does the robot know when exactly to stop?

So the robot can detect and track the stop sign, that is step number one. But when should the robot stop moving forward? Should it stop the instant it sees a stop sign? Or should it move a couple seconds more, then stop? 

At first I tried doing the latter idea, but it didn't work consistently enough. If the robot's speed was too great, sometimes the robot would stop after the stop sign. Other times it would stop to early, move a little forward, repeat that process, and eventually run the stop sign.

To fix this problem I implemented a threshold value. I calculated the area of the bounding box around the stop sign and picked a threshold value (in px squared) through trial and error. If the current bounding box area is less than the threshold value, that is an indication that the stop sign is still far away and the robot can continue moving forward. If the current bounding box area is the same threshold value (give or take some px), that is the cue to the robot to stop for 2 seconds, then continue.

## What I enjoyed about the project

## Improvements

## What I learned by doing this project