import os

import cv2
import face_recognition
import numpy as np
import open3d as o3d
from PIL import Image

from accelerate import Accelerator
from transformers import pipeline

device = Accelerator().device
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")

previous_depths = {}
test= False
def find_depth(found_object,face_id):
    if test is not False:
        #demo code for showing off
        if found_object is None or found_object.size == 0:
            return None
        # convert grey and to np array 
        found_object= cv2.cvtColor(found_object, cv2.COLOR_BGR2GRAY)
        found_object= found_object.astype(np.uint8)
        #return faux depthmap of image
        depth_heatmap = cv2.applyColorMap(found_object, cv2.COLORMAP_JET)
        return depth_heatmap
    

    global previous_depths
    if found_object is None or found_object.size == 0:
        return None
    #make predictions upon the image
    predictions = pipe(Image.fromarray(cv2.cvtColor(found_object, cv2.COLOR_BGR2RGB)))
    depth_map = np.array(predictions["depth"], dtype=np.float32)
    #get the previous ID
    prev = previous_depths.get(face_id)
    if prev is None or prev.shape != depth_map.shape:
        previous_depths[face_id] = depth_map
    else:
        #if we have a prev frane take the average apply it to current frame for color consitency 
        alpha_smooth = 0.4
        depth_map = alpha_smooth * prev + (1 - alpha_smooth) * depth_map
        previous_depths[face_id] = depth_map
    depth_map = depth_map.astype(np.uint8)
    depth_heat = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    return depth_map, depth_heat
pcd_history = []  # timeline: list of (points, colors) per frame

#if theres no test set up/ generate the point cloud window and outputs
if not test:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("3D Video")
    display_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(display_pcd)

    current_frame = [-1]
    live_mode = [True]    # True = always show newest frame; False = browsing history
    #display current index 
    def show_frame(index):
        if 0 <= index < len(pcd_history):
            points, colors = pcd_history[index]
            display_pcd.points = o3d.utility.Vector3dVector(points)
            display_pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(display_pcd)
            current_frame[0] = index
            print(f"Frame {index + 1}R/{len(pcd_history)}")
    #move foward on timeline
    def next_frame(vis):
        live_mode[0] = False
        show_frame(current_frame[0] + 1)
        return False
    #moveback on timeline
    def prev_frame(vis):
        live_mode[0] = False
        show_frame(current_frame[0] - 1)
        return False
    #resume live showing
    def resume_live(vis):
        live_mode[0] = True
        show_frame(len(pcd_history) - 1)
        return False
    #each key press allows to traverse through timeline
    vis.register_key_callback(ord("D"), next_frame)   # step forward in time
    vis.register_key_callback(ord("A"), prev_frame)   # step backward in time
    vis.register_key_callback(ord("L"), resume_live)  # jump back to "live"


def generate_point_cloud(image, depth):
    #take in rgb image and depth to rgbd image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_raw = o3d.geometry.Image(rgb)
    depth_raw = o3d.geometry.Image(depth.astype(np.float32))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000.0,
        depth_trunc=200.0, convert_rgb_to_intensity=False
    )

    ##force camera intrinsics
    height, width, channels = image.shape
    fx = fy = 1200
    cx, cy = width / 2, height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    #create pc from rgbd image and intrinsic
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic, project_valid_depth_only=False
    )
    #rotate and compress our points 
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    points = np.asarray(pcd.points)
    points[:, 2] *= 0.1

    if test is False:
        # save this frame into the timeline
        pcd_history.append((points, np.asarray(pcd.colors)))

        # only auto-advance the live view if the user hasn't started browsing
        if live_mode[0]:
            show_frame(len(pcd_history) - 1)
            # only frame the camera on the very first point cloud —
            # after that, leave it alone so drag-to-rotate actually works
            if len(pcd_history) == 1:
                vis.reset_view_point(True)

        vis.poll_events()
        vis.update_renderer()

alpha = 0.6
#count and FC will be used to implement frame skipping 
count = 0
fc = 4
cap = cv2.VideoCapture("YTDown.com_Shorts_John-cena-bing-chilling-ORIGINAL-1080p_Media_HWQqabCkAjU_001_480p.mp4")
#this is ratio for final video output fps... it keeps the same relative "fps" vs the available frames
source_fps = cap.get(cv2.CAP_PROP_FPS)/fc
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()



#get resolution    
src_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
src_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
target_max = 360
##if width is bigger go on width else go on height for verticle
if src_width >= src_height:
    scale = target_max / src_width
else:
    scale = target_max / src_height
width = int(src_width * scale)
height = int(src_height * scale)

video = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (width, height))

success = True
while success:
    count+=1
    if count % fc != 0:
        success = cap.grab() # Fetches frame from buffer but DOES NOT decode it
        if not success:
            break
        # Instantly skip to the next loop iteration
        continue 
    success, image = cap.read() # Read frame
    if success:
        if not success:
            break
        #resize image for faster processing
        image = cv2.resize(image,(width, height), interpolation=cv2.INTER_NEAREST)
        #look for faces in resized image and loop through each
        face_locations = face_recognition.face_locations(image)    
        for i,faces in enumerate(face_locations):
            # get face location
            top, right, bottom, left = faces
            face = image[top:bottom, left:right] 
            if not test:
                #apply depth of the found object
                face_depth, face_heat = find_depth(face, i)
                generate_point_cloud(face,face_depth)
            else: face_heat= find_depth(face, i)
            
            if face_heat is not None:
                ##apply depth filter upon the face image
                blended = cv2.addWeighted(face_heat, alpha, image[top:bottom, left:right], 1 - alpha, 0) 
                image[top:bottom, left:right] = blended
        ##write depthmap output to video file for later use
        video.write(image)
        if test:
            cv2.imshow('Depth Feed',image)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):  # Press 'q' to quit
                break
cv2.destroyAllWindows()
video.release()
