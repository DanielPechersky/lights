# import the opencv library
import cv2
import pandas as pd
  
  
# define a video capture object
vid = cv2.VideoCapture(0)

vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
vid.set(cv2.CAP_PROP_BRIGHTNESS, 128)
vid.set(cv2.CAP_PROP_GAIN, 40)
#vid.set(cv2.CAP_PROP_ISO_SPEED, 800)
#vid.set(cv2.CAP_PROP_GAMMA, 128)
#vid.set(cv2.CAP_PROP_BACKLIGHT, 1)
#vid.set(cv2.CAP_PROP_FPS, 2)

vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
#vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
#vid.set(cv2.CAP_PROP_EXPOSURE, -6)
vid.set(cv2.CAP_PROP_FOCUS, 170)
#vid.set(cv2.CAP_PROP_ZOOM, 170)

#vid.set(cv2.CAP_PROP_EXPOSURE_AUTO_PRIORITY, 1)

print(vid.get(cv2.CAP_PROP_EXPOSURE))
print(vid.get(cv2.CAP_PROP_AUTO_EXPOSURE))
#print(vid.get(cv2.CAP_PROP_ISO_SPEED))

print(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
print(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
  
#url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
#table = pd.read_html(url)[0]
#table.columns = table.columns.droplevel()

resolutions = {}

#for index, row in table[['W', 'H']].iterrows():
#    vid.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
#    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
#    print(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#    print(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))


iso = -50

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  

    #vid.set(cv2.CAP_PROP_FOCUS, iso)
    #print(vid.get(cv2.CAP_PROP_GAIN))

    iso = iso + 0.25
    print(iso)

    print(vid.get(cv2.CAP_PROP_ZOOM))
    #print(vid.get(cv2.CAP_PROP_AUTO_EXPOSURE))

    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()