import sys
import logging
import pandas as pd
import numpy as np
import cv2

from cpsdriver.clients import (
    CpsMongoClient,
    CpsApiClient,
    TestCaseClient,
)
from cpsdriver.cli import parse_configs
from cpsdriver.log import setup_logger

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def main(args=None):
    global testName
    args = parse_configs(args)
    setup_logger(args.log_level)
    mongo_client = CpsMongoClient(args.db_address)
    api_client = CpsApiClient()
    test_client = TestCaseClient(mongo_client, api_client)
    testName = f"{args.command}-{args.sample}"
    test_client.load(f"{args.command}-{args.sample}")
    logger.info(f"Available Test Cases are {test_client.available_test_cases}")
    test_client.set_context(args.command, load=False)
    extract_videos(test_client, testName)

def extract_videos(test_client, testName):
    camera_num = 12
    imgSize = None
    fps = 10
    # do we have enough memory for this? hope so!
    frameArrays = [[] for i in range(camera_num)]
    startTimes = [None]*camera_num
    nextData = test_client.find_first_after_time("frame_message", 0.0)
    while len(nextData)> 0:
        frameData = nextData[0]
        logger.debug("Camera {}".format(frameData.camera_id))
        whichCam = frameData.camera_id - 1
        currentTime = frameData.timestamp
        if startTimes[whichCam] is None:
            startTimes[whichCam] = frameData.timestamp
        imgArr = np.fromstring(frameData.frame,np.uint8)
        jpgFrame = cv2.imdecode(imgArr,-1)
        frameArrays[whichCam].append(jpgFrame)
        h,w,l = jpgFrame.shape
        imgSize = (w,h)
        #break
        nextData = test_client.find_first_after_time("frame_message", currentTime)
    for ii in range(camera_num):
        save_path = "/app/video/{}-cam{}.mp4".format(testName, ii)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, imgSize)
        for frame in frameArrays[ii]:
            out.write(frame)
        out.release()

if __name__ == "__main__":
    main(args=sys.argv[1:])
