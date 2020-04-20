import sys
import logging
import pandas as pd
import numpy as np

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
    generate_receipts(test_client, testName)

def plateIdToStr(plate_id):
    idStr = "gondola_{0.gondola_id}_shelf_{0.shelf_index}_plate_{0.plate_index}".format(plate_id)
    return idStr

def generate_receipts(test_client, testName):

    didWrite = False
    dT = 1.0/60
    obj_type = "plate_data"
    csvFile = "/app/csv/{}-weight.csv".format(testName)
    nextData = test_client.find_first_after_time(obj_type, 0.0)
    logger.info('Extracting weight sensor data from {}'.format(testName))
    while len(nextData) > 0:
        rawData = nextData[0] 
        currentTime = rawData.timestamp
        startShelf = rawData.plate_id.shelf_index
        startPlate = rawData.plate_id.plate_index
        gondolaId = rawData.plate_id.gondola_id
        dataSize = rawData.data.shape
        logger.debug("Data looks like: {}".format(dataSize))
        logger.debug("Gondola: {}".format(gondolaId))

        nSamples = dataSize[0]
        nShelves = dataSize[1]
        nPlates = dataSize[2]
        ts = np.array(range(nSamples))*dT + currentTime # the timestamps in this packet
        ts = ts.reshape((nSamples,1))
        for jj in range(nShelves):
            for ii in range(nPlates):
                weightData = (rawData.data[:,jj,ii]).reshape(nSamples,1)
                if not(np.isnan(weightData).all()):
                    logger.debug("Gondola {}, shelf {}, plate {}".format(gondolaId, jj,ii))
                    plateLoc = np.tile(np.array([gondolaId, jj, ii]), (nSamples,1))
                    #TODO: make a big DataFrame then write to file?
                    dataFrame = pd.DataFrame(np.hstack([ts, weightData, plateLoc]), index=None, columns=["timestamp","reading", "gondola", "shelf", "plate"])
                    if not didWrite:
                        file_mode='w'
                        do_header = True
                    else:
                        file_mode='a'
                        do_header = False
                    dataFrame.to_csv(csvFile, mode=file_mode,header=do_header, index=False)
                    didWrite=True

            
        #break
        nextData = test_client.find_first_after_time(obj_type, currentTime)

if __name__ == "__main__":
    main(args=sys.argv[1:])
