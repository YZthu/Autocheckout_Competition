import sys
import logging
import pandas as pd

from cpsdriver.clients import (
    CpsMongoClient,
    CpsApiClient,
    TestCaseClient,
)
from cpsdriver.cli import parse_configs
from cpsdriver.log import setup_logger


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def main(args=None):
    args = parse_configs(args)
    setup_logger(args.log_level)
    mongo_client = CpsMongoClient(args.db_address)
    api_client = CpsApiClient()
    test_client = TestCaseClient(mongo_client, api_client)
    testName = f"{args.command}-{args.sample}"
    test_client.load(f"{args.command}-{args.sample}")
    logger.info(f"Available Test Cases are {test_client.available_test_cases}")
    test_client.set_context(args.command, load=False)
    extract_product_data(test_client, testName)

def extract_product_data(test_client,testName):
    allProducts = test_client.list_products()

    didWrite = False
    csvFile = "/app/csv/{}-product.csv".format(testName)
    productData = pd.DataFrame(columns=['Name', 'Gondola', 'Shelf', 'Plate', 'Weight', 'Price', 'Image'])
    logger.info('Extracting product information from {}'.format(testName))
    for aProduct in allProducts:
        baseDict = {'Name':aProduct.name, 'Image':aProduct.thumbnail, 'Price':aProduct.price,
            'Weight':aProduct.weight}
        allFacings = test_client.find_product_facings(aProduct.product_id)
        logger.debug(aProduct.name)
        if len(allFacings) == 0:
            locDict = {'Gondola':-1, 'Shelf':-1,'Plate':-1}
            locDict.update(baseDict)
            productData = productData.append(locDict, ignore_index=True)
        else:
            for aFacing in allFacings:
                for plateLoc in aFacing.plate_ids:
                    locDict = {'Gondola':plateLoc.gondola_id, 'Shelf':plateLoc.shelf_index,
                        'Plate':plateLoc.plate_index}
                    locDict.update(baseDict)
                    productData = productData.append(locDict, ignore_index=True)

    productData.to_csv(csvFile, mode='w', header=True, index=False)
    logger.info("Product extraction complete")
                    
if __name__ == "__main__":
    main(args=sys.argv[1:])
