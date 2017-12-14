# Author: Tao Hu <taohu620@gmail.com>
import logger
def print_params(arg_params, aux_params):
    logger.info("arg_params: ")
    logger.info("{:<30}  {:<20}".format("name","shape"))
    for k,v in arg_params.items():
        logger.info("{:<30}  {:<20}".format(k, v.shape))

    logger.info("aux_params: ")
    logger.info("{:<30}  {:<20}".format("name", "shape"))
    for k,v in aux_params.items():
        logger.info("{:<30}  {:<20}".format(k, v.shape))

