import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.handlers = [] # This is the key thing for the question!

    # Start defining and assigning your handlers here
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    logger.handlers = [handler]
    logger.propagate = False

    return logger