# coding = utf8
# ========================================================
#   Copyright (C) 2016 All rights reserved.
#   
#   filename : logMod.py
#   desc     : Friendly Logger Module
# ======================================================== 

import os
import sys
import time
import logging 

# CONFIG SET
path_log = "./log/Log.txt"

class logMod:

    Log = path_log
    LastTime = 0
    LastDate = "Last Recorded Date"
    CurrentDate = "Current Date"

    logger=logging.getLogger()
    handler=logging.FileHandler(Log)
    logger.addHandler(handler)
    logger.setLevel(logging.NOTSET)

    reload(sys)
    sys.setdefaultencoding('utf-8')

    def __init__(self):
        self.LastTime = time.localtime(time.time())
    
    def getTime(self):
        return str(time.strftime('%m-%d %H:%M:%S',time.localtime(time.time())))
    
    def Warning(self, tstring):
        self.logger.info("%s\t[Warning]\t%s" % (self.getTime(), str(tstring)))
        
    def Fatal(self, tstring):
        self.logger.info("%s\t[Fatal]\t%s" % (self.getTime(), str(tstring)))
        
    def Notice(self, tstring):
        self.logger.info("%s\t[Notice]\t%s" % (self.getTime(), str(tstring)))
    
    def echo(self, tstring):
        self.logger.info("%s" % str(tstring))
    
    def custom(self, mark, tstring):
        self.logger.info("%s\t[%s]\t%s" % (self.getTime(), str(mark), str(tstring)))
    
    def CheckLog(self):
        if os.path.exists(Log):
            return True
        else :
            return False