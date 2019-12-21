_Author_ = "*******************************"
# Web service file of the ArchLearner backend

import tornado.web
from tornado.options import define, options
import tornado.httpserver
import tornado.ioloop
from Initializer import Initialize
from Custom_Logger import logger
import json
from tornado.escape import json_encode

from configparser import ConfigParser

import traceback
import base64

init_object = Initialize()


class PatternChangeService(tornado.web.RequestHandler):
    def post(self):
        # Update the new pattern
        logger.info("Inside Pattern Changer")
        response_json = {}
        response_json["status"] = "failed"
        try:
            json_request_string= self.request.body
            json_object = json.loads(json_request_string)
            pattern = json_object["pattern"]
            print (pattern)
            adaptation_file = open(init_object.adaptation_file, "w")
            string_to_write = "ada " + pattern
            adaptation_file.write(string_to_write)
            adaptation_file.close()
            response_json["status"]= "success"
            self.write(json_encode(response_json))
            #print (pattern)
        except Exception as e:
            self.write(json_encode(response_json))
            logger.error(e)






class Application(tornado.web.Application):
    def __init__(self):
        try:
            handlers = [
                (r"/changePattern",PatternChangeService),
            ]
            tornado.web.Application.__init__(self, handlers)
        except:
            print ("Exception occurred when initializing Application")
            print (traceback.format_exc())

def main():
    try:
        print ("Starting Tornado Web Server on " + str(init_object.port))
        http_server = tornado.httpserver.HTTPServer(Application())
        http_server.listen(init_object.port)
        tornado.ioloop.IOLoop.instance().start()
    except:
        #logger.exception( "Exception occurred when trying to start tornado on " + str(options.port))
        traceback.print_exc()

if __name__ == "__main__":
    main()


