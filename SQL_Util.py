_Author_ = "*********"
from  Custom_Logger import logger
from configparser import ConfigParser
import mysql.connector

CONFIGURATION_FILE = "settings.conf"
CONFIG_SECTION = "database"

parser = ConfigParser()
parser.read(CONFIGURATION_FILE)


# Module for inserting and retreving data to and from MySQL
class SQLUtils:
    host = ""
    database =""
    user = ""
    password = ""
    def __init__(self):
        # Initiaizing the SQL Utils
        logger.info("Initializing SQL Class")
        parser = ConfigParser()
        parser.read(CONFIGURATION_FILE)
        self.host = parser.get(CONFIG_SECTION,"host")
        self.user = parser.get(CONFIG_SECTION, 'user')
        self.database = parser.get(CONFIG_SECTION,"database")
        self.password = parser.get(CONFIG_SECTION,"password")


    def insert(self,query):
        # Insert data based on the query
        logger.info("inserting query " + query)
        mydb = mysql.connector.connect(
            host=self.host,
            user=self.user,
            passwd=self.password,
            database='sys',
            auth_plugin='mysql_native_password'
        )
        mycursor = mydb.cursor()
        print (query)
        mycursor.execute(query)
        mydb.commit()
        mydb.close()

    def query_table(self,query):
        # Query the table to generate the results
        logger.info("Inside the query table call")
        mydb = mysql.connector.connect(
            host=self.host,
            user=self.user,
            passwd=self.password,
            database='sys',
            auth_plugin='mysql_native_password'
        )

        cursor  = mydb.cursor()
        cursor.execute(query)
        data_dict = {}
        for (sensorId, timeVal, dataval) in cursor:
            #print (reviewId + "  "  + review_text)
            data_dict[timeVal] = dataval
        cursor.close()
        mydb.close()
        return data_dict



