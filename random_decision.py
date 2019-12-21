_Author_="*********"

import random
import time

def gen_decision():
    # Take the path file and write the text

    while(1):
        file_writer = open("adaptation.txt","w")

        decision_list = [10,20,30]
        rand_val = random.randint(0,2)
        print (rand_val)
        file_writer.write("ada" + " " + str(decision_list[rand_val]))

        time.sleep(120)



if __name__ == '__main__':
    gen_decision()