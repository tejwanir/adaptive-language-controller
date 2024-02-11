import serial
from datetime import datetime
import csv
import time
import keyboard

#Open a csv file and set it up to receive comma delimited input
session_name = 'test_session_17'
logging = open('sessions/'+session_name+'/data.csv', mode='a')
#logging = open('rich1.csv',mode='a')
writer = csv.writer(logging, delimiter=",", escapechar=' ', quoting=csv.QUOTE_NONE)

#Open a serial port that is connected to an Arduino (below is Linux, Windows and Mac would be "COM4" or similar)
#No timeout specified; program will wait until all serial data is received from Arduino
#Port description will vary according to operating system. Linux will be in the form /dev/ttyXXXX
#Windows and MAC will be COMX. Use Arduino IDE to find out name 'Tools -> Port'
ser = serial.Serial("COM5")
ser.flushInput()

#Write out a single character encoded in utf-8; this is defalt encoding for Arduino serial comms
#This character tells the Arduino to start sending data
ser.write(bytes('x', 'utf-8'))

writer.writerow(['workings'])

start_time = time.time()  # Record the start time
            

def read_data(ser, writer):
    writer.writerow(['timestamp', 'reading'])

    while True:

        if keyboard.is_pressed('esc'):
            print("Escape key pressed, stopping...")
            break  # Exit the loop

        # Read data from Serial until \n (new line) received
        ser_bytes = ser.readline()

        print(ser_bytes)

        # Convert received bytes to text format
        decoded_bytes = ser_bytes[0:len(ser_bytes)-2].decode("utf-8")
        print(decoded_bytes)

        # Retrieve current time
        c = datetime.now()
        current_time = c.strftime('%H:%M:%S')
        print(current_time)

        # If Arduino has sent a string "stop", exit loop
        if decoded_bytes == "stop":
            break

        # Write received data to CSV file
        writer.writerow([time.time(), decoded_bytes])

read_data(ser,writer)

# Close port and CSV file to exit
ser.close()
logging.close()
print("logging finished")

	
