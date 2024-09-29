import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
import bluetooth
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import sys
import os
import requests
import socket
import imageio
import imageio_ffmpeg

kivy.require('2.0.0')

# A flag to control the reading loop
reading = False
data_rows = []  # List to store the data
global saved_image_path 

class MyApp(App):
    def build(self):
        self.title = "Bluetooth Data Reader"
        
        # Main layout
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Text input for displaying temperature and heart rate
        self.data_output = TextInput(readonly=True, multiline=True, size_hint=(1, 0.5))
        self.layout.add_widget(self.data_output)

        # Timer label
        self.timer_label = Label(text="Timer: Not started")
        self.layout.add_widget(self.timer_label)

        # Temperature input
        self.temp_input = TextInput(hint_text="Temperature (°C)", multiline=False)
        self.layout.add_widget(self.temp_input)

        # Heart rate input
        self.heart_input = TextInput(hint_text="Heart Rate (BPM)", multiline=False)
        self.layout.add_widget(self.heart_input)

        # Email input
        self.email_input = TextInput(hint_text="Email", multiline=False)
        self.layout.add_widget(self.email_input)

        # Buttons
        self.start_button = Button(text="Start Reading")
        self.start_button.bind(on_press=self.start_reading)
        self.layout.add_widget(self.start_button)

        self.stop_button = Button(text="Stop Reading")
        self.stop_button.bind(on_press=self.stop_reading)
        self.layout.add_widget(self.stop_button)

        self.save_button = Button(text="Start Prediction")
        self.save_button.bind(on_press=self.run_prediction)
        self.layout.add_widget(self.save_button)

        self.email_button = Button(text="Send Email")
        self.email_button.bind(on_press=self.send_email)
        self.layout.add_widget(self.email_button)

        self.camera_button = Button(text="Capture Image")
        self.camera_button.bind(on_press=self.capture_image)
        self.layout.add_widget(self.camera_button)

        return self.layout
 
    def capture_image(self,*args):
        global saved_image_path
        try:
            # Open the video stream using ffmpeg
            camera = imageio.get_reader('<video0>', format='ffmpeg')  # Use your system's webcam
            frame = camera.get_data(0)  # Capture the first frame

            # Convert to image and save
            image_path = "captured_image.png"
            saved_image_path = image_path
            imageio.imwrite(image_path, frame)
            print(f"Image saved at {image_path}")

        except Exception as e:
            print(f"Error: {e}")


    def read_from_bluetooth(self, dt):
        global reading
        if reading:
            self.fetch_data_from_esp32("D0:EF:76:32:56:B2")


    def fetch_data_from_esp32(self, bt_address):
        try:
            # Create a Bluetooth socket
            sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

            # Connect to the ESP32
            sock.connect((bt_address, 1))  # Channel 1 is commonly used

            # Receive data
            data = sock.recv(1024)  # Buffer size of 1024 bytes
            received_data = data.decode('utf-8').strip()  # Decode and strip whitespace
            print("Received:", received_data)
            data_rows.clear()  # Clear old data
            self.data_output.text = "Temperature (°C)    Heart Rate (BPM)\n" + "-" * 40 + "\n"

            # Parse the received data
            heart_rate, temperature = self.parse_data(received_data)

            # Save to Excel
            self.save_to_excel(heart_rate, temperature)

            # Close the socket
            sock.close()

        except bluetooth.btcommon.BluetoothError as e:
            print(f"An error occurred: {e}")

    def parse_data(self, data):
        """Parse the received data string into heart rate and temperature."""
        try:
            heart_rate_part, temp_part = data.split(" ")
            heart_rate = float(heart_rate_part.split(":")[1])
            temperature = float(temp_part.split(":")[1])
            data_rows.append({"Temperature (°C)": temperature, "Heart Rate (BPM)": heart_rate})
            data = {
            "run_ml": False,
            "sensor_data":data_rows,
                }
            server_url = "http://0.0.0.0:5000/data"
            try:
            # Send a POST request
                response = requests.post(server_url, json=data)
            
                if response.status_code == 200:
                    print("Data sent successfully!")
                else:
                    print(f"Failed to send data: {response.status_code}")

            except Exception as e:
                print(f"Error: {e}")
            return heart_rate, temperature
        except Exception as e:
            print(f"Error parsing data: {e}")
            return None, None
        

    def save_to_excel(self, heart_rate, temperature):
        """Save the heart rate and temperature to an Excel file."""
        if heart_rate is not None and temperature is not None:
            # Create a DataFrame for the new data
            new_data = pd.DataFrame({
                'Heart Rate': [heart_rate],
                'Temperature': [temperature]
            })

            # Try to load existing data, if the file exists
            try:
                # Read existing data
                existing_data = pd.read_excel("temperature_heart_data.xlsx")

                # Concatenate new data to existing data
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            except FileNotFoundError:
                # If file doesn't exist, use new_data as combined_data
                combined_data = new_data

            # Save combined data to Excel file
            combined_data.to_excel("temperature_heart_data.xlsx", index=False)

            print("Data saved to Excel:", heart_rate, temperature)
        else:
            print("No valid data to save.")



    def start_reading(self, instance):
        global reading
        reading = True
        
        # data_rows.app
        Clock.schedule_interval(self.read_from_bluetooth, 1.0)  # Read Bluetooth every second
        Clock.schedule_once(self.stop_reading, 10)  # Stop after 10 seconds

    def stop_reading(self, dt):
        global reading
        reading = False
        

    def send_email(self, instance):
        recipient_email = self.email_input.text
        data={
            'run_ml':False,
            'send_data':True,
            'sensor_data':data_rows,
            'recipient_email':recipient_email
        }
        server_url = "http://0.0.0.0:5000/data"
        try:
            # Send a POST request
                response = requests.post(server_url, json=data)
            
                if response.status_code == 200:
                    print("Data sent successfully!")
                else:
                    print(f"Failed to send data: {response.status_code}")

        except Exception as e:
                print(f"Error: {e}")
        
        

  


    
    
    def run_prediction(self,*args):
        global saved_image_path
        data = {
        "run_ml": True,
        "send_data":False,
        # "sensor_data":data_rows,
        
            }
        server_url = "http://0.0.0.0:5000/data"
        try:
        # Send a POST request
            self.send_image(saved_image_path)
            response = requests.post(server_url, json=data)
        
            if response.status_code == 200:
                print("Data sent successfully!")
            else:
                print(f"Failed to send data: {response.status_code}")

        except Exception as e:
            print(f"Error: {e}")
    
    
    
    def send_image(self,image_path):
        server_url = 'http://127.0.0.1:5000/upload'
        with open(image_path, 'rb') as img_file:
            files = {'file': img_file}
            response = requests.post(server_url, files=files)
            return response.json()

        
      


if __name__ == '__main__':
    MyApp().run()
