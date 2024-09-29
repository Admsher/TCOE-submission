import tkinter as tk
from tkinter import ttk
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import serial
import threading
import cv2  # OpenCV for camera functionality
import pandas as pd


# Setup the serial port connection to Arduino with a baud rate of 57600
ser = serial.Serial('/dev/ttyUSB2', 9600, timeout=1)

# A flag to control the reading loop
reading = False
timer_label = None
data_rows = []  # List to store the data

# Function to read and display data from serial
def read_from_serial():
    global reading
    while reading:
        # print(ser.in_waiting)
      
        if ser.in_waiting > 0:
            try:
                # Decode the incoming serial data
                line = ser.readline().decode('utf-8', errors='ignore').rstrip()
                # print(line)
                
                # Simulating data parsing for temperature and heart rate from the serial input
                if "Temp:" in line and "Heart:" in line:
                    heart_data=line.split(":")[1]
                    temp_data=(line.split(":")[2])
                    
                    
                    # Update temperature and heart rate fields
                    temp_entry.delete(0, tk.END)
                    temp_entry.insert(0, temp_data)
                    
                    heart_entry.delete(0, tk.END)
                    heart_entry.insert(0, heart_data)

                    # Append the data in tabular format to the Text widget
                    data_text.insert(tk.END, f"{temp_data:<20} {heart_data:<20}\n")
                    data_text.see(tk.END)  # Automatically scroll to the bottom

                    # Save the data row for later use
                    data_rows.append({"Temperature (°C)": temp_data, "Heart Rate (BPM)": heart_data})
                    pd.DataFrame(data_rows).to_excel('temperature_heart_data.xlsx', index=False)
            except UnicodeDecodeError:
                print("Could not decode data")

def start_reading():
    global reading, data_rows
    reading = True
    data_rows.clear()  # Clear old data

    # Clear the data_text widget
    data_text.delete(1.0, tk.END)
    # Add table headers
    data_text.insert(tk.END, f"{'Temperature (°C)':<20} {'Heart Rate (BPM)':<20}\n")
    data_text.insert(tk.END, f"{'-'*20} {'-'*20}\n")
    
    # Start a thread to continuously read data from the serial port
    thread = threading.Thread(target=read_from_serial)
    thread.daemon = True  # Ensure the thread stops when the app closes
    thread.start()

    # Start a timer to stop reading after 10 seconds
    countdown(10)

def stop_reading():
    global reading
    reading = False  # Set the flag to False to stop the loop

def countdown(seconds):
    if seconds > 0:
        # Update the label to show the countdown
        timer_label.config(text=f"Time Remaining: {seconds} seconds")
        # Call countdown again after 1 second
        root.after(1000, countdown, seconds - 1)
    else:
        stop_reading()  # Stop reading after 10 seconds
        timer_label.config(text="Stopped after 10 seconds")

# Function to save the data to an Excel file
def save_to_excel():
    if data_rows:
        df = pd.DataFrame(data_rows)
        df.to_excel('temperature_heart_data.xlsx', index=False)
        status_label.config(text="Data saved to Excel.")

# Function to send data via email
def send_email():
    recipient_email = email_entry.get()
    print("Sending")
    if recipient_email and data_rows:
        # Compose the email
        msg = MIMEMultipart()
        msg['From'] = ''  # Replace with your email
        msg['To'] = recipient_email
        msg['Subject'] = 'Temperature and Heart Rate Data'
        

        # Attach the data as plain text in the email body
        body = f"Good evening doctor,\n\n Here is the data that was procured from the patient:\n\n{'Temperature (°C)':<20} {'Heart Rate (BPM)':<20}\n The contact for them is given here: 878543219 \n ,Thanks and Regards"
        body += f"{'-'*20} {'-'*20}\n"
        for row in data_rows:
           body += f"{row['Temperature (°C)']:<20} {row['Heart Rate (BPM)']:<20}\n"

        msg.attach(MIMEText(body, 'plain'))

        # Send the email
        try:
            # print("Sending")
            smtp_server = 'smtp.gmail.com'  # For Gmail
            smtp_port = 587  # For TLS
            username = ''  # Your email
            password = ''  # Your email password or app password

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
                server.login(username, password)
                server.send_message(msg)

            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")

# Function to open camera using OpenCV
def open_camera():
    cap = cv2.VideoCapture(0)  # Use default camera
    if not cap.isOpened():
        status_label.config(text="Could not open camera")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            status_label.config(text="Failed to grab frame")
            break

        cv2.imshow("Camera Feed", frame)

        # Break loop with 'q' key or when the window is closed
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Camera Feed", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    status_label.config(text="Camera closed")
    


def run_prediction():
    from Dr_Tongue.ipynb.Dr_Tongue import predict_image_and_save

    
    img_path = 'WhatsApp Image 2021-07-31 at 3.51.20 PM.jpeg'  # Replace with your image path or get it from user input
    predictions = predict_image_and_save(img_path)  # Call the prediction function
    # result_text = '\n'.join([f"{label}: {score:.4f}" for label, score in predictions.items()])
    # status_label.config(text=result_text)  # Update the status label with results





# Setup the tkinter window
root = tk.Tk()
root.title("Arduino Serial Reader")

# Create the main frame
main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Create a Text widget to display data from the Arduino in a tabular format
data_text = tk.Text(main_frame, height=15, width=50, font=('Courier', 10))
data_text.grid(row=0, column=0, padx=10, pady=10)

# Create a Scrollbar for the Text widget
scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=data_text.yview)
scrollbar.grid(row=0, column=1, sticky='ns')
data_text['yscrollcommand'] = scrollbar.set

# Create a Timer label
timer_label = ttk.Label(main_frame, text="Timer: Not started")
timer_label.grid(row=1, column=0, pady=10)

# Create labels and entry fields for Temperature and Heart Rate
temp_label = ttk.Label(main_frame, text="Temperature (°C):")
temp_label.grid(row=2, column=0, sticky=tk.W, padx=10)

temp_entry = ttk.Entry(main_frame, width=20)
temp_entry.grid(row=2, column=0, padx=150, pady=5)

heart_label = ttk.Label(main_frame, text="Heart Rate (BPM):")
heart_label.grid(row=3, column=0, sticky=tk.W, padx=10)

heart_entry = ttk.Entry(main_frame, width=20)
heart_entry.grid(row=3, column=0, padx=150, pady=5)

# Email label and entry
email_label = ttk.Label(main_frame, text="Email:")
email_label.grid(row=4, column=0, sticky=tk.W, padx=10)

email_entry = ttk.Entry(main_frame, width=40)
email_entry.grid(row=4, column=0, padx=150, pady=5)

# Create a Start button to initiate serial communication
start_button = ttk.Button(main_frame, text="Start Reading", command=start_reading)
start_button.grid(row=5, column=0, pady=10)

# Create a Stop button to stop serial communication
stop_button = ttk.Button(main_frame, text="Stop Reading", command=stop_reading)
stop_button.grid(row=5, column=1, pady=10)

# Create a Save to Excel button
save_button = ttk.Button(main_frame, text="Save to Excel", command=save_to_excel)
save_button.grid(row=6, column=0, pady=10)

# Create a Send Email button
email_button = ttk.Button(main_frame, text="Send Email", command=send_email)
email_button.grid(row=6, column=1, pady=10)

# Create a button to open the camera
camera_button = ttk.Button(main_frame, text="Open Camera", command=open_camera)
camera_button.grid(row=7, column=0, pady=10)


camera_button = ttk.Button(main_frame, text="Predict", command=run_prediction)
camera_button.grid(row=8, column=0, pady=10)

# Status label to show messages
status_label = ttk.Label(main_frame, text="")
status_label.grid(row=8, column=0, columnspan=2, pady=10)

# Run the main event loop
root.mainloop()
