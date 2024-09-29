



from flask import Flask, request, jsonify
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import socket
import os
from email import encoders
from email.mime.base import MIMEBase


SAVE_DIRECTORY = "received_images"
global last_received_image_path
app = Flask(__name__)

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()  # Get JSON data from the request
    sensor_data=data.get('sensor_data')
    send_data=data.get('send_data')
    recipient_email=data.get('recipient_email')
    ml=data.get('run_ml')
    df= pd.DataFrame(sensor_data)
    df.to_excel('temperature_heart_data_from_router.xlsx', index=False)
    
    if ml:
            # img_path = 'WhatsApp Image 2021-07-31 at 3.51.20 PM.jpeg'  # Replace with your image path or get it from user input
            predictions = run_prediction()  # Call the prediction function
            print(f"Predictions: {predictions}")  # Log predictions
            
    if send_data:
        send_email(sensor_data,recipient_email)
    
    # Process the received data as needed
    return jsonify({"status": "success"}), 200
def run_prediction():
    global last_received_image_path

    # receive_image(8080)
    from Dr_Tongue.ipynb.Dr_Tongue import predict_image_and_save
    img_path = last_received_image_path  # Replace with your image path or get it from user input
    predictions = predict_image_and_save(img_path)  # Call the prediction function


def send_email(data_rows, recipient_email):
    global last_received_image_path
    image_file_path=last_received_image_path
    if recipient_email and data_rows:
        # Set up the MIME message
        msg = MIMEMultipart()
        msg['From'] = ''
        msg['To'] = recipient_email
        msg['Subject'] = 'Temperature and Heart Rate Data with Attachments'
        
        # Compose email body
        body = f"Good evening doctor,\n\nHere is the data that was procured from the patient:\n\n{'Temperature (°C)':<20} {'Heart Rate (BPM)':<20}\n"
        for row in data_rows:
            body += f"{row['Temperature (°C)']:<20} {row['Heart Rate (BPM)']:<20}\n"
        
        msg.attach(MIMEText(body, 'plain'))
        df=pd.DataFrame(data_rows)
        df.to_excel('temperature_heart_data_from_router.xlsx', index=False)
        excel_file_path = "temperature_heart_data_from_router.xlsx"
        # Attach Excel File
        if os.path.isfile(excel_file_path):
            with open(excel_file_path, 'rb') as file:
                excel_part = MIMEBase('application', 'octet-stream')
                excel_part.set_payload(file.read())
                encoders.encode_base64(excel_part)
                excel_part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(excel_file_path)}')
                msg.attach(excel_part)
        else:
            print(f"Excel file not found: {excel_file_path}")
        excel_file_path = "predictions.xlsx"
        if os.path.isfile(excel_file_path):
            with open(excel_file_path, 'rb') as file:
                excel_part = MIMEBase('application', 'octet-stream')
                excel_part.set_payload(file.read())
                encoders.encode_base64(excel_part)
                excel_part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(excel_file_path)}')
                msg.attach(excel_part)
        else:
            print(f"Excel file not found: {excel_file_path}")

        # Attach Image File
        if os.path.isfile(image_file_path):
            with open(image_file_path, 'rb') as file:
                image_part = MIMEBase('application', 'octet-stream')
                image_part.set_payload(file.read())
                encoders.encode_base64(image_part)
                image_part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_file_path)}')
                msg.attach(image_part)
        else:
            print(f"Image file not found: {image_file_path}")

        # Send the email
        try:
            smtp_server = 'smtp.gmail.com'
            smtp_port = 587
            username = ''
            password = ''  # Use a secure method to handle passwords
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")

                


@app.route('/upload', methods=['POST'])
def upload_image():
    global last_received_image_path
    print("Incoming")
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    file_path = os.path.join(SAVE_DIRECTORY, file.filename)
    last_received_image_path=file_path
    file.save(file_path)

    return jsonify({"status": "success", "message": "File uploaded", "file_path": file_path}), 200
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Listen on all interfaces on port 5000


