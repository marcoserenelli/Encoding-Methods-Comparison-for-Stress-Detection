import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from dotenv import load_dotenv

load_dotenv()


def attach_file(msg, file_path):
    filename = os.path.basename(file_path)
    with open(file_path, 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {filename}',
        )
        msg.attach(part)


def send_email(subject, body, attachment1=None, attachment2=None, attachment3=None):
    email_address = os.getenv('SENDER_EMAIL')
    app_password = os.getenv('SENDER_PASSWORD')
    to_address = [os.getenv('RECIPIENT_EMAIL')]

    msg = MIMEMultipart()
    msg['From'] = email_address
    msg['To'] = ', '.join(to_address)
    msg['Subject'] = subject

    all_recipients = to_address

    msg.attach(MIMEText(body, 'plain'))

    if attachment1 and os.path.exists(attachment1):
        attach_file(msg, attachment1)

    if attachment2 and os.path.exists(attachment2):
        attach_file(msg, attachment2)

    if attachment3 and os.path.exists(attachment3):
        attach_file(msg, attachment3)

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)

    server.login(email_address, app_password)
    server.sendmail(email_address, all_recipients, msg.as_string())
    server.quit()

    print('Email sent successfully!')
