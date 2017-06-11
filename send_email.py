# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:59:08 2017

@author: memedai

@target:send email
"""
import smtplib
import mimetypes

from datetime import date
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.header import Header
from email.mime.text import MIMEText
from email.mime.audio import MIMEAudio
from email.mime.image import MIMEImage

class MailForWarning():
    def __init__(self,from_addr,password,smtp_server,to_addr,subject):
        self.from_addr = from_addr
        self.password = password
        self.smtp_server = smtp_server
        self.to_addr = to_addr
        self.subject = subject
        self.msg = None
    def mailText(self,text,textType,appendix=None):
        self.msg = MIMEMultipart('all')
        self.msg.attach(MIMEText(text, textType, 'utf-8'))
        if appendix:
            ctype, encoding = mimetypes.guess_type(appendix)
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            if maintype == "text":
                fp = open(appendix)
                # Note: we should handle calculating the charset
                attachment = MIMEText(fp.read(), _subtype=subtype)
                fp.close()
            elif maintype == "image":
                fp = open(appendix, "rb")
                attachment = MIMEImage(fp.read(), _subtype=subtype)
                fp.close()
            elif maintype == "audio":
                fp = open(appendix, "rb")
                attachment = MIMEAudio(fp.read(), _subtype=subtype)
                fp.close()
            else:
                fp = open(appendix, "rb")
                attachment = MIMEBase(maintype, subtype)
                attachment.set_payload(fp.read())
                fp.close()
                encoders.encode_base64(attachment)
            attachment.add_header("Content-Disposition", "attachment", filename=appendix)
            self.msg.attach(attachment)
    def mailHeader(self):
        today = date.today().isoformat()
        self.msg['Subject'] = Header(today+'_'+self.subject, 'utf-8')
        self.msg['From'] = (u'发件人 <%s>' % self.from_addr)
        self.msg['To'] = (u'收件人 <%s>' % self.to_addr)
        server = smtplib.SMTP(self.smtp_server, 25) # SMTP协议默认端口是25
        server.set_debuglevel(0)
        server.login(self.from_addr, self.password)
        server.sendmail(self.from_addr, [self.to_addr], self.msg.as_string())
        server.quit()





