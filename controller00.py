from pox.core import core
import pox.openflow.libopenflow_01 as of
import json
import smtplib
from email.mime.text import MIMEText
import os
import sys

sys.path.append(os.path.abspath('utils'))
from test_model import predict_packet_type  # Your ML-based classification logic

log = core.getLogger()

class DDoSDetectionController(object):
    def __init__(self, connection):
        self.connection = connection
        connection.addListeners(self)
        log.info("Controller initialized for switch %s", connection)

    def send_email_alert(self, attack_type, src_ip, dst_ip):
        try:
            sender = "youralert@email.com"
            receiver = "admin@example.com"
            msg = MIMEText(f"DDoS Alert!\n\nType: {attack_type}\nFrom: {src_ip}\nTo: {dst_ip}")
            msg['Subject'] = f"DDoS Detected: {attack_type}"
            msg['From'] = sender
            msg['To'] = receiver

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender, "yourpassword")
                server.sendmail(sender, receiver, msg.as_string())
            log.info("Alert email sent for attack %s", attack_type)
        except Exception as e:
            log.error("Email failed: %s", e)

    def _handle_PacketIn(self, event):
        packet = event.parsed

        if not packet.parsed:
            log.warning("Ignoring incomplete packet")
            return

        src_ip = str(packet.next.src)
        dst_ip = str(packet.next.dst)
        pkt_features = {
            "src": src_ip,
            "dst": dst_ip,
            "protocol": str(packet.next.__class__.__name__),
            "length": len(packet)
        }

        attack_type = predict_packet_type(pkt_features)

        if attack_type != "legitimate":
            log.warning("Potential attack detected: %s from %s", attack_type, src_ip)
            self.send_email_alert(attack_type, src_ip, dst_ip)

            # Drop the packet
            msg = of.ofp_flow_mod()
            msg.match = of.ofp_match.from_packet(packet, event.port)
            msg.idle_timeout = 10
            msg.hard_timeout = 30
            msg.priority = 1000
            msg.actions = []  # No actions = drop
            self.connection.send(msg)
            return

        # Normal forwarding
        msg = of.ofp_packet_out()
        msg.data = event.ofp
        msg.actions.append(of.ofp_action_output(port=of.OFPP_FLOOD))
        msg.in_port = event.port
        self.connection.send(msg)

def launch():
    def start_switch(event):
        log.info("Connection from %s", event.connection)
        DDoSDetectionController(event.connection)
    core.openflow.addListenerByName("ConnectionUp", start_switch)
