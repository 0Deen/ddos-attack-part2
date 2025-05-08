from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
from ryu.lib import hub

import joblib
import numpy as np
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions)

        data = msg.data
        out = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath, priority=priority,
            match=match, instructions=inst)
        datapath.send_msg(mod)

class DDoSDetectionApp(SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(DDoSDetectionApp, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.model = joblib.load('finalized_model.pkl')  # Your trained ensemble model

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
                self.logger.info("Registered datapath: %016x", datapath.id)
        elif ev.state == 'DEAD_DISPATCHER':
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]
                self.logger.info("Unregistered datapath: %016x", datapath.id)

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)

    def _request_stats(self, datapath):
        self.logger.info("Requesting flow stats from: %016x", datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        flows = [flow for flow in body if flow.priority == 1]

        features = []
        for flow in flows:
            features.append([
                flow.match.get('in_port', 0),
                flow.match.get('eth_type', 0),
                flow.match.get('ip_proto', 0),
                flow.match.get('ipv4_src', '0.0.0.0'),
                flow.match.get('ipv4_dst', '0.0.0.0'),
                flow.packet_count,
                flow.byte_count,
                flow.duration_sec
            ])

        if features:
            df = pd.DataFrame(features, columns=[
                'in_port', 'eth_type', 'ip_proto',
                'ipv4_src', 'ipv4_dst',
                'packet_count', 'byte_count', 'duration_sec'
            ])

            X = df[['in_port', 'eth_type', 'ip_proto',
                    'packet_count', 'byte_count', 'duration_sec']].fillna(0)
            try:
                y_pred = self.model.predict(X)
                self.logger.info("Predictions: %s", y_pred.tolist())
                if np.any(y_pred != 0):
                    self.logger.warning("Potential DDoS Attack Detected!")
                    self.analyze_attack(X.values, y_pred)
                    self.send_email_alert(y_pred)
            except Exception as e:
                self.logger.error("Prediction failed: %s", str(e))

    def analyze_attack(self, X, y_pred):
        try:
            attack_details = []
            for i, pred in enumerate(y_pred):
                if pred != 0:
                    proto = int(X[i][2])
                    attack_type = "Unknown"
                    reason = "Unknown"
                    prevention = "Monitor flow thresholds and apply mitigation rules"
                    objective = "General Objective: Ensemble-based detection for IoT"

                    if proto == 1:
                        attack_type = "ICMP Flood"
                        reason = "Overwhelms the target with ICMP Echo requests"
                        prevention = "Rate limiting, ICMP blocking"
                        objective = "Objective ii & iii"

                    elif proto == 6:
                        attack_type = "TCP SYN Flood"
                        reason = "Exploits TCP handshake by sending many SYN packets"
                        prevention = "SYN cookies, connection throttling"
                        objective = "Objective ii & iii"

                    elif proto == 17:
                        attack_type = "UDP Flood"
                        reason = "Sends numerous UDP packets to random ports"
                        prevention = "Block unused ports, UDP rate limiting"
                        objective = "Objective ii, iii & iv"

                    attack_details.append(f"""
                    -----------------------------------
                    Attack #{i+1}
                    Type       : {attack_type}
                    Reason     : {reason}
                    Prevention : {prevention}
                    Objective  : {objective}
                    -----------------------------------
                    """)

            for detail in attack_details:
                self.logger.warning(detail)
        except Exception as e:
            self.logger.error(f"Failed to analyze attack type: {e}")

    def send_email_alert(self, y_pred):
        sender = 'deenramah4@gmail.com'
        receiver = 'deenramaah.com'
        password = '@Taliah66'  # Use app password if needed

        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = 'ALERT: DDoS Attack Detected in SDN Network'

        body = f'DDoS attack detected in SDN controller. Predicted labels: {y_pred.tolist()}'
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            server.quit()
            self.logger.info("Email alert sent.")
        except Exception as e:
            self.logger.error("Failed to send email: %s", str(e))
