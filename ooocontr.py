'''
Enhanced Random Forest SDN Controller for DDoS Detection and Classification

Required Python modules:
  - ryu
  - pandas
  - scikit-learn
  - numpy

Install via:
  pip install ryu pandas scikit-learn numpy
'''

from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch  # inherits SimpleSwitch13 behavior
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

class EnhancedMonitor13(switch.SimpleSwitch13):
    """
    Ryu app that:
      - trains a multi-class Random Forest on FlowStatsfile.csv
      - monitors flow stats and classifies incoming traffic
      - logs legitimacy or specific DDoS attack types, reasons, and prevention advice
    """
    def __init__(self, *args, **kwargs):
        super(EnhancedMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()
        self.flow_model = None
        self._prepare_and_train()
        end = datetime.now()
        self.logger.info(f"Model training time: {end - start}")

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug(f"Register datapath: {datapath.id}")
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug(f"Unregister datapath: {datapath.id}")
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)
            hub.sleep(10)
            self._predict_and_report()

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        # write latest stats to PredictFlowStatsfile.csv for prediction
        timestamp = datetime.now().timestamp()
        with open("PredictFlowStatsfile.csv", "w") as fw:
            header = ("timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,"
                      "ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,"
                      "idle_timeout,hard_timeout,flags,packet_count,byte_count,"
                      "packet_count_per_second,packet_count_per_nsecond,"
                      "byte_count_per_second,byte_count_per_nsecond")
            fw.write(header + "\n")

            body = ev.msg.body
            for flow in sorted([f for f in body if f.priority == 1],
                                key=lambda f: (f.match['eth_type'], f.match['ipv4_src'], f.match['ipv4_dst'], f.match['ip_proto'])):
                record = self._build_flow_record(flow, ev.msg.datapath.id, timestamp)
                fw.write(record + "\n")

    def _build_flow_record(self, stat, dpid, timestamp):
        # extract fields with safe defaults
        ip_proto = stat.match.get('ip_proto', -1)
        icmp_code = stat.match.get('icmpv4_code', -1) if ip_proto == 1 else -1
        icmp_type = stat.match.get('icmpv4_type', -1) if ip_proto == 1 else -1
        tp_src = stat.match.get('tcp_src', stat.match.get('udp_src', 0))
        tp_dst = stat.match.get('tcp_dst', stat.match.get('udp_dst', 0))
        ip_src = stat.match.get('ipv4_src', '0.0.0.0')
        ip_dst = stat.match.get('ipv4_dst', '0.0.0.0')

        flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"
        # avoid division by zero
        dur_sec = stat.duration_sec or 1
        dur_nsec = stat.duration_nsec or 1
        pkt_per_s = stat.packet_count / dur_sec
        pkt_per_ns = stat.packet_count / dur_nsec
        byte_per_s = stat.byte_count / dur_sec
        byte_per_ns = stat.byte_count / dur_nsec

        fields = [timestamp, dpid, flow_id, ip_src, tp_src, ip_dst, tp_dst,
                  ip_proto, icmp_code, icmp_type, stat.duration_sec,
                  stat.duration_nsec, stat.idle_timeout, stat.hard_timeout,
                  stat.flags, stat.packet_count, stat.byte_count,
                  pkt_per_s, pkt_per_ns, byte_per_s, byte_per_ns]
        return ",".join(map(str, fields))

    def _prepare_and_train(self):
        # Load and train on historic flow stats
        try:
            df = pd.read_csv('FlowStatsfile.csv')
        except Exception as e:
            self.logger.error("Cannot read FlowStatsfile.csv: %s", e)
            return

        # Clean string columns
        for col in ['flow_id', 'ip_src', 'ip_dst']:
            df[col] = df[col].astype(str).str.replace('.', '', regex=False)

        # Features and labels
        X = df.iloc[:, :-1].values.astype('float64')
        y = df.iloc[:, -1].values.astype('int')

        if len(y) == 0:
            self.logger.error("No training data found. Collect flow stats first.")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0)

        clf = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
        self.flow_model = clf.fit(X_train, y_train)

        preds = self.flow_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        acc = accuracy_score(y_test, preds)

        self.logger.info("Training complete. Accuracy: %.2f%%", acc * 100)
        self.logger.info("Confusion Matrix:\n%s", cm)

    def _predict_and_report(self):
        if self.flow_model is None:
            return
        try:
            df = pd.read_csv('PredictFlowStatsfile.csv')
        except FileNotFoundError:
            return
        if df.empty or df.shape[0] == 1:
            return

        for col in ['flow_id', 'ip_src', 'ip_dst']:
            df[col] = df[col].astype(str).str.replace('.', '', regex=False)

        Xp = df.values.astype('float64')
        preds = self.flow_model.predict(Xp)

        legit, attack_count = 0, {}
        for label in preds:
            if label == 0:
                legit += 1
            else:
                attack_count[label] = attack_count.get(label, 0) + 1

        total = len(preds)
        self.logger.info('-' * 60)
        if legit / total >= 0.8:
            self.logger.info("Traffic is legitimate (%d/%d)", legit, total)
        else:
            self.logger.info("DDoS Traffic Detected (%d/%d)", total - legit, total)
            for lbl, cnt in attack_count.items():
                atype, reason, advice = self._attack_info(lbl)
                self.logger.info("Type: %s (%d flows)", atype, cnt)
                self.logger.info("Cause: %s", reason)
                self.logger.info("Prevention: %s", advice)
        self.logger.info('-' * 60)

        # reset prediction file
        open('PredictFlowStatsfile.csv', 'w').close()

    def _attack_info(self, label):
        # Returns (attack_type, cause, prevention advice)
        if label == 1:
            return (
                "ICMP Flood",
                "Many ICMP echo requests overwhelming the target.",
                "Implement ICMP rate limiting and ping filters at the firewall."
            )
        if label == 2:
            return (
                "TCP SYN Flood",
                "Excessive SYN packets to exhaust server connection table.",
                "Enable SYN cookies, increase backlog, and use stateful firewalls."
            )
        if label == 3:
            return (
                "UDP Flood",
                "High-volume UDP packets saturating bandwidth.",
                "Drop or rate-limit UDP traffic from untrusted sources."
            )
        return (
            "Unknown Attack",
            "Unrecognized anomalous traffic pattern.",
            "Apply general anomaly detection and rate-limiting."
        )
