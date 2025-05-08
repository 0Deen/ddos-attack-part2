from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
import switch
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import logging
import os

class SimpleMonitor13(switch.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.flow_stats = []
        self.logger.info("🚀 Starting SimpleMonitor13 for DDoS Detection")
        self._print_objectives()
        self.monitor_thread = hub.spawn(self._monitor)
        self.flow_model = None
        self.flow_training()

    def _print_objectives(self):
        self.logger.info("🎯 General Objective: Build an Ensemble-Based DDoS Detection System")
        self.logger.info("🔹 i. Identify shortcomings of existing systems.")
        self.logger.info("🔹 ii. Integrate Ensemble ML Techniques.")
        self.logger.info("🔹 iii. Assess Detection & Mitigation Strategies.")
        self.logger.info("🔹 iv. Test on real-time SDN flows.")

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('Register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('Unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.info("📡 Requesting flow stats from switch %016x", datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        file_path = "PredictFlowStatsfile.csv"
        with open(file_path, "w") as file0:
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
            body = ev.msg.body

            for stat in sorted([flow for flow in body if flow.priority == 1],
                               key=lambda flow: (flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'])):

                ip_src = stat.match['ipv4_src']
                ip_dst = stat.match['ipv4_dst']
                ip_proto = stat.match['ip_proto']
                icmp_code = -1
                icmp_type = -1
                tp_src = 0
                tp_dst = 0

                if ip_proto == 1:
                    icmp_code = stat.match.get('icmpv4_code', -1)
                    icmp_type = stat.match.get('icmpv4_type', -1)
                elif ip_proto == 6:
                    tp_src = stat.match.get('tcp_src', 0)
                    tp_dst = stat.match.get('tcp_dst', 0)
                elif ip_proto == 17:
                    tp_src = stat.match.get('udp_src', 0)
                    tp_dst = stat.match.get('udp_dst', 0)

                flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"

                try:
                    pps = stat.packet_count / stat.duration_sec
                except:
                    pps = 0
                try:
                    ppns = stat.packet_count / stat.duration_nsec
                except:
                    ppns = 0
                try:
                    bps = stat.byte_count / stat.duration_sec
                except:
                    bps = 0
                try:
                    bpns = stat.byte_count / stat.duration_nsec
                except:
                    bpns = 0

                file0.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},{ip_proto},{icmp_code},{icmp_type},{stat.duration_sec},{stat.duration_nsec},{stat.idle_timeout},{stat.hard_timeout},{stat.flags},{stat.packet_count},{stat.byte_count},{pps},{ppns},{bps},{bpns}\n")

    def flow_training(self):
        self.logger.info("Flow Training ...")
        parquet_files = [f for f in os.listdir() if f.endswith(".parquet")]
        df_list = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                df_list.append(df)
                self.logger.info(f"Loaded: {file} - {df.shape}")
            except Exception as e:
                self.logger.error(f"Failed to load {file}: {e}")
        if not df_list:
            self.logger.error("No valid parquet data loaded.")
            return

        flow_dataset = pd.concat(df_list, ignore_index=True)
        try:
            flow_dataset.iloc[:, 2] = flow_dataset.iloc[:, 2].astype(str).str.replace('.', '')
            flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].astype(str).str.replace('.', '')
            flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].astype(str).str.replace('.', '')
        except Exception as e:
            self.logger.warning(f"IP column cleanup skipped or failed: {e}")

        X = flow_dataset.iloc[:, :-1].values.astype('float64')
        y = LabelEncoder().fit_transform(flow_dataset.iloc[:, -1].values)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        svc = SVC(kernel='linear', probability=True, random_state=42)
        lr = LogisticRegression(random_state=42)

        self.flow_model = VotingClassifier(estimators=[('rf', rf), ('svc', svc), ('lr', lr)], voting='hard')
        self.flow_model.fit(X_train, y_train)

        y_pred = self.flow_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        self.logger.info("------------------------------------------------------------------------------")
        self.logger.info(f"✅ Ensemble model accuracy: {acc:.2f}")
        self.logger.info(f"🧮 Confusion matrix:\n{cm}")
        self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        try:
            df = pd.read_csv('PredictFlowStatsfile.csv')
            df.iloc[:, 2] = df.iloc[:, 2].astype(str).str.replace('.', '')
            df.iloc[:, 3] = df.iloc[:, 3].astype(str).str.replace('.', '')
            df.iloc[:, 5] = df.iloc[:, 5].astype(str).str.replace('.', '')
            X_predict_flow = df.iloc[:, :].values.astype('float64')
            y_flow_pred = self.flow_model.predict(X_predict_flow)
            legitimate_traffic = (y_flow_pred == 0).sum()
            ddos_traffic = (y_flow_pred != 0).sum()
            self.logger.info("------------------------------------------------------------------------------")
            if (legitimate_traffic / len(y_flow_pred)) * 100 > 80:
                self.logger.info("✅ Legitimate traffic ...")
            else:
                self.logger.warning("🚨 DDoS Attack Detected!")
                victim = int(df.iloc[0, 5]) % 20
                self.logger.warning("Victim is likely host: h%d", victim)
            self.logger.info("------------------------------------------------------------------------------")
            with open("PredictFlowStatsfile.csv", "w") as file0:
                file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
        except Exception as e:
            self.logger.error(f"Flow prediction failed: {e}")
