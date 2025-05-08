from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
import switch
from datetime import datetime
import os
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import smtplib
from email.message import EmailMessage

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end - start))

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        with open("PredictFlowStatsfile.csv", "w") as file0:
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

        X_flow = flow_dataset.iloc[:, :-1].values.astype('float64')
        y_flow = flow_dataset.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        self.flow_model = classifier.fit(X_train, y_train)

        y_pred = self.flow_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        self.logger.info("------------------------------------------------------------------------------")
        self.logger.info("Confusion Matrix:\n%s", cm)
        self.logger.info("Success accuracy = %.2f %%", acc * 100)
        self.logger.info("Fail accuracy = %.2f %%", (1.0 - acc) * 100)
        self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        try:
            df = pd.read_csv('PredictFlowStatsfile.csv')
            df.iloc[:, 2] = df.iloc[:, 2].astype(str).str.replace('.', '')
            df.iloc[:, 3] = df.iloc[:, 3].astype(str).str.replace('.', '')
            df.iloc[:, 5] = df.iloc[:, 5].astype(str).str.replace('.', '')

            X = df.values.astype('float64')
            y_pred = self.flow_model.predict(X)

            legit = sum([1 for i in y_pred if i == 0])
            ddos = len(y_pred) - legit

            self.logger.info("------------------------------------------------------------------------------")
            if (legit / len(y_pred) * 100) > 80:
                self.logger.info("Legitimate traffic ...")
            else:
                self.logger.info("DDoS traffic detected ...")
                self.send_alert("DDoS Attack Detected! Mitigation Required.")
            self.logger.info("------------------------------------------------------------------------------")

            with open("PredictFlowStatsfile.csv", "w") as f:
                f.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")

    def send_alert(self, message_body):
        try:
            msg = EmailMessage()
            msg.set_content(message_body)
            msg['Subject'] = 'DDoS Alert Notification'
            msg['From'] = 'ddos.alert@example.com'
            msg['To'] = 'admin@example.com'

            server = smtplib.SMTP('localhost')
            server.send_message(msg)
            server.quit()
            self.logger.info("Alert email sent successfully.")
        except Exception as e:
            self.logger.error(f"Failed to send alert email: {e}")
