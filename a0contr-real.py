# controller.py
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, tcp, udp, icmp, in_proto

from datetime import datetime
import os
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        print("Training time: ", (end - start))

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle=0, hard=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    idle_timeout=idle, hard_timeout=hard,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    idle_timeout=idle, hard_timeout=hard,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            self.flow_predict()

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now().timestamp()
        with open("PredictFlowStatsfile.csv", "w") as file0:
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
            for stat in ev.msg.body:
                if stat.priority != 1:
                    continue

                ip_src = stat.match.get('ipv4_src', '0.0.0.0')
                ip_dst = stat.match.get('ipv4_dst', '0.0.0.0')
                ip_proto = stat.match.get('ip_proto', 0)
                icmp_code = stat.match.get('icmpv4_code', -1)
                icmp_type = stat.match.get('icmpv4_type', -1)
                tp_src = stat.match.get('tcp_src', stat.match.get('udp_src', 0))
                tp_dst = stat.match.get('tcp_dst', stat.match.get('udp_dst', 0))

                flow_id = f"{ip_src}{tp_src}{ip_dst}{tp_dst}{ip_proto}"
                try:
                    pps = stat.packet_count / stat.duration_sec if stat.duration_sec else 0
                    ppns = stat.packet_count / stat.duration_nsec if stat.duration_nsec else 0
                    bps = stat.byte_count / stat.duration_sec if stat.duration_sec else 0
                    bpns = stat.byte_count / stat.duration_nsec if stat.duration_nsec else 0
                except:
                    pps = ppns = bps = bpns = 0

                file0.write(f"{timestamp},{ev.msg.datapath.id},{flow_id},{ip_src},{tp_src},{ip_dst},{tp_dst},{ip_proto},{icmp_code},{icmp_type},{stat.duration_sec},{stat.duration_nsec},{stat.idle_timeout},{stat.hard_timeout},{stat.flags},{stat.packet_count},{stat.byte_count},{pps},{ppns},{bps},{bpns}\n")

    def flow_training(self):
        print("Flow Training ...")
        parquet_files = [f for f in os.listdir() if f.endswith(".parquet")]
        df_list = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                df_list.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if not df_list:
            print("No data found!")
            return

        df = pd.concat(df_list, ignore_index=True)
        df.iloc[:, 2] = df.iloc[:, 2].astype(str).str.replace('.', '')
        df.iloc[:, 3] = df.iloc[:, 3].astype(str).str.replace('.', '')
        df.iloc[:, 5] = df.iloc[:, 5].astype(str).str.replace('.', '')

        X = df.iloc[:, :-1].values.astype('float64')
        y = df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        clf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        self.flow_model = clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Training Accuracy: {:.2f}%".format(acc * 100))

    def flow_predict(self):
        try:
            df = pd.read_csv("PredictFlowStatsfile.csv")
            df.iloc[:, 2] = df.iloc[:, 2].astype(str).str.replace('.', '')
            df.iloc[:, 3] = df.iloc[:, 3].astype(str).str.replace('.', '')
            df.iloc[:, 5] = df.iloc[:, 5].astype(str).str.replace('.', '')
            X = df.values.astype('float64')

            y_pred = self.flow_model.predict(X)
            legit = sum(1 for y in y_pred if y == 0)
            ddos = len(y_pred) - legit

            print("------------------------------------------------------------")
            if ddos > 0 and (ddos / len(y_pred)) > 0.2:
                print("⚠️  DDoS Attack Detected! Legit: {}, DDoS: {}".format(legit, ddos))
            else:
                print("✔️  Traffic is Legitimate. Legit: {}, DDoS: {}".format(legit, ddos))
            print("------------------------------------------------------------")
        except Exception as e:
            print("Prediction failed:", e)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD and eth.ethertype == ether_types.ETH_TYPE_IP:
            ip = pkt.get_protocol(ipv4.ipv4)
            protocol = ip.proto
            match_fields = dict(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=ip.src, ipv4_dst=ip.dst, ip_proto=protocol)

            if protocol == in_proto.IPPROTO_ICMP:
                icmp_pkt = pkt.get_protocol(icmp.icmp)
                match_fields.update(icmpv4_code=icmp_pkt.code, icmpv4_type=icmp_pkt.type)
            elif protocol == in_proto.IPPROTO_TCP:
                tcp_pkt = pkt.get_protocol(tcp.tcp)
                match_fields.update(tcp_src=tcp_pkt.src_port, tcp_dst=tcp_pkt.dst_port)
            elif protocol == in_proto.IPPROTO_UDP:
                udp_pkt = pkt.get_protocol(udp.udp)
                match_fields.update(udp_src=udp_pkt.src_port, udp_dst=udp_pkt.dst_port)

            match = parser.OFPMatch(**match_fields)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle=20, hard=100)
                return
            else:
                self.add_flow(datapath, 1, match, actions, idle=20, hard=100)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
