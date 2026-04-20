"""
Attack Classification Service
Classifies detected attacks into specific subtypes per network slice
"""

from typing import Dict, Any


class AttackClassifier:
    """Rule-based attack subtype classifier for 6G network slices"""

    def __init__(self):
        # Attack classification rules per dataset
        self.rules = {
            "mMTC": self._classify_mmtc,
            "URLLC": self._classify_urllc,
            "eMBB": self._classify_embb,
            "TON_IoT": self._classify_toniot,
        }

    def classify(self, dataset: str, features: Dict[str, Any], prediction: str) -> str:
        """
        Classify attack subtype based on network slice and features

        Args:
            dataset: Network slice name (mMTC, URLLC, eMBB, TON_IoT)
            features: Feature dictionary
            prediction: Base prediction (Benign/Malicious)

        Returns:
            Attack subtype or 'Benign'
        """
        if prediction == "Benign":
            return "Benign"

        if dataset not in self.rules:
            return "Unknown Attack"

        return self.rules[dataset](features)

    def _classify_mmtc(self, features: Dict[str, Any]) -> str:
        """Classify mMTC (Massive Machine Type Communications) attacks"""
        rate = features.get("Rate", 0)
        tot_pkts = features.get("TotPkts", 0)
        loss = features.get("Loss", 0)
        tcp_rtt = features.get("TcpRtt", 0)

        # DDoS: High packet rate with high loss
        if rate > 100 and loss > 5:
            return "DDoS Attack"

        # Flooding: Very high packet count
        if tot_pkts > 1000:
            return "Flooding Attack"

        # Scanning: Low rate, many packets, high RTT
        if rate < 10 and tot_pkts > 100 and tcp_rtt > 0.1:
            return "Port Scanning"

        # Resource Exhaustion: High loss, low rate
        if loss > 10 and rate < 50:
            return "Resource Exhaustion"

        return "Generic mMTC Attack"

    def _classify_urllc(self, features: Dict[str, Any]) -> str:
        """Classify URLLC (Ultra-Reliable Low Latency) attacks"""
        tcp_rtt = features.get("TcpRtt", 0)
        loss = features.get("Loss", 0)
        rate = features.get("Rate", 0)
        dur = features.get("Dur", 0)

        # Latency Attack: Very high RTT
        if tcp_rtt > 0.5:
            return "Latency Manipulation Attack"

        # Packet Loss Attack: High loss rate
        if loss > 15:
            return "Packet Loss Attack"

        # Timing Attack: Short duration, high rate
        if dur < 0.1 and rate > 200:
            return "Timing Attack"

        # Reliability Attack: Moderate loss, high RTT
        if loss > 5 and tcp_rtt > 0.2:
            return "Reliability Degradation Attack"

        return "Generic URLLC Attack"

    def _classify_embb(self, features: Dict[str, Any]) -> str:
        """Classify eMBB (Enhanced Mobile Broadband) attacks"""
        tot_bytes = features.get("TotBytes", 0)
        rate = features.get("Rate", 0)
        load = features.get("Load", 0)
        loss = features.get("Loss", 0)

        # Bandwidth Exhaustion: Very high bytes and load
        if tot_bytes > 50000 and load > 10000:
            return "Bandwidth Exhaustion Attack"

        # DDoS: High rate with packet loss
        if rate > 150 and loss > 3:
            return "DDoS Attack"

        # Data Exfiltration: High bytes, low loss
        if tot_bytes > 100000 and loss < 1:
            return "Data Exfiltration"

        # Congestion Attack: High load, moderate rate
        if load > 5000 and rate > 50:
            return "Network Congestion Attack"

        return "Generic eMBB Attack"

    def _classify_toniot(self, features: Dict[str, Any]) -> str:
        """Classify TON_IoT (Real-world IoT) attacks"""
        src_bytes = features.get("src_bytes", 0)
        dst_bytes = features.get("dst_bytes", 0)
        src_pkts = features.get("src_pkts", 0)
        duration = features.get("duration", 0)
        proto = features.get("proto", "")
        conn_state = features.get("conn_state", "")
        service = features.get("service", "")

        # DDoS: High packet count, short duration
        if src_pkts > 1000 and duration < 10:
            return "DDoS Attack"

        # Port Scanning: Many connections, low bytes
        if src_pkts > 100 and src_bytes < 1000 and conn_state in ["REJ", "S0"]:
            return "Port Scanning"

        # Ransomware: High data transfer, specific services
        if src_bytes > 100000 and service in ["http", "ftp", "smb"]:
            return "Ransomware"

        # Backdoor: Persistent connection, moderate traffic
        if duration > 300 and src_bytes > 10000:
            return "Backdoor"

        # Injection: Specific protocols with anomalous patterns
        if proto == "tcp" and conn_state == "SF" and src_bytes > dst_bytes * 10:
            return "Injection Attack"

        # MITM: Balanced bidirectional traffic
        if abs(src_bytes - dst_bytes) < 1000 and src_pkts > 50:
            return "Man-in-the-Middle"

        # Password Attack: Many small packets to specific services
        if src_pkts > 50 and src_bytes < 5000 and service in ["ssh", "ftp", "http"]:
            return "Password Attack"

        # XSS: HTTP service with specific patterns
        if service == "http" and src_bytes > 5000:
            return "Cross-Site Scripting (XSS)"

        return "Generic IoT Attack"

    def get_attack_severity(self, attack_type: str) -> str:
        """
        Determine attack severity level

        Args:
            attack_type: Attack classification

        Returns:
            Severity level (Critical/High/Medium/Low)
        """
        critical_attacks = [
            "DDoS Attack",
            "Ransomware",
            "Data Exfiltration",
            "Backdoor",
        ]
        high_attacks = [
            "Flooding Attack",
            "Bandwidth Exhaustion Attack",
            "Injection Attack",
            "Man-in-the-Middle",
        ]
        medium_attacks = [
            "Port Scanning",
            "Password Attack",
            "Latency Manipulation Attack",
            "Packet Loss Attack",
        ]

        if attack_type in critical_attacks:
            return "Critical"
        elif attack_type in high_attacks:
            return "High"
        elif attack_type in medium_attacks:
            return "Medium"
        elif attack_type == "Benign":
            return "None"
        else:
            return "Low"

    def get_recommended_action(self, attack_type: str) -> str:
        """
        Get recommended mitigation action

        Args:
            attack_type: Attack classification

        Returns:
            Recommended action
        """
        actions = {
            "DDoS Attack": (
                "Block source IP, enable rate limiting, " "activate DDoS mitigation"
            ),
            "Ransomware": (
                "Isolate affected systems, block file encryption, "
                "restore from backup"
            ),
            "Port Scanning": (
                "Block source IP, enable firewall rules, "
                "monitor for follow-up attacks"
            ),
            "Data Exfiltration": (
                "Block outbound connection, investigate data breach, "
                "alert security team"
            ),
            "Backdoor": "Terminate connection, scan for malware, reset credentials",
            "Flooding Attack": "Enable traffic filtering, increase bandwidth, block source",
            "Bandwidth Exhaustion Attack": (
                "Implement QoS policies, throttle traffic, block source"
            ),
            "Injection Attack": "Sanitize inputs, update WAF rules, patch vulnerabilities",
            "Man-in-the-Middle": (
                "Enforce encryption, verify certificates, terminate connection"
            ),
            "Password Attack": "Lock account, enable MFA, block source IP",
            "Latency Manipulation Attack": (
                "Reroute traffic, investigate network path, enable monitoring"
            ),
            "Packet Loss Attack": (
                "Check network integrity, reroute traffic, investigate source"
            ),
            "Cross-Site Scripting (XSS)": "Sanitize inputs, update WAF, patch application",
            "Benign": "No action required",
        }

        return actions.get(attack_type, "Investigate further and monitor")


# Singleton instance
_classifier = None


def get_classifier() -> AttackClassifier:
    """Get or create singleton classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = AttackClassifier()
    return _classifier
