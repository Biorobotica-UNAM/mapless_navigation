#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import sys

class DebugManager(Node):
    def __init__(self):
        super().__init__('debug_manager')
        self.node_states = {}               # (node, msg_id) -> dict
        self.node_first_time = {}           # node -> timestamp of first message
        self.node_msgs = {}                 # node -> list of msg_ids in arrival order
        self.sub = self.create_subscription(String, '/debugger/debug_msgs', self.callback, 10)
        self.timer = self.create_timer(0.1, self.render)
        sys.stdout.write("\033[2J\033[H")   # clear screen

    def callback(self, msg):
        try:
            parts = msg.data.split('|')
            if len(parts) == 5: # Updated to 5 parts
                node, msg_id, level, ttl, content = [p.strip() for p in parts]
                key = (node, msg_id)
                
                if node not in self.node_first_time:
                    self.node_first_time[node] = time.time()
                    self.node_msgs[node] = []
                
                if msg_id not in self.node_msgs[node]:
                    self.node_msgs[node].append(msg_id)
                
                self.node_states[key] = {
                    "content": content,
                    "last_seen": time.time(),
                    "level": level.lower(),
                    "ttl": float(ttl) # Store the TTL
                }
        except Exception:
            pass

    def render(self):
        now = time.time()
        sys.stdout.write("\033[H\033[J")

        node_order = sorted(self.node_first_time.keys())
        max_node_len = max((len(node) for node in node_order), default=0)
        node_col_width = max_node_len + 2   

        out = "=" * (node_col_width + 1 + 12 + 1 + 40) + "\n"
        out += f"{'NODE':<{node_col_width}} | {'ID':<12} | MESSAGE\n"
        out += "-" * (node_col_width + 1 + 12 + 1 + 40) + "\n"

        for node in node_order:
            # Grouping logic remains, with an expiration check
            all_ids = self.node_msgs[node]
            
            # Prioritize NODE_STATUS
            if "NODE_STATUS" in all_ids:
                data = self.node_states.get((node, "NODE_STATUS"))
                if data:
                    out += self._format_line(node, "NODE_STATUS", data, now, node_col_width)
            
            for msg_id in all_ids:
                if msg_id == "NODE_STATUS": continue
                
                data = self.node_states.get((node, msg_id))
                if data:
                    # --- EXPIRATION LOGIC ---
                    # If ttl > 0 and time has passed, don't add to output
                    if data["ttl"] > 0 and (now - data["last_seen"]) > data["ttl"]:
                        continue
                    # ------------------------
                    out += self._format_line(node, msg_id, data, now, node_col_width)

        out += "\n(Ctrl+C to stop monitor)"
        sys.stdout.write(out)
        sys.stdout.flush()

    def _format_line(self, node, msg_id, data, now, col_width):
        age = now - data["last_seen"]
        content = data["content"]
        level = data["level"]

        # Colour logic
        if age > 7.0:
            display = "Node deactivated or lost" if msg_id == "NODE_STATUS" else "---"
            status_str = f"\033[91m{display}\033[0m"
        elif age > 5.0:
            tag = "\033[91m\033[7m\033[1m [ LOST ] \033[0m"
            display = "Node deactivated or lost" if msg_id == "NODE_STATUS" else content
            status_str = f"{tag} {display}"
        elif age > 1.5:
            tag = "\033[93m\033[7m\033[1m [Inactive] \033[0m"
            display = "Node is not responding" if msg_id == "NODE_STATUS" else content
            status_str = f"{tag} {display}"
        else:
            if level == "err":
                status_str = f"\033[91m{content}\033[0m"
            elif level == "warn":
                status_str = f"\033[93m{content}\033[0m"
            else:
                status_str = content

        bracketed = f"[{node}]"
        return f"{bracketed:<{col_width}} | {msg_id:<12} | {status_str}\n"

def main(args=None):
    rclpy.init(args=args)
    node = DebugManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()