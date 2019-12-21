package archlearner.mc.configuration;

import java.util.HashMap;
import java.util.LinkedList;

import archlearner.mc.configuration.ConfigurationNode.NodeType;


public class Configuration {

	public enum PatternType {SU, SC, CO};
	private HashMap <String, ConfigurationNode> m_nodes = new HashMap<String, ConfigurationNode>();
	private HashMap <String, ConfigurationEdge> m_edges = new HashMap<String, ConfigurationEdge>();
	private PatternType m_type;
	
	
	public void addNode(ConfigurationNode n) {
		if (!m_nodes.containsKey(n.getM_id())) {
			m_nodes.put(n.getM_id(),n);
		}
	}
	
	public ConfigurationNode getNode(String id) {
		return m_nodes.get(id);
	}
	
	public String getDatabaseId() {
		for (HashMap.Entry<String, ConfigurationNode> e : m_nodes.entrySet()) {
			if (e.getValue().getM_type().equals(ConfigurationNode.NodeType.DATABASE))
				return e.getValue().getM_id();
		}
		return ""; // No database found
	}
	
	public String getComputeId() {
		for (HashMap.Entry<String, ConfigurationNode> e : m_nodes.entrySet()) {
			if (e.getValue().getM_type().equals(ConfigurationNode.NodeType.COMPUTE))
				return e.getValue().getM_id();
		}
		return ""; // No database found
	}
	
	public int getNumberOfDisplays() {
		int count=0;
		for (HashMap.Entry<String, ConfigurationNode> e : m_nodes.entrySet()) {
			if (e.getValue().getM_type().equals(ConfigurationNode.NodeType.DISPLAY))
				count++;
		}
		return count;
	}
	
	public Double getEnergyDBDisplays() {
		Double res=0.0;
		LinkedList<ConfigurationEdge> edges = getEdgesToNeighbors(getDatabaseId(),NeighborMode.FROM);
		for (int i=0; i<edges.size(); i++) {
			ConfigurationEdge edge = edges.get(i);
			if (getNode(edge.getM_to()).getM_type().equals(NodeType.DISPLAY))
				res += edge.getM_processing() + edge.getM_sent() + edge.getM_received();
		}
		return res;
	}

	public void addEdge(ConfigurationEdge e) {
		if (!m_edges.containsKey(e.getM_id())) {
			m_edges.put(e.getM_id(), e);
		}
	}
	
	public ConfigurationEdge getEdge(String id) {
		return m_edges.get(id);
	}

	public LinkedList<ConfigurationEdge> getEdgesToNeighbors(String node_id) {
		return getEdgesToNeighbors(node_id, NeighborMode.BOTH);
	}

	public enum NeighborMode { FROM, TO, BOTH };
	public LinkedList<ConfigurationEdge> getEdgesToNeighbors(String node_id, NeighborMode mode) {
		boolean match=false;
		LinkedList<ConfigurationEdge> res = new LinkedList<ConfigurationEdge>();
		for (HashMap.Entry<String, ConfigurationEdge> e : m_edges.entrySet()) {
			switch (mode) {
			case FROM:
				match = e.getValue().getM_from().equals(node_id);
			break;
			case TO:
				match = e.getValue().getM_to().equals(node_id);
			break;
			case BOTH:
				match = e.getValue().getM_from().equals(node_id)||e.getValue().getM_to().equals(node_id);
			break;
			}
			if (match)
				res.add(e.getValue());
		}
		return res;
	}

	
	public HashMap<String, ConfigurationNode> getM_nodes() {
		return m_nodes;
	}

	public HashMap<String, ConfigurationEdge> getM_edges() {
		return m_edges;
	}

	public void setM_edges(HashMap<String, ConfigurationEdge> m_edges) {
		this.m_edges = m_edges;
	}

	public PatternType getM_type() {
		return m_type;
	}

	public void setM_type(PatternType m_type) {
		this.m_type = m_type;
	}

	public void setM_nodes(HashMap<String, ConfigurationNode> m_nodes) {
		this.m_nodes = m_nodes;
	}

	@Override
	public String toString() {
		return "Configuration [m_nodes=" + m_nodes + ", m_edges=" + m_edges + ", m_type=" + m_type + "]";
	}

	
	
	
}