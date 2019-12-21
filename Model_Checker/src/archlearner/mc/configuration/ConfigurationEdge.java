package archlearner.mc.configuration;

public class ConfigurationEdge {

	private String m_id;
	private String m_from;
	private String m_to;
	private double m_sent, m_received, m_processing;
	
	public ConfigurationEdge (String id, double energySent, double energyReceived, double energyProcessing) {
		m_id = id;
		m_from = id.split("_")[0];
		m_to = id.split("_")[1];
		m_sent = energySent;
		m_received = energyReceived;
		m_processing = energyProcessing;
	}

	public String getM_id() {
		return m_id;
	}

	public void setM_id(String m_id) {
		this.m_id = m_id;
	}

	public String getM_from() {
		return m_from;
	}

	public void setM_from(String m_from) {
		this.m_from = m_from;
	}

	public String getM_to() {
		return m_to;
	}

	public void setM_to(String m_to) {
		this.m_to = m_to;
	}

	public double getM_sent() {
		return m_sent;
	}

	public void setM_sent(double m_sent) {
		this.m_sent = m_sent;
	}

	public double getM_received() {
		return m_received;
	}

	public void setM_received(double m_received) {
		this.m_received = m_received;
	}

	public double getM_processing() {
		return m_processing;
	}

	public void setM_processing(double m_processing) {
		this.m_processing = m_processing;
	}

	@Override
	public String toString() {
		return "ConfigurationEdge [m_id=" + m_id + ", m_from=" + m_from + ", m_to=" + m_to + ", m_sent=" + m_sent
				+ ", m_received=" + m_received + ", m_processing=" + m_processing + "]";
	}
	
	
}
