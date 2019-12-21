package archlearner.mc.configuration;

import java.util.LinkedList;

public class ConfigurationNode {
	
	public enum NodeType {SENSOR, CONTROLLER, DATABASE, DISPLAY, COMPUTE };
	public enum Mode {NONE, NORMAL, CRITICAL}
	
	private String m_id;
	private NodeType m_type;
	private Mode m_mode;
	private double m_rate_normal;
	private double m_rate_critical;
	private double m_remaining_energy;
	private double m_energy_idle;
	private LinkedList<SensorForecastPeriod> m_forecast;
	
	public ConfigurationNode(String id, NodeType type) {
		m_id = id;
		m_type = type;
	}
	
	public ConfigurationNode (String id, NodeType type, Mode mode, 
			Double rateNormal, Double rateCritical, Double remainingEnergy, 
			Double energyIdle, LinkedList<SensorForecastPeriod> forecast) {
		m_id = id;
		m_type = type;
		m_mode = mode;
		m_rate_normal = rateNormal;
		m_rate_critical = rateCritical;
		m_remaining_energy = remainingEnergy;
		m_energy_idle = energyIdle;
		m_forecast = forecast;
	}

	public String getM_id() {
		return m_id;
	}

	public void setM_id(String m_id) {
		this.m_id = m_id;
	}
	
	

	public NodeType getM_type() {
		return m_type;
	}

	public void setM_type(NodeType m_type) {
		this.m_type = m_type;
	}

	public double getM_rate_normal() {
		return m_rate_normal;
	}

	public void setM_rate_normal(double m_rate_normal) {
		this.m_rate_normal = m_rate_normal;
	}

	public double getM_rate_critical() {
		return m_rate_critical;
	}

	public void setM_rate_critical(double m_rate_critical) {
		this.m_rate_critical = m_rate_critical;
	}

	public Mode getM_mode() {
		return m_mode;
	}

	public void setM_mode(Mode m_mode) {
		this.m_mode = m_mode;
	}

	public double getM_remaining_energy() {
		return m_remaining_energy;
	}

	public void setM_remaining_energy(double m_remaining_energy) {
		this.m_remaining_energy = m_remaining_energy;
	}

	public double getM_energy_idle() {
		return m_energy_idle;
	}

	public void setM_energy_idle(double m_energy_idle) {
		this.m_energy_idle = m_energy_idle;
	}

	
	public LinkedList<SensorForecastPeriod> getM_forecast() {
		return m_forecast;
	}

	public void setM_forecast(LinkedList<SensorForecastPeriod> m_forecast) {
		this.m_forecast = m_forecast;
	}

	@Override
	public String toString() {
		return "ConfigurationNode [m_id=" + m_id + ", m_type=" + m_type + ", m_mode=" + m_mode + ", m_rate_normal="
				+ m_rate_normal + ", m_rate_critical=" + m_rate_critical + ", m_remaining_energy=" + m_remaining_energy
				+ ", m_energy_idle=" + m_energy_idle + ", m_forecast=" + m_forecast + "]";
	}
	
	
}
